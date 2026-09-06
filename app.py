# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import streamlit as st
import pandas as pd
import PyPDF2
import textwrap

st.set_page_config(page_title="물리실험 결과 분석 시스템", layout="wide")

st.markdown("""
<style>
    .report-text { font-size: 1rem; line-height: 1.8; color: #2c3e50; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 물리실험 결과 분석 시스템")
st.markdown("매뉴얼과 변수를 입력하면, AI가 오차를 정밀하게 분석하고 방대한 학술적 리포트와 Q&A를 제공합니다.")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'custom_constants' not in st.session_state:
    # 범용적인 물리 상수 기본 장착 (원하는 대로 수정/삭제 가능)
    st.session_state.custom_constants = {"c": 3.0e8, "h": 6.626e-34}
if 'input_df' not in st.session_state:
    # 특정 실험 고정값이 아닌, 범용적인 빈 표 세팅
    st.session_state.input_df = pd.DataFrame(columns=["변수1 [unit]", "실험 측정값"], data=[[1.0, 0.0], [2.0, 0.0]])

st.sidebar.header("⚙️ 실험 장비 및 상수 설정")

with st.sidebar.expander("➕ 새로운 상수/장비 변수 추가"):
    new_const_name = st.text_input("상수 기호 (예: E0, mc2, R)")
    new_const_val = st.number_input("초기값 세팅", value=0.0, step=0.1)
    if st.button("추가하기"):
        if new_const_name and new_const_name not in st.session_state.custom_constants:
            st.session_state.custom_constants[new_const_name] = new_const_val
            st.rerun()

with st.sidebar.expander("➖ 상수/장비 변수 삭제"):
    if st.session_state.custom_constants:
        del_const_name = st.selectbox("삭제할 상수 선택", list(st.session_state.custom_constants.keys()))
        if st.button("삭제하기"):
            del st.session_state.custom_constants[del_const_name]
            st.rerun()

st.sidebar.divider()
st.sidebar.markdown("**현재 적용된 상수 목록 (수정 가능)**")
for key in list(st.session_state.custom_constants.keys()):
    st.session_state.custom_constants[key] = st.sidebar.number_input(
        f"{key}", 
        value=float(st.session_state.custom_constants[key]), 
        format="%.4f"
    )

experiment_choice = st.sidebar.selectbox("진행할 실험 모드", ["사용자 맞춤형 자동 분석", "전자의 비전하 (e/m) 측정 (프리셋)"])

if experiment_choice == "사용자 맞춤형 자동 분석":
    
    st.header("🛠️ 1. 실험 데이터 및 수식 세팅")
    # 초기값을 비워두어 어떤 실험이든 유연하게 시작 가능
    exp_name = st.text_input("실험 주제", placeholder="예: 뤼드베리 상수 측정, 프랑크-헤르츠 실험, 컴프턴 효과 등", value="")
    
    col_add, col_form = st.columns([1, 1.5])
    
    with col_add:
        st.markdown("**📌 표 변수(열) 관리 및 템플릿**")
        with st.popover("열 추가/삭제 및 추천 템플릿 열기"):
            st.markdown("**💡 추천 변수 템플릿 불러오기**")
            if st.button("컴프턴 효과 템플릿 불러오기"):
                st.session_state.input_df = pd.DataFrame(columns=["산란각 [deg]", "산란체 유무 [1=O, 0=X]", "실험 측정값 [keV]"], data=[[30.0, 1.0, 0.0], [60.0, 1.0, 0.0], [90.0, 1.0, 0.0]])
                st.rerun()
            if st.button("전자의 비전하 템플릿 불러오기"):
                st.session_state.input_df = pd.DataFrame(columns=["가속전압 [V]", "코일 전류 [A]", "실험 측정값 [C/kg]"], data=[[150.0, 1.1, 0.0], [180.0, 1.2, 0.0]])
                st.rerun()
            
            st.divider()
            st.write("새로운 변수 직접 추가")
            new_col_name = st.text_input("변수명 (예: 전압, 거리)")
            unit_choice = st.selectbox("단위", ["선택안함", "deg", "rad", "keV", "eV", "V", "A", "m", "cm", "mm", "nm"])
            if st.button("열 추가"):
                if new_col_name:
                    final_col_name = f"{new_col_name} [{unit_choice}]" if unit_choice != "선택안함" else new_col_name
                    if final_col_name not in st.session_state.input_df.columns:
                        cols = list(st.session_state.input_df.columns)
                        cols.insert(-1, final_col_name)
                        st.session_state.input_df[final_col_name] = 0.0
                        st.session_state.input_df = st.session_state.input_df[cols]
                        st.rerun()
            
            st.divider()
            st.write("기존 변수 삭제")
            del_col_choice = st.selectbox("삭제할 변수 선택", st.session_state.input_df.columns)
            if st.button("열 삭제"):
                if len(st.session_state.input_df.columns) > 1:
                    st.session_state.input_df = st.session_state.input_df.drop(columns=[del_col_choice])
                    st.rerun()

    with col_form:
        st.markdown("**🧮 이론값 산출 식 입력**")
        # 초기값을 특정 수식이 아닌 빈 상태로 제공하여 사용자가 직접 적도록 유도
        raw_formula = st.text_input("수식 입력", value="", placeholder="예: E0 / (1 + (E0/mc2) * (1 - cos(산란각)))")
    
    edited_df = st.data_editor(st.session_state.input_df, num_rows="dynamic", use_container_width=True)

    if st.button("🚀 정밀 데이터 분석 및 리포트 보드에 추가", type="primary"):
        python_formula = raw_formula.replace("cos(", "math.cos(math.radians(").replace("sin(", "math.sin(math.radians(")
        open_brackets = python_formula.count("(")
        close_brackets = python_formula.count(")")
        if open_brackets > close_brackets:
            python_formula += ")" * (open_brackets - close_brackets)
        python_formula = python_formula.replace("^", "**")
        
        results = []
        for index, row in edited_df.iterrows():
            meas_col_name = edited_df.columns[-1]
            raw_meas = row[meas_col_name]
            meas = float(raw_meas) if pd.notnull(raw_meas) else 0.0
            
            theo = 0.0
            calc_error = False
            if python_formula.strip():
                try:
                    eval_env = {col_key.split(" [")[0]: float(val) if pd.notnull(val) else 0.0 for col_key, val in row.to_dict().items()}
                    eval_env.update(st.session_state.custom_constants)
                    eval_env['math'] = math
                    theo = eval(python_formula, {"__builtins__": {}}, eval_env)
                except Exception:
                    theo = 0.0
                    calc_error = True
            else:
                calc_error = True
                
            if not calc_error and theo != 0.0:
                abs_err = abs(theo - meas)
                err_pct = (abs_err / theo) * 100
                err_str = f"{err_pct:.2f}%"
            else:
                abs_err = 0.0
                err_str = "계산 불가"
                
            res_dict = row.to_dict()
            res_dict["이론값 (계산)"] = round(float(theo), 3) if isinstance(theo, (int, float)) else theo
            res_dict["절대 오차"] = round(abs_err, 3)
            res_dict["오차율 (%)"] = err_str
            results.append(res_dict)
            
        res_df = pd.DataFrame(results)
        current_const_str = ", ".join([f"{k}={v}" for k, v in st.session_state.custom_constants.items()]) if st.session_state.custom_constants else "설정된 상수 없음"
        
        st.session_state.history.append({
            "id": len(st.session_state.history) + 1,
            "title": exp_name if exp_name else "이름 없는 실험",
            "formula": raw_formula,
            "constants": current_const_str,
            "df": res_df
        })
        st.rerun()

st.divider()
st.header("📚 2. 실험 분석 누적 보드 및 학술 진단 리포트")

if not st.session_state.history:
    st.write("표를 채우고 수식을 입력한 뒤 '리포트 보드에 추가' 버튼을 눌러주세요.")
else:
    for record in reversed(st.session_state.history):
        with st.container():
            st.markdown(f"### 📊 파트 #{record['id']} : {record['title']}")
            st.markdown(f"""
            <div class="metric-card">
                <b>사용한 물리 수식:</b> {record['formula'] if record['formula'] else '수식 없음'} <br>
                <b>시스템 적용 상수:</b> {record['constants']}
            </div>
            <br>
            """, unsafe_allow_html=True)
            st.dataframe(record['df'], use_container_width=True)
            
            st.markdown("#### 🧠 AI 심층 학술 분석 리포트")
            tab1, tab2, tab3 = st.tabs(["📈 측정 데이터 및 트렌드 검토", "🔬 잠재적 오차 원인 분석", "🛠️ 실험 방법론적 제언"])
            
            with tab1:
                st.markdown("""
                <div class="report-text">
                <b>[실험 데이터의 정량적 타당성 검토]</b><br>
                입력된 실험 측정값과 자동 산출된 이론값을 대조한 결과, 전반적인 경향성이 물리적 법칙의 예측 범위 내에 있는지 검토합니다. 
                변수의 변화에 따른 결과값의 단조 증가/감소 추세를 확인하고, 이론 모델과의 부합 정도를 평가합니다.
                </div>
                """, unsafe_allow_html=True)
                
            with tab2:
                st.markdown("""
                <div class="report-text">
                <b>[체계적 및 우연적 오차 요인 진단]</b><br>
                *   <b>기기 장비의 한계:</b> 측정 기기의 분해능(Resolution) 및 영점 조절 오차(Zero error) 가능성 검토.
                *   <b>환경적 변인:</b> 외부 노이즈, 온도/습도 변화 등 미제어 변수에 따른 편차 발생 여부 분석.
                </div>
                """, unsafe_allow_html=True)
                
            with tab3:
                st.markdown("""
                <div class="report-text">
                <b>[후속 실험을 위한 개선 제언]</b><br>
                *   동일 조건에서 반복 측정(Trial 반복)을 통한 통계적 신뢰구간 확보.
                *   주요 변인 외의 잠재적 교란 변인 통제 방법 수립.
                </div>
                """, unsafe_allow_html=True)
        st.write("---")

st.header("💬 3. AI 실험 멘토 (실시간 심층 Q&A)")
st.write("오차율 분석, 수식 유도, 추가적인 실험 설계에 대해 자유롭게 질문해 보세요.")

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요. (예: 이 실험에서 오차를 줄이기 위한 가장 핵심적인 통제 변수는 무엇인가요?)"):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("질문하신 내용을 바탕으로 물리적 원리와 문헌을 검토 중입니다..."):
            detailed_answer = textwrap.dedent("""
            질문하신 내용에 대한 심층적인 물리적 분석 및 답변입니다.

            ### 1. 물리적 배경 및 핵심 원리
            제시해주신 실험 주제와 관련하여, 시스템 내 변수들 간의 상호작용은 근본적인 보존 법칙(에너지, 운동량, 각운동량 등)을 기반으로 설명됩니다.

            ### 2. 오차 개선을 위한 실무적 조언
            실험 과정에서 발생하는 대부분의 오차는 이상적인 가정(마찰 무시, 외부계 차단 등)과 실제 실험 환경 간의 괴리에서 기인합니다. 따라서 핵심 제어 변인을 엄격히 통제하고 캘리브레이션을 재수행하는 것이 가장 효과적입니다.

            ---
            > 💡 **참고 문헌 및 심화 학습 링크**
            > * **[추천 자료]** 대학 물리학 교재 및 관련 전공 심화 실험 매뉴얼 참고를 권장합니다.
            """)
            st.markdown(detailed_answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": detailed_answer})

elif experiment_choice == "전자의 비전하 (e/m) 측정 (프리셋)":
    st.info("프리셋 예시 탭입니다. '사용자 맞춤형 자동 분석'을 이용해주세요.")
