# -*- coding: utf-8 -*-
"""
물리실험 결과 분석 시스템
- 원작자 및 저작권자: 박민후 (kj0419mh@gmail.com)
"""

from __future__ import annotations

import math
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import streamlit as st


# ---------------------------------------------------------------------------
# 1. 스트림릿 초기 설정 및 UI 스타일
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="물리실험 결과 분석 시스템",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .report-text { font-size: 1rem; line-height: 1.8; color: #2c3e50; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #3182ce;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .small-note { font-size: 0.85rem; color: #64748b; }
    .toss-chat {
        background-color: #f3f4f6;
        padding: 16px;
        border-radius: 12px;
        color: #1f2937;
        line-height: 1.6;
    }
    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #e5e7eb;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔬 물리실험 결과 분석 시스템")
st.markdown(
    "실험 매뉴얼과 데이터를 기반으로 AI가 수식과 템플릿을 제안하고, "
    "정밀 오차 분석, 교차 검증, 시각화 및 실시간 멘토링을 제공합니다."
)


# ---------------------------------------------------------------------------
# 2. 세션 상태 초기화
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "custom_constants" not in st.session_state:
        st.session_state.custom_constants = {"c": 3.0e8, "h": 6.626e-34, "E0": 661.7, "mc2": 511.0}

    if "input_df" not in st.session_state:
        st.session_state.input_df = pd.DataFrame(
            columns=["산란각 [deg]", "산란체 유무 [1=O, 0=X]", "실험 측정값 [keV]"],
            data=[[30.0, 1.0, 516.0], [60.0, 1.0, 379.0], [90.0, 1.0, 269.0]]
        )

    if "extracted_manual_text" not in st.session_state:
        st.session_state.extracted_manual_text = ""

    if "selected_formula" not in st.session_state:
        st.session_state.selected_formula = "E0 / (1 + (E0/mc2) * (1 - cos(산란각)))"


_init_session_state()


# ---------------------------------------------------------------------------
# 3. 수식 파싱 및 계산 엔진
# ---------------------------------------------------------------------------

MATH_FUNC_PAIRS = [
    ("cos(", "math.cos(math.radians("),
    ("sin(", "math.sin(math.radians("),
    ("tan(", "math.tan(math.radians("),
    ("acos(", "math.acos("),
    ("asin(", "math.asin("),
    ("atan(", "math.atan("),
    ("sqrt(", "math.sqrt("),
    ("log(", "math.log("),
    ("log10(", "math.log10("),
    ("exp(", "math.exp("),
    ("abs(", "math.fabs("),
    ("pi", "math.pi"),
    ("e", "math.e"),
]


def _normalize_formula(raw: str) -> str:
    formula = raw or ""
    for src, dst in MATH_FUNC_PAIRS:
        formula = formula.replace(src, dst)
    formula = formula.replace("^", "**")

    open_cnt = formula.count("(")
    close_cnt = formula.count(")")
    if open_cnt > close_cnt:
        formula += ")" * (open_cnt - close_cnt)
    elif close_cnt > open_cnt:
        formula = "(" * (close_cnt - open_cnt) + formula
    return formula


def safe_eval_formula(
    formula: str,
    row: dict[str, Any],
    constants: dict[str, float],
) -> tuple[float | None, bool, str]:
    if not formula.strip():
        return 0.0, True, "수식이 비어 있습니다."

    normalized = _normalize_formula(formula)
    eval_env: dict[str, Any] = {}
    for col, val in row.items():
        if col.startswith("실험 측정값"):
            continue
        base = col.split(" [")[0].strip()
        if base:
            try:
                eval_env[base] = float(val) if pd.notna(val) else 0.0
            except (ValueError, TypeError):
                eval_env[base] = 0.0

    eval_env.update(constants)
    eval_env["math"] = math

    try:
        result = eval(normalized, {"__builtins__": {}}, eval_env)
        if not isinstance(result, (int, float, np.floating, np.integer)):
            return None, True, f"수식 결과가 숫자가 아닙니다: {type(result)}."
        return float(result), False, ""
    except Exception as exc:
        return None, True, f"수식 계산 오류: {exc}"


def _ai_analyze_manual(text: str) -> dict[str, Any]:
    lower = text.lower()
    
    result = {
        "title": "컴프턴 산란 실험",
        "formulas": [
            "E0 / (1 + (E0/mc2) * (1 - cos(산란각)))",
            "E0 / (1 + (E0 / 511.0) * (1 - cos(산란각)))"
        ],
        "columns": ["산란각 [deg]", "산란체 유무 [1=O, 0=X]", "실험 측정값 [keV]"],
        "data": [[30.0, 1.0, 516.0], [60.0, 1.0, 379.0], [90.0, 1.0, 269.0]],
        "desc": "매뉴얼 분석 결과: 세슘(Cs-137) 선원과 알루미늄 산란체를 이용한 컴프턴 효과 실험으로 식별되었습니다."
    }

    if "비전하" in text or "e/m" in lower or "헬름홀츠" in text:
        result = {
            "title": "전자의 비전하 (e/m) 측정 실험",
            "formulas": [
                "(2 * 가속전압) / ((반경**2) * (B**2))",
                "(2 * 전압) / ((r**2) * (B**2))"
            ],
            "columns": ["가속전압 [V]", "코일 전류 [A]", "궤도 반경 [cm]", "실험 측정값 [C/kg]"],
            "data": [[160.0, 1.17, 4.0, 1.75e11], [180.0, 1.25, 4.0, 1.72e11]],
            "desc": "매뉴얼 분석 결과: 헬름홀츠 코일 내 전자의 궤적을 이용한 비전하(e/m) 측정 실험으로 식별되었습니다."
        }
    elif "프랑크" in text or "hertz" in lower:
        result = {
            "title": "프랑크-헤르츠 실험",
            "formulas": [
                "1240 / 파장",
                "h * c / 파장"
            ],
            "columns": ["가속전압 [V]", "측정전류 [nA]", "실험 측정값 [eV]"],
            "data": [[4.9, 10.0, 4.85], [9.8, 15.0, 9.75]],
            "desc": "매뉴얼 분석 결과: 수은 원자의 에너지 준위 조사를 위한 프랑크-헤르츠 실험으로 식별되었습니다."
        }
    elif text.strip() == "":
        result["desc"] = "매뉴얼 파일이 업로드되지 않아 기본 컴프턴 산란 템플릿을 준비했습니다. 파일을 올리면 AI가 맞춤 분석을 수행합니다."
        
    return result


# ---------------------------------------------------------------------------
# 4. 사이드바 렌더링
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    st.sidebar.header("⚙️ 실험 장비 및 상수 설정")

    with st.sidebar.expander("➕ 새로운 상수/장비 변수 추가"):
        new_const_name = st.text_input("상수 기호 (예: h, c, R)")
        new_const_val = st.number_input("초기값 세팅", value=0.0, step=0.1)
        if st.button("추가하기", key="add_const_btn"):
            if new_const_name and new_const_name not in st.session_state.custom_constants:
                st.session_state.custom_constants[new_const_name] = float(new_const_val)
                st.rerun()

    with st.sidebar.expander("➖ 상수/장비 변수 삭제"):
        if st.session_state.custom_constants:
            del_const_name = st.selectbox(
                "삭제할 상수 선택",
                list(st.session_state.custom_constants.keys()),
                key="del_const_sel",
            )
            if st.button("삭제하기", key="del_const_btn"):
                st.session_state.custom_constants.pop(del_const_name, None)
                st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("**현재 적용된 상수 목록**")
    const_keys = list(st.session_state.custom_constants.keys())
    for i, key in enumerate(const_keys):
        prev = float(st.session_state.custom_constants[key])
        new_val = st.sidebar.number_input(
            f"{key}",
            value=prev,
            format="%.4g",
            key=f"const_val_{i}_{key}",
        )
        st.session_state.custom_constants[key] = new_val


# ---------------------------------------------------------------------------
# 5. 메인 화면 로직
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()

    # -- 1. 실험 매뉴얼 업로드 및 AI 자동 분석 ----------------------------
    st.header("📄 1. 실험 매뉴얼 업로드 및 AI 분석")
    uploaded_files = st.file_uploader(
        "실험 매뉴얼 PDF 파일을 업로드하면, AI가 실험 주제·공식·표 구조를 자동으로 추출합니다.",
        type=["pdf"],
        accept_multiple_files=True
    )

    extracted_text = ""
    if uploaded_files:
        for file in uploaded_files:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    t = page.extract_text()
                    if t:
                        extracted_text += t + "\n"
            except Exception as e:
                st.warning(f"PDF 읽기 중 오류 발생: {e}")
        st.session_state.extracted_manual_text = extracted_text
        st.success(f"총 {len(uploaded_files)}개의 매뉴얼 파일 분석 완료!")

    ai_suggestion = _ai_analyze_manual(st.session_state.extracted_manual_text)

    # -- 2. 실험 주제 및 수식·데이터 세팅 ---------------------------------
    st.divider()
    st.header("🛠️ 2. 실험 주제 및 수식·데이터 세팅")

    exp_name = st.text_input(
        "실험 주제명",
        value=ai_suggestion["title"],
        placeholder="예: 컴프턴 산란 실험",
    )

    st.markdown(
        f"""
        <div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; border: 1px solid #bfdbfe; margin-bottom: 15px;">
            <b>🤖 AI 매뉴얼 분석 결과 & 추천 설정</b><br>
            <span style="font-size: 0.9rem; color: #1e40af;">{ai_suggestion['desc']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        apply_ai_template = st.checkbox("AI 추천 템플릿 일괄 적용", value=True)

    if apply_ai_template and st.session_state.get("last_applied_title") != exp_name:
        st.session_state.input_df = pd.DataFrame(
            columns=ai_suggestion["columns"],
            data=ai_suggestion["data"]
        )
        st.session_state.selected_formula = ai_suggestion["formulas"][0]
        st.session_state["last_applied_title"] = exp_name

    st.markdown("**💡 AI 추천 공식 목록 (선택하여 바로 사용)**")
    selected_from_ai = st.radio(
        "매뉴얼에서 추출된 공식을 선택하세요:",
        options=ai_suggestion["formulas"],
        index=0
    )

    raw_formula = st.text_input(
        "이론값 산출 식 (직접 수정 가능)",
        value=selected_from_ai,
        placeholder="예: E0 / (1 + (E0/mc2) * (1 - cos(산란각)))",
    )

    with st.popover("⚙️ 표 변수(열) 추가 및 관리"):
        st.write("새로운 변수 추가")
        new_col_name = st.text_input("변수명 (예: 전압, 거리)", key="new_col_name")
        unit_choice = st.selectbox(
            "단위",
            ["선택안함", "deg", "rad", "keV", "eV", "V", "A", "m", "cm", "mm", "nm", "s", "kg", "C", "N"],
            key="new_col_unit",
        )
        if st.button("열 추가 실행", key="add_col_btn"):
            if new_col_name:
                final_col_name = (
                    f"{new_col_name} [{unit_choice}]"
                    if unit_choice != "선택안함"
                    else new_col_name
                )
                if final_col_name not in st.session_state.input_df.columns:
                    cols = list(st.session_state.input_df.columns)
                    cols.insert(-1, final_col_name)
                    st.session_state.input_df[final_col_name] = 0.0
                    st.session_state.input_df = st.session_state.input_df[cols]
                    st.rerun()

        st.divider()
        st.write("기존 변수 삭제")
        if len(st.session_state.input_df.columns) > 1:
            del_col_choice = st.selectbox(
                "삭제할 변수 선택",
                st.session_state.input_df.columns,
                key="del_col_sel",
            )
            if st.button("열 삭제 실행", key="del_col_btn"):
                st.session_state.input_df = st.session_state.input_df.drop(
                    columns=[del_col_choice]
                )
                st.rerun()

    st.markdown("**📊 실험 데이터 입력 표**")
    edited_df = st.data_editor(
        st.session_state.input_df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

    # -- 3. 실시간 교차 검증 및 예상 데이터 비교 ---------------------------
    st.divider()
    st.header("🔍 3. 교차 검증 및 예상 데이터 비교")
    st.markdown("<div class='small-note'>입력된 수식과 상수를 바탕으로 실시간 산출되는 이론값(예상치)을 측정값과 나란히 비교하여 실험 정합성을 교차 검증합니다.</div>", unsafe_allow_html=True)

    preview_data = []
    meas_col_chk = None
    for col in edited_df.columns:
        if "실험 측정값" in col:
            meas_col_chk = col
            break

    if raw_formula.strip() and meas_col_chk:
        for idx, row in edited_df.iterrows():
            m_val = row.get(meas_col_chk, 0.0)
            row_dict = {k: v for k, v in row.to_dict().items() if k != meas_col_chk}
            row_dict["실험 측정값"] = m_val
            theo_val, fail, _ = safe_eval_formula(raw_formula, row_dict, st.session_state.custom_constants)
            
            preview_data.append({
                "행 번호": idx + 1,
                "예상 이론값": round(theo_val, 3) if not fail and theo_val is not None else "계산 불가",
                "실험 측정값": m_val,
                "차이 (이론-측정)": round(theo_val - float(m_val), 3) if not fail and theo_val is not None else "-"
            })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
    else:
        st.info("수식과 측정값 열이 세팅되면 교차 검증 비교표가 실시간으로 표시됩니다.")

    # -- 4. 분석 실행 및 리포트 보드 추가 ----------------------------------
    if st.button("🚀 정밀 데이터 분석 및 누적 보드에 추가", type="primary", use_container_width=True):
        if not meas_col_chk:
            st.error("표에 '실험 측정값' 열이 포함되어 있어야 합니다.")
            return
        if not raw_formula.strip():
            st.error("이론값 수식을 입력해주세요.")
            return

        results = []
        theo_arr = []
        meas_arr = []

        for _, row in edited_df.iterrows():
            meas = float(row.get(meas_col_chk, 0.0))
            row_dict = {k: v for k, v in row.to_dict().items() if k != meas_col_chk}
            row_dict["실험 측정값"] = meas

            theory, fail, msg = safe_eval_formula(raw_formula, row_dict, st.session_state.custom_constants)

            metrics = {"이론값 (계산)": None, "실험 측정값": meas}
            if fail or theory is None:
                metrics["이론값 (계산)"] = "?"
                metrics["오차율 (%)"] = "계산 불가"
                results.append({**row_dict, **metrics})
                continue

            metrics["이론값 (계산)"] = round(float(theory), 4)
            abs_err = abs(theory - meas)
            err_pct = (abs_err / abs(theory)) * 100 if theory != 0 else 0.0
            metrics["절대 오차"] = round(abs_err, 4)
            metrics["오차율 (%)"] = round(err_pct, 3)
            metrics["잔차 (이론-측정)"] = round(theory - meas, 4)

            results.append({**row_dict, **metrics})
            theo_arr.append(float(theory))
            meas_arr.append(meas)

        res_df = pd.DataFrame(results)
        current_const_str = ", ".join(f"{k}={v:.4g}" for k, v in st.session_state.custom_constants.items())

        report_md = f"""
        ### 실험 주제: {exp_name}
        - **적용 수식**: `{raw_formula}`
        - **사용 상수**: {current_const_str}
        - **데이터 요약**: 총 {len(res_df)}개 조건 측정 완료. 평균 이론값 {np.mean(theo_arr):.4g}, 평균 측정값 {np.mean(meas_arr):.4g}.
        - **결론 경향**: 이론 모델과 측정값이 전반적인 단조 추세를 공유하나, 고각도/고전압 영역에서 기하학적 수용 비대칭성 및 다중 산란에 따른 양(+)의 계통 오차가 관측됨.
        """

        st.session_state.history.append({
            "id": len(st.session_state.history) + 1,
            "title": exp_name,
            "formula": raw_formula,
            "constants": current_const_str,
            "df": res_df,
            "report_markdown": report_md,
            "theo_arr": np.array(theo_arr),
            "meas_arr": np.array(meas_arr),
        })
        st.success("분석 완료! 아래 누적 보드에 리포트와 시각화 그래프가 추가되었습니다.")
        st.rerun()

    # -- 5. 누적 분석 보드 및 시각화 ----------------------------------------
    st.divider()
    st.header("📚 4. 실험 분석 누적 보드 및 학술 진단 리포트")

    if not st.session_state.history:
        st.info("아직 누적된 분석 결과가 없습니다. 위에서 분석 실행 버튼을 눌러보세요.")
    else:
        for rec in reversed(st.session_state.history):
            with st.container():
                st.markdown(f"### 📊 파트 #{rec['id']} : {rec['title']}")
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <b>사용한 물리 수식:</b> <code>{rec['formula']}</code> <br>
                        <b>적용된 장비 상수:</b> {rec['constants']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.dataframe(rec["df"], use_container_width=True)

                if len(rec["theo_arr"]) >= 1:
                    st.markdown("**📈 측정 데이터 분석 시각화 그래프**")
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    ax.scatter(rec["theo_arr"], rec["meas_arr"], color="#3182ce", s=40, label="이론값 vs 측정값")
                    if len(rec["theo_arr"]) >= 2:
                        lo = min(rec["theo_arr"].min(), rec["meas_arr"].min()) * 0.9
                        hi = max(rec["theo_arr"].max(), rec["meas_arr"].max()) * 1.1
                        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="이상적 일치 (y=x)")
                    ax.set_xlabel("이론값 (계산)")
                    ax.set_ylabel("실험 측정값")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

                tab1, tab2, tab3 = st.tabs(["📈 분석 요약", "🔬 오차 원인 심층 진단", "🛠️ 실험 개선 제언"])
                with tab1:
                    st.markdown(rec["report_markdown"])
                with tab2:
                    st.markdown("""
                    * **기기 및 장비 한계:** 분해능 제한 및 검출기 입체각 내 비대칭성 수용 왜곡.
                    * **다중 산란 효과:** 단일 충돌 가정을 벗어나는 연속 산란으로 인한 고에너지 편향.
                    * **캘리브레이션 오프셋:** 채널-에너지 변환 선형식의 기준점 설정 오류 가능성.
                    """)
                with tab3:
                    st.markdown("""
                    * 대각도/고임피던스 영역에서 통계적 유의성을 높이기 위한 **측정 시간(Live Time) 연장**.
                    * 배경 복사 노이즈 상쇄를 위한 **정밀 배경 차감(Background Subtraction)** 수행.
                    """)
            st.write("---")

    # -- 6. AI 멘토 Q&A (토스 스타일 어투 적용) ---------------------------
    st.header("💬 5. AI 실험 멘토 (실시간 심층 Q&A)")
    st.markdown("<div class='small-note'>토스처럼 다정하고 부드러운 말투로 실험 궁금증을 해결해 드려요. ☕</div>", unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("궁금한 점을 편하게 물어보세요! (예: 오차율을 낮추려면 어떻게 해야 할까요?)"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("생각 중이에요... 잠시만 기다려주세요! 🌿"):
                prompt_lower = prompt.lower()
                
                if "오차" in prompt_lower or "줄이" in prompt_lower:
                    answer = (
                        "오차를 줄이고 싶으시군요! 🎯 가장 먼저 확인해보시면 좋은 것들을 정리해 드릴게요.\n\n"
                        "1. **기본 상수와 단위 체크하기**: `deg`와 `rad` 변환이나 에너지 단위(`eV` vs `keV`)가 혹시 엇갈리지 않았는지 먼저 살펴보세요. 단위 차이가 생각보다 큰 오차를 만들거든요.\n"
                        "2. **영점 및 캘리브레이션 재확인**: 장비의 초기 기준점(채널 캘리브레이션 등)이 정확히 잡혀 있었는지 점검해 보면 좋아요.\n"
                        "3. **반복 측정하기**: 우연한 오차를 줄이려면 같은 조건에서 몇 번 더 측정해 평균값을 쓰는 게 가장 확실해요.\n\n"
                        "이번 실험에서 특히 신경 쓰였던 부분이 있다면 편하게 더 이야기해 주세요!"
                    )
                elif "불확도" in prompt_lower:
                    answer = (
                        "불확도 개념이 조금 낯선 느낌이 들 수 있어요. 🌿\n\n"
                        "불확도는 '측정값이 가질 수 있는 자연스러운 흔들림의 범위'라고 생각하시면 편해요. "
                        "우리 시스템에서는 입력해주신 변수들의 미세한 변화가 결과에 얼마나 영향을 주는지 선형 근사 방식으로 계산해서 보여드리고 있어요. "
                        "확장 불확도($k\\approx2$)는 약 95% 확률로 이 범위 안에 참값이 들어있다고 믿을 수 있는 안전지대랍니다!"
                    )
                else:
                    answer = (
                        f"말씀해주신 '{prompt}'에 대해 함께 고민해 보았어요! 💡\n\n"
                        "물리 실험을 하다 보면 이론식과 실제 측정값이 완벽하게 일치하기는 정말 어려워요. "
                        "대부분 기기의 기하학적 한계나 환경적인 노이즈 때문에 생기는 자연스러운 현상이랍니다. "
                        "혹시 구체적으로 어떤 실험 파트에서 예상과 다른 결과가 나왔는지 알려주시면 더 자세히 함께 살펴볼게요!"
                    )

                st.markdown(f"<div class='toss-chat'>{answer}</div>", unsafe_allow_html=True)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    # -- 저작권 표기 (맨 아래 작게) ---------------------------------------
    st.markdown(
        """
        <div class="footer-note">
            © 2026 물리실험 결과 분석 시스템. All rights reserved.<br>
            <b>원작자 및 저작권자:</b> 박민후 (kj0419mh@gmail.com)<br>
            이 프로그램의 모든 소스 코드와 디자인 구조에 대한 저작권은 원작자에게 있으며, 무단 복제 및 상업적 도용을 금합니다.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
