# -*- coding: utf-8 -*-
"""
물리실험 결과 분석 시스템
- 이론값 ↔ 실험값 오차 자동 계산
- 수식 파싱/검증/오차 전파/통계·시각화 패널을 내장
- 스트림릿에서 바로 실행 가능한 단일 파일 버전
"""

from __future__ import annotations

import base64
import io
import math
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# 1. 스트림릿 초기 설정
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
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #0d6efd;
    }
    .small-note { font-size: 0.85rem; color: #555; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔬 물리실험 결과 분석 시스템")
st.markdown(
    "실험 데이터와 수식을 입력하면 이론값·측정값의 오차를 계산하고, "
    "오차 지표 추천, 그래프, 불확도 추정, 실험 개선 조언을 함께 제공합니다."
)


# ---------------------------------------------------------------------------
# 2. 세션 상태 초기화
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """세션에 필요한 키들을 한 번에 초기화한다."""
    if "history" not in st.session_state:
        st.session_state.history: list[dict[str, Any]] = []

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages: list[dict[str, str]] = []

    if "custom_constants" not in st.session_state:
        # 범용 상수 기본 탑재 (자유롭게 수정·삭제 가능)
        st.session_state.custom_constants = {"c": 3.0e8, "h": 6.626e-34}

    if "input_df" not in st.session_state:
        # 측정값 전용 열 + 설명용 열이 기본 구조
        st.session_state.input_df = pd.DataFrame(
            columns=["변수1 [unit]", "실험 측정값"],
            data=[[1.0, 0.0], [2.0, 0.0]],
        )

    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "사용자 맞춤형 자동 분석"


_init_session_state()


# ---------------------------------------------------------------------------
# 3. 수식 파싱·검증·계산 / 불확도·지표 계산 (모듈화된 핵심 로직)
# ---------------------------------------------------------------------------

# -- (a) 수식 안의 함수 이름 치환 -------------------------------------------------
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
    """사용자 수식을 eval-safe 형태로 다듬는다.

    - 함수명 치환
    - ^ → **
    - 괄호 개수 보정 (심화 파싱이 아니므로 안전장치로만 사용)
    """
    formula = raw or ""
    for src, dst in MATH_FUNC_PAIRS:
        formula = formula.replace(src, dst)

    formula = formula.replace("^", "**")

    # 괄호 개수 보정 (단순 카운터 방식, 중첩·함수형 오류는 잡지 못함)
    open_cnt = formula.count("(")
    close_cnt = formula.count(")")
    if open_cnt > close_cnt:
        formula += ")" * (open_cnt - close_cnt)
    elif close_cnt > open_cnt:
        formula = "(" * (close_cnt - open_cnt) + formula

    return formula


def _extract_formula_variables(formula: str) -> list[str]:
    """수식에 등장할 법한 변수 기호를 대충 뽑아낸다.

    열 이름 단위로 쓰지 말고 '기호' 개념으로만 쓴다 (단위 붙은 열명 ➜ 심볼 매핑은 별도).
    """
    # 영문/숫자/밑줄 연속 묶음을 후보로 보고, math./math.sin 같은 예약어는 제외
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    exclude = {
        "math",
        "cos",
        "sin",
        "tan",
        "acos",
        "asin",
        "atan",
        "sqrt",
        "log",
        "log10",
        "exp",
        "fabs",
        "abs",
        "pi",
        "e",
        "radians",
    }
    symbols = []
    for t in raw_tokens:
        if t not in exclude and t not in symbols:
            symbols.append(t)
    return symbols


def _extract_formula_functions(formula: str) -> list[str]:
    """수식에서 등장하는 함수명을 뽑아 추천/검증에 쓴다."""
    found: list[str] = []
    for src, _ in MATH_FUNC_PAIRS:
        func_name = src.rstrip("(").rstrip(" ")
        if func_name and func_name in formula and func_name not in found:
            found.append(func_name)
    return found


def _suggest_variable_names(
    formula: str,
    existing_columns: list[str],
) -> list[str]:
    """수식에 쓰였지만 열에는 없는 심볼을 뽑아, 입력용 추천으로 보여준다."""
    used_symbols = set(_extract_formula_variables(formula))
    # 열 이름에서 단위 붙은 부분을 떼서 심볼 목록 만들기
    column_symbols = set()
    for col in existing_columns:
        base = col.split(" [")[0].strip()
        if base:
            column_symbols.add(base)

    missing = used_symbols - column_symbols - {"math"}
    return sorted(missing)


def safe_eval_formula(
    formula: str,
    row: dict[str, Any],
    constants: dict[str, float],
) -> tuple[float | None, bool, str]:
    """수식을 안전하게 평가해 이론값(또는 실패 정보)을 반환한다.

    반환: (theory, 계산실패여부, 사용자용 메시지)
    """
    if not formula.strip():
        return 0.0, True, "수식이 비어 있습니다."

    normalized = _normalize_formula(formula)

    # 행 ➜ 변수 매핑: 열 이름에서 단위 접미사 떼고 심볼화
    eval_env: dict[str, Any] = {}
    for col, val in row.items():
        if col == "실험 측정값" or col.startswith("실험 측정값"):
            continue  # 측정값은 별도로 넘겨야 하면 아래 별도 키로
        base = col.split(" [")[0].strip()
        if base:
            try:
                eval_env[base] = float(val) if pd.notna(val) else 0.0
            except (ValueError, TypeError):
                eval_env[base] = 0.0

    eval_env["측정값"] = (
        float(row.get("실험 측정값", 0.0))
        if pd.notna(row.get("실험 측정값", 0.0))
        else 0.0
    )

    eval_env.update(constants)
    eval_env["math"] = math

    try:
        result = eval(normalized, {"__builtins__": {}}, eval_env)
        if not isinstance(result, (int, float, np.floating, np.integer)):
            return None, True, f"수식 결과가 숫자가 아닙니다: {type(result)}."
        theory = float(result)
        return theory, False, ""
    except Exception as exc:
        return None, True, f"수식 계산 오류: {exc}"


def _compute_error_metrics(
    theory: float,
    measurement: float,
) -> dict[str, Any]:
    """단일 행에 대해 여러 오차 지표를 함께 계산한다."""
    abs_err = abs(theory - measurement)
    # 분자/분모 0 처리
    denom = abs(theory) if abs(theory) > 1e-30 else abs(measurement) if abs(measurement) > 1e-30 else 1.0
    rel_err = abs_err / denom
    err_pct = rel_err * 100.0
    residual = theory - measurement  # 이론값 기준 잔차
    return {
        "절대 오차": round(abs_err, 4),
        "상대 오차 (abs/(max(|theo|,|meas|)) ": round(rel_err, 6),
        "오차율 (%)": round(err_pct, 4),
        "잔차 (이론-측정)": round(residual, 4),
    }


def _estimate_uncertainty_linear(
    formula: str,
    row: dict[str, Any],
    constants: dict[str, float],
    input_uncertainties: dict[str, float] | None = None,
    measurement_uncertainty: float = 0.0,
) -> dict[str, Any]:
    """입력 변수·상수의 불확실성을 바탕으로 이론값의 근사 불확도를 계산한다.

    아주 단순한 선형 전파(편미분 ≈ 유한 차분)를 사용한다.
    - 입력 변수·상수는 ± 불확실성에 대해 small perturbation
    - 결과는 이론값 표준불확도(근사)와 확장불확도(k≈2) 형태로 제시
    """
    if input_uncertainties is None:
        input_uncertainties = {}

    # 기본값: 상수마다 1%, 변수는 ±1% (명시적 입력이 없으면 가정)
    def default_u(name: str, val: float) -> float:
        if val == 0.0:
            return 0.0
        # 상수/변수 구분 없이 기본 1%
        return abs(val) * 0.01

    base_theory, fail_base, msg_base = safe_eval_formula(formula, row, constants)
    if fail_base or base_theory is None:
        return {
            "이론값_불확도_근사": None,
            "확장불확도_k2": None,
            "비고": "이론값 계산에 실패해 불확도를 추정하지 못했습니다.",
        }

    theory_val = base_theory

    # 변수 기여도 계산
    contributions: list[float] = []

    # 1) 열 변수
    for col, val in row.items():
        base_sym = col.split(" [")[0].strip()
        if not base_sym or base_sym == "측정값":
            continue
        try:
            v = float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            continue
        if v == 0.0 and not input_uncertainties.get(base_sym):
            continue

        u = input_uncertainties.get(base_sym)
        if u is None:
            u = default_u(base_sym, v)
        if u == 0.0:
            continue

        # 편미분 근사: ±u 로 바꿔서 이론값 변화량 측정
        row_plus = dict(row)
        row_plus[col] = v + u
        _, fail_p, _ = safe_eval_formula(formula, row_plus, constants)
        if fail_p:
            continue
        t_plus = safe_eval_formula(formula, row_plus, constants)[0]
        if t_plus is None:
            continue

        row_minus = dict(row)
        row_minus[col] = v - u
        _, fail_m, _ = safe_eval_formula(formula, row_minus, constants)
        if fail_m:
            continue
        t_minus = safe_eval_formula(formula, row_minus, constants)[0]
        if t_minus is None:
            continue

        sens = abs(t_plus - t_minus) / (2.0 * u) if u != 0 else 0.0
        contributions.append(abs(sens) * u)

    # 2) 상수 기여도
    for const_name, const_val in constants.items():
        if const_val == 0.0:
            continue
        u = input_uncertainties.get(const_name)
        if u is None:
            u = default_u(const_name, const_val)
        if u == 0.0:
            continue

        const_plus = dict(constants)
        const_plus[const_name] = const_val + u
        _, fail_p, _ = safe_eval_formula(formula, row, const_plus)
        if fail_p:
            continue
        t_plus = safe_eval_formula(formula, row, const_plus)[0]
        if t_plus is None:
            continue

        const_minus = dict(constants)
        const_minus[const_name] = const_val - u
        _, fail_m, _ = safe_eval_formula(formula, row, const_minus)
        if fail_m:
            continue
        t_minus = safe_eval_formula(formula, row, const_minus)[0]
        if t_minus is None:
            continue

        sens = abs(t_plus - t_minus) / (2.0 * u) if u != 0 else 0.0
        contributions.append(abs(sens) * u)

    # 3) 측정값 불확도 직접 반영 (모델에 따라 다르지만 단순 합성에 넣음)
    if measurement_uncertainty > 0:
        contributions.append(measurement_uncertainty)

    if not contributions:
        return {
            "이론값_불확도_근사": 0.0,
            "확장불확도_k2": 0.0,
            "비고": "불확도 계산에 사용할 입력 불확실성 정보가 없었습니다.",
        }

    # RSS 합성
    u_theory = math.sqrt(sum(c ** 2 for c in contributions))
    u_expanded = 2.0 * u_theory

    return {
        "이론값_불확도_근사": u_theory,
        "확장불확도_k2": u_expanded,
        "비고": "선형 근사(원소별 유한 차분) 기반 불확도. 입력 불확실성이 없으면 기본값(1%) 사용.",
    }


def _compute_fit_summary(
    theory_arr: np.ndarray,
    meas_arr: np.ndarray,
) -> dict[str, float]:
    """이론값-측정값 쌍에 대해 기초 선형 적합 요약(결정계수 등)을 계산한다.

    이 분석은 '이론식이 측정값을 얼마나 잘 설명하는가'를 정량화하는 보조 지표다.
    """
    if len(theory_arr) < 2:
        return {}

    x = theory_arr
    y = meas_arr
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "기울기 (이론→측정 회귀)": round(float(slope), 4),
        "절편": round(float(intercept), 4),
        "R² (이론값 기반 설명력)": round(float(r_squared), 4),
    }


# ---------------------------------------------------------------------------
# 4. 리포트 문구 (실제 결과 요약 기반 문장 생성)
# ---------------------------------------------------------------------------

def _build_report_text(
    title: str,
    formula: str,
    constants: dict[str, float],
    metrics_df: pd.DataFrame,
    fit_summary: dict[str, float],
    uncertainty_estimate: dict[str, Any],
) -> str:
    """현재 분석 결과를 요약하는 보고서용 문장을 만든다.

    완전히 '실험별 맞춤'을 목표로 하지는 않지만,
    단순 템플릿보다 실제 수치 경향·지표·불확도를 인용한다.
    """
    lines: list[str] = []
    lines.append(f"### 실험: {title or '이름 없는 실험'}")
    lines.append("")
    lines.append("**사용 수식**")
    lines.append(f"```text\n{formula or '수식 없음'}\n```")
    lines.append("")
    lines.append("**적용 상수**")
    const_lines = [f"- {k} = {v:.4g}" for k, v in constants.items()]
    lines.extend(const_lines)
    lines.append("")

    if metrics_df.empty:
        lines.append("_분석된 데이터가 없습니다._")
        return "\n".join(lines)

    # 수치 요약
    theo_col = "이론값 (계산)"
    meas_col = "실험 측정값"
    err_pct_col = "오차율 (%)"

    has_err = err_pct_col in metrics_df.columns
    has_theo = theo_col in metrics_df.columns
    has_meas = meas_col in metrics_df.columns

    theo_vals = []
    meas_vals = []
    err_vals = []
    if has_theo and has_meas:
        for _, r in metrics_df.iterrows():
            try:
                theo_vals.append(float(r[theo_col]))
            except (ValueError, TypeError):
                pass
            try:
                meas_vals.append(float(r[meas_col]))
            except (ValueError, TypeError):
                pass
            if has_err:
                try:
                    err_vals.append(float(r[err_pct_col]))
                except (ValueError, TypeError):
                    pass

    theo_means = np.mean(theo_vals) if theo_vals else None
    meas_means = np.mean(meas_vals) if meas_vals else None
    avg_err_pct = np.mean(err_vals) if err_vals else None
    max_err_pct = np.max(err_vals) if err_vals else None

    lines.append("**수치 요약**")
    if theo_means is not None:
        lines.append(f"- 평균 이론값: {theo_means:.4g}")
    if meas_means is not None:
        lines.append(f"- 평균 측정값: {meas_means:.4g}")
    if avg_err_pct is not None:
        lines.append(f"- 평균 오차율: {avg_err_pct:.3f}%")
    if max_err_pct is not None:
        lines.append(f"- 최대 오차율: {max_err_pct:.3f}%")
    lines.append("")

    # 잔차·추세 관련 언급
    if has_theo and has_meas and len(theo_vals) >= 2:
        theo_arr = np.array(theo_vals)
        meas_arr = np.array(meas_vals)
        corr = np.corrcoef(theo_arr, meas_arr)[0, 1] if np.std(theo_arr) > 0 else float("nan")
        lines.append(f"- 이론값-측정값 상관계수: {corr:.3f}")
        if abs(corr - 1.0) < 0.1 and avg_err_pct is not None and avg_err_pct < 10:
            lines.append("- 이론값 증가에 따라 측정값도 대체로 비례하는 경향입니다.")
        elif corr < 0.5:
            lines.append("- 이론값과 측정값 간 선형 상관이 약합니다. 모델 가정이나 변수 설정을 점검해보세요.")
    lines.append("")

    # 오차 지표 해석
    if has_err and avg_err_pct is not None:
        if avg_err_pct < 5:
            lines.append("- 평균 오차율이 비교적 작습니다. 정량 대조 측면에서 모델이 측정과 잘 맞을 가능성이 있습니다.")
        elif avg_err_pct < 20:
            lines.append("- 평균 오차율이 중간 수준입니다. 주요 변수·상수·측정 조건 중 어느 부분이 민감했는지 살펴보세요.")
        else:
            lines.append("- 평균 오차율이 큽니다. 수식 형태, 단위(deg/rad 등), 상수 값, 측정값 단위 등을 다시 확인하세요.")
    lines.append("")

    # 불확도 언급
    if uncertainty_estimate and uncertainty_estimate.get("이론값_불확도_근사") is not None:
        u = uncertainty_estimate["이론값_불확도_근사"]
        ue = uncertainty_estimate["확장불확도_k2"]
        lines.append(f"- 이론값 근사 불확도(1σ 수준): {u:.4g}")
        lines.append(f"- 확장 불확도(k≈2, 약 95% 수준): {ue:.4g}")
        lines.append("- 이 값은 입력·상수·측정 불확실성을 단순 선형 전파로 근사한 추정치입니다.")
        lines.append("  실제 실험 불확도 평가와 다를 수 있으므로 참고용으로 사용하세요.")
    lines.append("")

    # 적합도
    if fit_summary:
        r2 = fit_summary.get("R² (이론값 기반 설명력)")
        if r2 is not None:
            lines.append(f"- 이론값 기반 회귀 R²: {r2:.3f}")
            if r2 > 0.9:
                lines.append("- 이론값을 독립변수로 볼 때 측정값을 꽤 잘 설명합니다.")
            elif r2 > 0.7:
                lines.append("- 설명력이 보통 수준입니다. 체계적 편차나 이상치가 있을 수 있습니다.")
            else:
                lines.append("- 설명력이 낮습니다. 이론식 또는 변수 매핑을 점검하세요.")
    lines.append("")

    # 체계적 편향 언급
    if has_theo and has_meas and len(theo_vals) >= 2:
        theo_arr = np.array(theo_vals)
        meas_arr = np.array(meas_vals)
        resid = theo_arr - meas_arr
        mean_resid = resid.mean()
        if abs(mean_resid) > 1e-6:
            direction = "과대평가 경향이 있습니다" if mean_resid > 0 else "과소평가 경향이 있습니다"
            lines.append(f"- 전반적으로 이론값 대비 측정값이 {direction} (평균 잔차 {mean_resid:.4g}).")
            lines.append("  일정한 방향의 편향이 보이면 보정 계수나 영점 오차 등을 검토하세요.")
    lines.append("")

    # 오차 원인 일반 조언
    lines.append("**오차 원인 점검 포인트**")
    lines.append("- 기기 분해능·눈금 읽기 오차·디지털 반올림")
    lines.append("- 영점/교정 상태, 온도·습도·진동 등 환경 요인")
    lines.append("- 이론 모델의 가정(마찰/저항/방사 손실 무시 등)과 실제 차이")
    lines.append("- 단위 변환 누락 (예: deg·rad, eV·J 등)")
    lines.append("")

    lines.append("**실험 개선 제언**")
    lines.append("- 주요 변수 구간에서 반복 측정 후 평균·표준편차 확인")
    lines.append("- 상수·기준 장비의 교정 상태 재점검")
    lines.append("- 민감도가 큰 변수를 우선 정밀하게 통제")
    lines.append("- 가능하면 독립변수 구간을 넓혀 모델 적합성을 더 넓게 검토")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. 사이드바: 상수·열·템플릿 관리 (보조 로직 포함)
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    """사이드바 전체(장비·상수·열·템플릿)를 렌더링하고 현재 실험 모드 문자열을 반환한다."""
    st.sidebar.header("⚙️ 실험 장비 및 상수 설정")

    # -- 상수 추가 ----------------------------------------------------------
    with st.sidebar.expander("➕ 새로운 상수/장비 변수 추가"):
        new_const_name = st.text_input("상수 기호 (예: E0, mc2, R)")
        new_const_val = st.number_input("초기값 세팅", value=0.0, step=0.1)
        if st.button("추가하기", key="add_const_btn"):
            if new_const_name and new_const_name not in st.session_state.custom_constants:
                st.session_state.custom_constants[new_const_name] = float(new_const_val)
                st.rerun()

    # -- 상수 삭제 ----------------------------------------------------------
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
    st.sidebar.markdown("**현재 적용된 상수 목록 (수정 가능)**")
    # 상수 목록은 고정 위젯 키가 깨지지 않도록 이름 기반 키로 관리
    const_keys = list(st.session_state.custom_constants.keys())
    for i, key in enumerate(const_keys):
        prev = float(st.session_state.custom_constants[key])
        new_val = st.sidebar.number_input(
            f"{key}",
            value=prev,
            format="%.4g",
            key=f"const_val_{i}_{key}",
            step=0.0 if prev == 0.0 else prev * 0.01,
        )
        st.session_state.custom_constants[key] = new_val

    st.sidebar.divider()

    # -- 상수 불확실성 입력 (불확도 모델링용) -------------------------------
    st.sidebar.markdown("**입력·상수 불확실성 (선택)**")
    st.sidebar.markdown(
        "<div class='small-note'>불확도 근사에 쓸 ±값. 비어 있으면 상수는 1%, "
        "변수는 측정값 기준 자동 가정.</div>",
        unsafe_allow_html=True,
    )
    if "input_uncertainty" not in st.session_state:
        st.session_state.input_uncertainty = {}

    all_names = list(st.session_state.custom_constants.keys()) + [
        c.split(" [")[0].strip() for c in st.session_state.input_df.columns if c != "실험 측정값"
    ]
    seen: set[str] = set()
    unique_names: list[str] = []
    for n in all_names:
        if n and n not in seen:
            seen.add(n)
            unique_names.append(n)

    col_u1, col_u2 = st.sidebar.columns(2)
    with col_u1:
        for nm in unique_names[: max(1, len(unique_names) // 2 + len(unique_names) % 2)]:
            default_u = 0.0
            cur = st.session_state.input_uncertainty.get(nm, None)
            if cur is not None:
                default_u = float(cur)
            val = st.number_input(
                f"u({nm})",
                value=default_u,
                format="%.4g",
                key=f"u_{nm}",
                step=0.0 if default_u == 0.0 else abs(default_u) * 0.1,
            )
            if val > 0 or st.session_state.input_uncertainty.get(nm) == 0.0:
                st.session_state.input_uncertainty[nm] = float(val)
            elif nm in st.session_state.input_uncertainty:
                del st.session_state.input_uncertainty[nm]
    with col_u2:
        for nm in unique_names[len(unique_names) // 2 + len(unique_names) % 2:]:
            default_u = 0.0
            cur = st.session_state.input_uncertainty.get(nm, None)
            if cur is not None:
                default_u = float(cur)
            val = st.number_input(
                f"u({nm})",
                value=default_u,
                format="%.4g",
                key=f"u2_{nm}",
                step=0.0 if default_u == 0.0 else abs(default_u) * 0.1,
            )
            if val > 0 or st.session_state.input_uncertainty.get(nm) == 0.0:
                st.session_state.input_uncertainty[nm] = float(val)
            elif nm in st.session_state.input_uncertainty:
                del st.session_state.input_uncertainty[nm]

    meas_u = st.sidebar.number_input(
        "측정값 불확도 (±)",
        value=0.0,
        format="%.4g",
        key="u_meas",
        step=0.0,
    )
    if meas_u > 0:
        st.session_state.input_uncertainty["측정값"] = float(meas_u)
    elif "측정값" in st.session_state.input_uncertainty:
        del st.session_state.input_uncertainty["측정값"]

    st.sidebar.divider()

    # -- 실험 모드 ----------------------------------------------------------
    experiment_choice = st.sidebar.selectbox(
        "진행할 실험 모드",
        ["사용자 맞춤형 자동 분석", "전자의 비전하 (e/m) 측정 (프리셋)"],
        key="mode_sel",
    )

    # -- 템플릿 프리셋 -------------------------------------------------------
    if experiment_choice == "사용자 맞춤형 자동 분석":
        st.header("🛠️ 1. 실험 데이터 및 수식 세팅")
        st.markdown("**📌 표 변수(열) 관리 및 템플릿**")

        with st.popover("열 추가/삭제 및 추천 템플릿 열기"):
            st.markdown("**💡 추천 변수 템플릿 불러오기**")
            if st.button("컴프턴 효과 템플릿 불러오기", key="tpl_compton"):
                st.session_state.input_df = pd.DataFrame(
                    columns=[
                        "산란각 [deg]",
                        "산란체 유무 [1=O, 0=X]",
                        "실험 측정값 [keV]",
                    ],
                    data=[
                        [30.0, 1.0, 0.0],
                        [60.0, 1.0, 0.0],
                        [90.0, 1.0, 0.0],
                    ],
                )
                st.rerun()
            if st.button("전자의 비전하 템플릿 불러오기", key="tpl_em"):
                st.session_state.input_df = pd.DataFrame(
                    columns=[
                        "가속전압 [V]",
                        "코일 전류 [A]",
                        "실험 측정값 [C/kg]",
                    ],
                    data=[
                        [150.0, 1.1, 0.0],
                        [180.0, 1.2, 0.0],
                    ],
                )
                st.rerun()

            st.divider()
            st.write("새로운 변수 직접 추가")
            new_col_name = st.text_input("변수명 (예: 전압, 거리)", key="new_col_name")
            unit_choice = st.selectbox(
                "단위",
                ["선택안함", "deg", "rad", "keV", "eV", "V", "A", "m", "cm", "mm", "nm", "s", "kg", "C", "N"],
                key="new_col_unit",
            )
            if st.button("열 추가", key="add_col_btn"):
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
                if st.button("열 삭제", key="del_col_btn"):
                    st.session_state.input_df = st.session_state.input_df.drop(
                        columns=[del_col_choice]
                    )
                    st.rerun()
            else:
                st.write("열을 하나 이상 남겨야 합니다.")

        # 수식 입력 + 검증 피드백
        st.markdown("**🧮 이론값 산출 식 입력**")
        raw_formula = st.text_input(
            "수식 입력",
            value="",
            placeholder="예: E0 / (1 + (E0/mc2) * (1 - cos(산란각)))",
            key="formula_input",
        )

        # 수식 검증 도우미 피드백
        formula_functions = _extract_formula_functions(raw_formula)
        missing_vars = _suggest_variable_names(raw_formula, st.session_state.input_df.columns)
        if raw_formula.strip():
            st.markdown("**수식 검증 도우미**")
            if formula_functions:
                st.info(
                    f"수식에 사용된 함수: {', '.join(formula_functions)}  \n"
                    "cos/sin/tan은 각도(deg) 입력을 rad로 자동 변환합니다. "
                    "acos/asin/atan/sqrt/log/exp 등은 괄호·인자 개수를 확인하세요."
                )
            else:
                st.success("수식에 삼각함수·루트·지수 등의 함수가 보이지 않습니다. 정상일 수 있습니다.")

            if missing_vars:
                st.warning(
                    f"수식에 쓰였지만 현재 표에 없는 기호: {', '.join(missing_vars)}  \n"
                    "변수명을 열 이름으로 추가하거나, 수식의 기호명을 표의 열 이름과 맞추세요."
                )
            else:
                st.success("수식에 쓰인 기호가 현재 표와 대체로 일치합니다 (상수·math 제외).")

            # 괄호/연산자 간단 경고
            if raw_formula.count("(") != raw_formula.count(")"):
                st.warning("괄호 '('와 ')' 개수가 다릅니다. 자동 보정이 적용되지만 구조를 확인하세요.")
            if "^" in raw_formula:
                st.info("'^'는 자동으로 '**'(거듭제곱)로 변환됩니다.")
        else:
            st.info("수식을 입력하면 함수·변수·괄호 검토를 보여줍니다.")

        col_add, col_form = st.columns([1, 1.5])

        with col_add:
            # (이미 popover에서 열 관리 처리했으므로 여기서는 비워둠) -> 실제로는 위 popover가 담당
            st.write("열 관리는 우측 상단의 '열 추가/삭제 및 추천 템플릿 열기' 팝오버를 사용하세요.")

        with col_form:
            st.markdown("**🧮 이론값 산출 식 입력 (재입력 영역)**")
            _ = st.text_input(
                "수식 입력",
                value=raw_formula,
                key="formula_input2",
                placeholder="예: E0 / (1 + (E0/mc2) * (1 - cos(산란각)))",
            )
            st.write("※ 수식 입력은 상단에서 이미 받았으며, 여기서는 확인용입니다.")

        # 데이터 편집기
        edited_df = st.data_editor(
            st.session_state.input_df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor",
        )

        # 분석 실행
        if st.button("🚀 정밀 데이터 분석 및 리포트 보드에 추가", type="primary", key="analyze_btn"):
            meas_col_name = None
            for col in edited_df.columns:
                if col == "실험 측정값" or col.startswith("실험 측정값"):
                    meas_col_name = col
                    break
            if meas_col_name is None:
                st.error("표에 '실험 측정값' 열이 없습니다. 열 이름을 확인하거나 추가해주세요.")
                st.rerun()

            formula = raw_formula.strip()
            if not formula:
                st.error("이론값 수식을 입력하세요.")
                st.rerun()

            results: list[dict[str, Any]] = []
            theo_arr: list[float] = []
            meas_arr: list[float] = []
            row_details: list[dict[str, Any]] = []

            for idx, row in edited_df.iterrows():
                raw_meas = row.get(meas_col_name, 0.0)
                meas = float(raw_meas) if pd.notna(raw_meas) else 0.0

                if meas_col_name and meas_col_name in row.index:
                    row_dict = {k: v for k, v in row.to_dict().items() if k != meas_col_name}
                    row_dict["실험 측정값"] = meas
                else:
                    row_dict = {k: v for k, v in row.to_dict().items()}
                    row_dict["실험 측정값"] = meas

                # 이론값 계산
                theory, fail, msg = safe_eval_formula(formula, row_dict, st.session_state.custom_constants)

                # 불확도 추정
                unc = _estimate_uncertainty_linear(
                    formula,
                    row_dict,
                    st.session_state.custom_constants,
                    input_uncertainties=st.session_state.input_uncertainty,
                    measurement_uncertainty=float(st.session_state.input_uncertainty.get("측정값", 0.0) or 0.0),
                )

                metrics: dict[str, Any] = {"이론값 (계산)": None, "실험 측정값": meas}

                if fail or theory is None:
                    metrics["이론값 (계산)"] = "?"
                    metrics["절대 오차"] = "계산 불가"
                    metrics["오차율 (%)"] = "계산 불가"
                    metrics["잔차 (이론-측정)"] = "계산 불가"
                    metrics["오차 원인 메시지"] = msg
                    results.append({**row_dict, **metrics})
                    continue

                metrics["이론값 (계산)"] = round(float(theory), 4)
                em = _compute_error_metrics(float(theory), meas)
                metrics.update(em)

                # 불확도 근사치 추가
                if unc and unc.get("이론값_불확도_근사") is not None:
                    metrics["이론값 불확도 (근사)"] = round(unc["이론값_불확도_근사"], 4)
                    metrics["확장 불확도 (k≈2)"] = round(unc["확장불확도_k2"], 4)

                results.append({**row_dict, **metrics})
                theo_arr.append(float(theory))
                meas_arr.append(meas)
                row_details.append({"theory": float(theory), "meas": meas, "unc": unc})

            res_df = pd.DataFrame(results)

            # 적합도 요약
            if theo_arr and meas_arr:
                fit_summary = _compute_fit_summary(np.array(theo_arr), np.array(meas_arr))
            else:
                fit_summary = {}

            # 상수 문자열
            current_const_str = ", ".join(
                f"{k}={v:.4g}" for k, v in st.session_state.custom_constants.items()
            ) if st.session_state.custom_constants else "설정된 상수 없음"

            exp_name = st.session_state.get("exp_name", "")

            report_text = _build_report_text(
                title=exp_name,
                formula=formula,
                constants=st.session_state.custom_constants,
                metrics_df=res_df,
                fit_summary=fit_summary,
                uncertainty_estimate=row_details[0]["unc"] if row_details else {},
            )

            st.session_state.history.append({
                "id": len(st.session_state.history) + 1,
                "title": exp_name if exp_name else "이름 없는 실험",
                "formula": formula,
                "constants": current_const_str,
                "df": res_df,
                "report_markdown": report_text,
                "fit_summary": fit_summary,
                "uncertainty_estimate": row_details[0]["unc"] if row_details else {},
                "theo_arr": np.array(theo_arr),
                "meas_arr": np.array(meas_arr),
            })
            st.rerun()

        st.divider()
        st.header("📚 2. 실험 분석 누적 보드 및 학술 진단 리포트")

        if not st.session_state.history:
            st.write("표를 채우고 수식을 입력한 뒤 '리포트 보드에 추가' 버튼을 눌러주세요.")
        else:
            for rec in reversed(st.session_state.history):
                with st.container():
                    st.markdown(f"### 📊 파트 #{rec['id']} : {rec['title']}")
                    st.markdown(
                        f"""
<div class="metric-card">
    <b>사용한 물리 수식:</b> {rec['formula'] if rec['formula'] else '수식 없음'} <br>
    <b>시스템 적용 상수:</b> {rec['constants']}
</div>
<br>
""",
                        unsafe_allow_html=True,
                    )

                    st.dataframe(rec["df"], use_container_width=True)

                    # 적합도 요약 박스
                    if rec.get("fit_summary"):
                        fs = rec["fit_summary"]
                        st.markdown("**적합 요약 (이론값→측정값 회귀)**")
                        st.json(fs)

                    if rec.get("uncertainty_estimate") and rec["uncertainty_estimate"].get("이론값_불확도_근사") is not None:
                        ue = rec["uncertainty_estimate"]
                        st.markdown("**불확도 추정 (선형 근사)**")
                        st.json({
                            "이론값 불확도 (근사, 1σ)": ue["이론값_불확도_근사"],
                            "확장 불확도 (k≈2)": ue["확장불확도_k2"],
                            "비고": ue.get("비고", ""),
                        })

                    st.divider()

                    # 통계·시각화 패널
                    st.markdown("**📈 측정 데이터 및 추세 검토 (통계·시각화)**")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_hist = _make_residual_and_hist(rec)
                        if fig_hist:
                            st.pyplot(fig_hist)
                        else:
                            st.write("_그래프를 그리려면 분석에 최소 1개 이상의 행이 필요합니다._")
                    with c2:
                        fig_scatter = _make_scatter_and_residual(rec)
                        if fig_scatter:
                            st.pyplot(fig_scatter)
                        else:
                            st.write("_그래프를 그리려면 분석에 최소 2개 이상의 행이 필요합니다._")

                    st.markdown("**🧠 AI 심층 학술 분석 리포트**")
                    tab1, tab2, tab3 = st.tabs(
                        ["📈 측정 데이터 및 트렌드 검토", "🔬 잠재적 오차 원인 분석", "🛠️ 실험 방법론적 제언"]
                    )
                    with tab1:
                        st.markdown(
                            f"<div class='report-text'>{rec.get('report_markdown', '')}</div>",
                            unsafe_allow_html=True,
                        )
                    with tab2:
                        st.markdown(
                            """
<div class="report-text">
<b>[체계적 및 우연적 오차 요인 진단]</b><br>
*   <b>기기 장비의 한계:</b> 측정 기기의 분해능, 눈금 읽기 오차, 반올림, 영점 오차 가능성 검토.
*   <b>환경적 변인:</b> 외부 노이즈, 온도/습도 변화, 진동, 전원 변동 등 미제어 변수에 따른 편차.
*   <b>모델 가정과의 괴리:</b> 마찰·저항·복사 손실·무시된 항 등 실제가 이론과 다른 부분 점검.
*   <b>단위·변환 오류:</b> deg ↔ rad, eV ↔ J 등 단위 불일치가 큰 오차로 이어질 수 있음.
</div>
""",
                            unsafe_allow_html=True,
                        )
                    with tab3:
                        st.markdown(
                            """
<div class="report-text">
<b>[후속 실험을 위한 개선 제언]</b><br>
*   동일 조건 반복 측정으로 통계적 신뢰구간 확보.
*   상수·기준 장비의 교정 상태 재확인.
*   민감도가 큰 변수 우선 정밀 통제.
*   독립변수 구간을 넓혀 모델 적합성을 더 넓게 검토.
*   가능하면 불확도 전파를 고려한 실험 설계(측정 횟수·분해능 선택).
</div>
""",
                            unsafe_allow_html=True,
                        )
                st.write("---")

        # 챗봇
        st.header("💬 3. AI 실험 멘토 (실시간 심층 Q&A)")
        st.write("오차율 분석, 수식 유도, 추가 실험 설계, 불확도 해석 등에 대해 질문할 수 있습니다.")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(
            "질문을 입력하세요. (예: 이 실험에서 오차를 줄이기 위한 가장 핵심적인 통제 변수는 무엇인가요?)"
        ):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("최근 분석 결과를 참고해 답변을 준비 중입니다..."):
                    # 최근 분석 결과를 맥락에 포함
                    context_lines: list[str] = []
                    if st.session_state.history:
                        last = st.session_state.history[-1]
                        context_lines.append(f"**최근 분석 요약 (파트 #{last['id']} · {last['title']})**")
                        context_lines.append(f"- 수식: `{last['formula'] or '수식 없음'}`")
                        context_lines.append(f"- 적용 상수: {last['constants']}")
                        df = last["df"]
                        theo_col = "이론값 (계산)"
                        meas_col = "실험 측정값"
                        err_col = "오차율 (%)"
                        if theo_col in df.columns and meas_col in df.columns:
                            theo_vals = []
                            meas_vals = []
                            err_vals = []
                            for _, r in df.iterrows():
                                try:
                                    theo_vals.append(float(r[theo_col]))
                                except (ValueError, TypeError):
                                    pass
                                try:
                                    meas_vals.append(float(r[meas_col]))
                                except (ValueError, TypeError):
                                    pass
                                if err_col in df.columns:
                                    try:
                                        err_vals.append(float(r[err_col]))
                                    except (ValueError, TypeError):
                                        pass
                            if theo_vals and meas_vals:
                                corr = np.corrcoef(theo_vals, meas_vals)[0, 1] if np.std(theo_vals) > 0 else float("nan")
                                avg_err = float(np.mean(err_vals)) if err_vals else None
                                context_lines.append(f"- 평균 오차율: {avg_err:.3f}%" if avg_err is not None else "- 오차율: 계산 불가")
                                context_lines.append(f"- 이론값-측정값 상관계수: {corr:.3f}")
                                context_lines.append(f"- 평균 이론값: {np.mean(theo_vals):.4g}, 평균 측정값: {np.mean(meas_vals):.4g}")
                        if last.get("uncertainty_estimate") and last["uncertainty_estimate"].get("이론값_불확도_근사") is not None:
                            u = last["uncertainty_estimate"]["이론값_불확도_근사"]
                            ue = last["uncertainty_estimate"]["확장불확도_k2"]
                            context_lines.append(f"- 이론값 근사 불확도: {u:.4g}, 확장 불확도(k≈2): {ue:.4g}")

                    context_block = "\n".join(context_lines)
                    answer = _build_chat_answer(prompt, context_block)
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    elif experiment_choice == "전자의 비전하 (e/m) 측정 (프리셋)":
        st.info("프리셋 예시 탭입니다. '사용자 맞춤형 자동 분석'을 이용해주세요.")


# ---------------------------------------------------------------------------
# 6. 보조 렌더링 함수
# ---------------------------------------------------------------------------

def _make_scatter_and_residual(record: dict[str, Any]) -> Any | None:
    """이론값 vs 측정값 산점도 + 잔차 막대 그래프를 하나의 figure로 그린다."""
    df = record.get("df")
    if df is None or df.empty:
        return None
    theo_vals: list[float] = []
    meas_vals: list[float] = []
    labels: list[str] = []
    theo_col = "이론값 (계산)"
    meas_col = "실험 측정값"
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            t = float(r[theo_col])
        except (ValueError, TypeError):
            continue
        try:
            m = float(r[meas_col])
        except (ValueError, TypeError):
            continue
        theo_vals.append(t)
        meas_vals.append(m)
        labels.append(f"#{i+1}")

    if len(theo_vals) < 1:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1, ax2 = axes

    # 산점도 + y=x 선
    ax1.scatter(theo_vals, meas_vals, s=30, color="#0d6efd")
    lo = min(min(theo_vals), min(meas_vals)) * 0.9 if theo_vals and meas_vals else 0
    hi = max(max(theo_vals), max(meas_vals)) * 1.1 if theo_vals and meas_vals else 1
    if hi - lo < 1e-6:
        lo -= 1
        hi += 1
    ax1.plot([lo, hi], [lo, hi], color="#888", linestyle="--", linewidth=1)
    ax1.set_ylabel("실험 측정값")
    ax1.set_title("이론값 vs 실험 측정값")
    ax1.grid(True, alpha=0.3)

    # 잔차
    theo_arr = np.array(theo_vals)
    meas_arr = np.array(meas_vals)
    resid = theo_arr - meas_arr
    ax2.bar(range(len(resid)), resid, color="#0d6efd", alpha=0.7)
    ax2.axhline(0, color="#888", linestyle="-", linewidth=1)
    ax2.set_xticks(range(len(resid)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("잔차 (이론 - 측정)")
    ax2.set_title("잔차 분포")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _make_residual_and_hist(record: dict[str, Any]) -> Any | None:
    """잔차 히스토그램 + 측정값 분포(측정값이 의미 있을 때)를 그린다."""
    df = record.get("df")
    if df is None or df.empty:
        return None
    theo_vals: list[float] = []
    meas_vals: list[float] = []
    theo_col = "이론값 (계산)"
    meas_col = "실험 측정값"
    for _, r in df.iterrows():
        try:
            theo_vals.append(float(r[theo_col]))
        except (ValueError, TypeError):
            pass
        try:
            meas_vals.append(float(r[meas_col]))
        except (ValueError, TypeError):
            pass

    if not theo_vals:
        return None

    resid = np.array(theo_vals) - np.array(meas_vals) if meas_vals else np.zeros_like(theo_vals)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    ax1, ax2 = axes

    ax1.hist(resid, bins=max(5, min(15, len(resid))), color="#0d6efd", alpha=0.7, edgecolor="white")
    ax1.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax1.set_title("잔차 히스토그램")
    ax1.set_xlabel("이론값 - 측정값")
    ax1.set_ylabel("빈도")
    ax1.grid(True, alpha=0.3)

    if meas_vals:
        ax2.hist(meas_vals, bins=max(5, min(15, len(meas_vals))), color="#28a745", alpha=0.7, edgecolor="white")
        ax2.set_title("측정값 분포")
        ax2.set_xlabel("실험 측정값")
        ax2.set_ylabel("빈도")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis("off")

    plt.tight_layout()
    return fig


def _build_chat_answer(prompt: str, context_block: str) -> str:
    """질문과 최근 분석 맥락을 반영해 챗봇 답변을 만든다.

    (실제 LLM 연동 없이 규칙 기반이지만, 맥락을 인용하는 구조로 만든다.)
    """
    prompt_lower = prompt.lower()

    lines: list[str] = []
    lines.append("안녕하세요. 질문을 분석해 답변드립니다.")

    if context_block.strip():
        lines.append("")
        lines.append("---")
        lines.append("📌 최근 분석 맥락")
        lines.append(context_block)
        lines.append("---")
        lines.append("")

    # 키워드 기반 보조 답변
    if "오차" in prompt_lower and ("줄이" in prompt_lower or "감소" in prompt_lower or "감소시키" in prompt_lower):
        lines.append("오차를 줄이기 위한 일반적인 우선순위는 다음과 같습니다.")
        lines.append("1. **가장 민감한 변수/상수**를 먼저 정밀하게 통제하거나 교정하세요.")
        lines.append("2. 측정 장비의 **분해능·정밀도**가 요구 수준에 맞는지 확인하고, 필요하면 더 정밀한 장비를 사용하세요.")
        lines.append("3. **영점 교정·기준값 교정**을 먼저 재수행하세요. 착오 오차가 클 때 효과가 큽니다.")
        lines.append("4. 단위 변환(deg↔rad, eV↔J 등) 누락이 없는지 확인하세요. 단순한 사고가 큰 오차로 이어집니다.")
        lines.append("5. 가능하면 **반복 측정 후 평균**을 사용하고, 이상치 처리 기준을 미리 정해두세요.")
        lines.append("")
        lines.append("최근 분석 결과에 특정 변수나 상수의 불확도가 계산되어 있다면, 그 항목이 오차에 가장 크게 기여했을 가능성이 높습니다.")
    elif "불확도" in prompt_lower or "uncertainty" in prompt_lower:
        lines.append("불확도는 측정값이나 계산값이 가질 수 있는 변동 범위를 나타냅니다.")
        lines.append("이 시스템에서는 입력한 변수별 ±불확도 및 상수·측정값 불확도를 **선형 근사(유한 차분)**로 전파해 이론값의 근사 불확도를 추정합니다.")
        lines.append("확장불확도(k≈2)는 약 95% 신뢰 수준에 해당하는 범위로 참고용으로 사용하세요.")
        lines.append("정확한 불확도 평가는 실험 설계, 분포를 고려한 전파, 반복 측정 통계가 필요합니다.")
    elif "추천" in prompt_lower or "지표" in prompt_lower or "무엇을 보면" in prompt_lower:
        lines.append("실험 성격에 따라 유용한 지표가 다릅니다.")
        lines.append("- **상수 결정 실험**이면 추정값·불확도·이론 상수와의 비교가 핵심입니다.")
        lines.append("- **선형성/모델 검증 실험**이면 R², 잔차의 무작위 분포, 이론-측정 상관이 유용합니다.")
        lines.append("- **단순 오차 확인**이면 절대오차·오차율·잔차 방향이 기본입니다.")
        lines.append("현재 분석 결과에는 적합 요약과 불확도 추정치가 함께 제공되니 함께 참고하세요.")
    elif "실험 설계" in prompt_lower or "실험" in prompt_lower:
        lines.append("실험을 설계할 때는 다음을 먼저 정하면 좋습니다.")
        lines.append("- 확인하려는 **물리 법칙/관계**와 그에 필요한 **독립변수·종속변수**")
        lines.append("- 측정할 구간과 **반복 횟수**, 필요한 **정밀도 목표**")
        lines.append("- 통제해야 할 **교란 변수**와 보정 방법")
        lines.append("- 데이터 분석 방식(이론식, 피팅, 불확도 전파 등)을 사전에 정해두면 해석이 쉬워집니다.")
    else:
        lines.append("질문 내용을 바탕으로 일반적인 물리 실험 관점에서 답변드리면 다음과 같습니다.")
        lines.append("구체적인 판단을 위해서는 실험 주제, 사용 수식, 주요 변수/상수, 측정 방식, 목표 정밀도를 함께 알려주시면 더 정확히 안내할 수 있습니다.")
        lines.append("필요하면 보드에서 최근 분석 결과를 선택하거나, 관련 행 data를 함께 말해주세요.")

    lines.append("")
    lines.append("---")
    lines.append("더 구체적인 실험 상황을 알려주시면 해당 맥락에 맞춰 다시 설명드리겠습니다.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. 메인 진입
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()


if __name__ == "__main__":
    main()
