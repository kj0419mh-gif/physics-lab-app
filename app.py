# -*- coding: utf-8 -*-
"""
물리실험 결과 분석 시스템
- 원작자 및 저작권자: 박민후 (kj0419mh@gmail.com)

구성
  1. 페이지 설정 / 스타일
  2. 세션 상태 초기화 (새로고침 시 항상 빈 상태)
  3. 수식 계산 엔진  (AST 기반 안전 계산, 공백·한글 변수명 지원)
  4. 매뉴얼 분석 엔진 (템플릿 매칭 + 범용 수식·상수·변수 추출)
  5. 통계·감도 분석 및 리포트 생성 엔진
  6. AI 멘토 지식 베이스 (실험별 원리·유도·오차 물리 + 범용 방법론 + 참고 자료)
  7. AI 멘토 엔진 (내장 지식 엔진 / 외부 LLM API 연동)
  8. UI (사이드바 / 메인 / 누적 보드 / AI 멘토)
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    from PyPDF2 import PdfReader


# ---------------------------------------------------------------------------
# 1. 페이지 설정 및 스타일
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="물리실험 결과 분석 시스템",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .report-text { font-size: 0.95rem; line-height: 1.7; color: #2c3e50; }
    .metric-card {
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border-left: 5px solid #3182ce; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px; word-break: break-all;
    }
    .ai-box {
        background-color: #eff6ff; padding: 15px; border-radius: 8px;
        border: 1px solid #bfdbfe; margin-bottom: 15px; color: #1e40af; font-size: 0.92rem;
    }
    .small-note { font-size: 0.85rem; color: #64748b; }
    .footer-note {
        text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 40px;
        padding: 20px; border-top: 1px solid #e5e7eb;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔬 물리실험 결과 분석 시스템")
st.markdown(
    "실험 매뉴얼(PDF)과 측정 데이터를 기반으로 수식·상수·표 형식을 자동 추천하고, "
    "통계·감도 분석이 포함된 심층 학술 리포트와 실시간 교차 검증, 시각화, "
    "원리 설명·참고 자료를 제공하는 AI 멘토를 갖춘 **범용 물리실험 분석 도구**입니다."
)

MEAS_KEY = "실험 측정값"


def _setup_korean_font() -> bool:
    candidates = [
        "Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
        "Noto Sans CJK KR", "Noto Sans KR", "D2Coding", "Gulim", "Batang",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


KOREAN_FONT_OK = _setup_korean_font()


# ---------------------------------------------------------------------------
# 2. 세션 상태 초기화
#    Streamlit 은 브라우저 새로고침 시 세션을 새로 만들므로 아래 기본값이 곧 '초기 상태'다.
#    서버·디스크에 어떤 상태도 저장하지 않는다 → 새로고침 = 완전 초기화.
# ---------------------------------------------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({"변수 1": [np.nan, np.nan, np.nan], MEAS_KEY: [np.nan, np.nan, np.nan]})


def _init_session_state() -> None:
    ss = st.session_state
    ss.setdefault("history", [])
    ss.setdefault("chat_messages", [])
    ss.setdefault("custom_constants", {})
    ss.setdefault("input_df", _empty_df())
    ss.setdefault("manual_text", "")
    ss.setdefault("manual_sig", None)
    ss.setdefault("suggestion", None)
    ss.setdefault("table_version", 0)


_init_session_state()


# ---------------------------------------------------------------------------
# 3. 수식 계산 엔진
# ---------------------------------------------------------------------------

_FUNC_NAMES = {
    "sin", "cos", "tan", "asin", "acos", "atan", "arcsin", "arccos", "arctan",
    "sqrt", "log", "ln", "log10", "exp", "abs", "pow", "min", "max",
    "radians", "degrees", "pi", "PI",
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
)

_SYMBOL_MAP = [
    ("×", "*"), ("·", "*"), ("∙", "*"), ("⋅", "*"), ("÷", "/"),
    ("−", "-"), ("–", "-"), ("²", "**2"), ("³", "**3"), ("^", "**"),
]


def _make_namespace(angle_unit: str) -> dict[str, Any]:
    deg = angle_unit == "deg"

    def _in(fn):
        return (lambda x: fn(math.radians(x))) if deg else (lambda x: fn(x))

    def _out(fn):
        return (lambda x: math.degrees(fn(x))) if deg else (lambda x: fn(x))

    return {
        "sin": _in(math.sin), "cos": _in(math.cos), "tan": _in(math.tan),
        "asin": _out(math.asin), "acos": _out(math.acos), "atan": _out(math.atan),
        "arcsin": _out(math.asin), "arccos": _out(math.acos), "arctan": _out(math.atan),
        "sqrt": math.sqrt, "log": math.log, "ln": math.log, "log10": math.log10, "exp": math.exp,
        "abs": abs, "pow": pow, "min": min, "max": max,
        "radians": math.radians, "degrees": math.degrees,
        "pi": math.pi, "PI": math.pi,
    }


def _substitute_names(formula: str, names: list[str]) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    out = formula
    for i, name in enumerate(sorted({n for n in names if n}, key=len, reverse=True)):
        token = f"__v{i}__"
        out, n_sub = re.subn(r"(?<![\w])" + re.escape(name) + r"(?![\w])", token, out)
        if n_sub:
            mapping[token] = name
    return out, mapping


def _to_python_expr(formula: str) -> str:
    f = formula.strip()
    for src, dst in _SYMBOL_MAP:
        f = f.replace(src, dst)
    f = re.sub(r"√\s*\(", "sqrt(", f)
    f = re.sub(r"√\s*([\w.]+)", r"sqrt(\1)", f)
    return f.replace(")(", ")*(")


def _to_float(v: Any) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def safe_eval_formula(
    formula: str, variables: dict[str, Any], constants: dict[str, float], angle_unit: str = "deg",
) -> tuple[float | None, str]:
    if not formula or not formula.strip():
        return None, "수식이 비어 있습니다."

    ns = _make_namespace(angle_unit)
    all_names = {**constants, **variables}
    substituted, mapping = _substitute_names(formula, list(all_names))
    for token, name in mapping.items():
        fv = _to_float(all_names[name])
        if math.isnan(fv):
            return None, f"'{name}' 값이 입력되지 않았습니다."
        ns[token] = fv

    expr = _to_python_expr(substituted)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return None, f"수식 문법 오류: {exc.msg} (변수명은 표 컬럼명과 정확히 일치해야 합니다)"

    unknown: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, f"허용되지 않는 표현식입니다: {type(node).__name__}"
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _FUNC_NAMES):
                return None, "허용되지 않는 함수 호출입니다."
        elif isinstance(node, ast.Name) and node.id not in ns:
            unknown.append(node.id)
    if unknown:
        return None, "정의되지 않은 기호: " + ", ".join(dict.fromkeys(unknown))

    try:
        result = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, ns)
    except ZeroDivisionError:
        return None, "0으로 나누기가 발생했습니다."
    except (ValueError, OverflowError) as exc:
        return None, f"수학 계산 오류: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"계산 오류: {exc}"

    if isinstance(result, (bool, complex)) or not isinstance(result, (int, float, np.integer, np.floating)):
        return None, "결과가 실수가 아닙니다."
    if math.isnan(result) or math.isinf(result):
        return None, "결과가 유효한 수가 아닙니다 (NaN/Inf)."
    return float(result), ""


def formula_symbols(formula: str, variables: list[str], constants: list[str]) -> dict[str, list[str]]:
    result = {"vars": [], "consts": [], "unknown": []}
    if not formula.strip():
        return result
    names = list(dict.fromkeys(list(variables) + list(constants)))
    substituted, mapping = _substitute_names(formula, names)
    for token, name in mapping.items():
        (result["vars"] if name in variables else result["consts"]).append(name)
    try:
        tree = ast.parse(_to_python_expr(substituted), mode="eval")
    except SyntaxError:
        return result
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in mapping and node.id not in _FUNC_NAMES:
            result["unknown"].append(node.id)
    result["unknown"] = list(dict.fromkeys(result["unknown"]))
    return result


def _find_meas_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if MEAS_KEY in str(col) or "Measured Value" in str(col):
            return str(col)
    return None


def _base_name(col: str) -> str:
    return str(col).split(" [")[0].split(" (")[0].strip()


def evaluate_table(
    df: pd.DataFrame, formula: str, constants: dict[str, float], angle_unit: str
) -> tuple[str | None, list[dict[str, Any]]]:
    meas_col = _find_meas_col(df)
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        variables = {_base_name(c): _to_float(row[c]) for c in df.columns if c != meas_col and _base_name(c)}
        meas = _to_float(row[meas_col]) if meas_col else float("nan")
        if all(math.isnan(v) for v in variables.values()) and math.isnan(meas):
            continue
        theo, err = safe_eval_formula(formula, variables, constants, angle_unit)
        rows.append({"idx": i, "vars": variables, "meas": meas, "theo": theo, "err": err})
    return meas_col, rows


# ---------------------------------------------------------------------------
# 4. 매뉴얼 분석 엔진 (템플릿 + 범용 추출)
# ---------------------------------------------------------------------------

def _nan_rows(n_cols: int, n_rows: int = 3) -> list[list[float]]:
    return [[np.nan] * n_cols for _ in range(n_rows)]


# 템플릿은 '컬럼·상수·수식'만 제공하고 값은 채우지 않는다. example 은 버튼으로만 불러온다.
KNOWN_TEMPLATES: list[dict[str, Any]] = [
    {
        "key": "compton", "keys": ["컴프턴", "콤프턴", "compton"],
        "title": "컴프턴 산란 실험",
        "constants": {"E0": 661.7, "mc2": 511.0},
        "formulas": ["E0 / (1 + (E0/mc2) * (1 - cos(산란각)))", "1 / (1/E0 + (1 - cos(산란각))/mc2)"],
        "columns": ["산란각 [deg]", "산란체 유무 [1=O, 0=X]", f"{MEAS_KEY} [keV]"],
        "example": [[30.0, 1.0, 516.0], [60.0, 1.0, 379.0], [90.0, 1.0, 269.0]],
    },
    {
        "key": "em", "keys": ["비전하", "e/m", "헬름홀츠", "helmholtz", "전자의 전하"],
        "title": "전자의 비전하(e/m) 측정",
        "constants": {"N": 130.0, "R": 0.15, "mu0": 1.2566e-6},
        "formulas": [
            "2 * 가속전압 / ((0.7155 * mu0 * N * 코일전류 / R)**2 * 궤도반지름**2)",
            "2 * 가속전압 / (자기장**2 * 궤도반지름**2)",
        ],
        "columns": ["가속전압 [V]", "코일전류 [A]", "궤도반지름 [m]", f"{MEAS_KEY} [C/kg]"],
        "example": [[150.0, 1.35, 0.04, 1.72e11], [200.0, 1.55, 0.04, 1.70e11]],
    },
    {
        "key": "franck", "keys": ["프랑크", "franck", "헤르츠", "hertz"],
        "title": "프랑크-헤르츠 실험",
        "constants": {"E_Hg": 4.9},
        "formulas": ["피크 번호 * E_Hg"],
        "columns": ["피크 번호", f"{MEAS_KEY} [V]"],
        "example": [[1.0, 5.1], [2.0, 10.0], [3.0, 14.8]],
    },
    {
        "key": "photo", "keys": ["광전", "photoelectric", "플랑크 상수", "planck", "정지 전압", "저지 전압"],
        "title": "광전효과와 플랑크 상수 측정",
        "constants": {"h": 6.626e-34, "c": 2.998e8, "e": 1.602e-19, "W": 2.3},
        "formulas": ["h*c / (파장 * 1e-9 * e) - W"],
        "columns": ["파장 [nm]", f"{MEAS_KEY} [V]"],
        "example": [[365.0, 1.08], [405.0, 0.74], [436.0, 0.52], [546.0, 0.02]],
    },
    {
        "key": "millikan", "keys": ["밀리컨", "millikan", "기름방울", "유적"],
        "title": "밀리컨 기름방울 실험 (기본 전하량)",
        "constants": {"e_ref": 1.602e-19},
        "formulas": ["전자 개수 * e_ref"],
        "columns": ["전자 개수", f"{MEAS_KEY} [C]"],
        "example": [[1.0, 1.65e-19], [2.0, 3.15e-19], [3.0, 4.90e-19]],
    },
    {
        "key": "pendulum", "keys": ["단진자", "진자", "pendulum", "중력가속도"],
        "title": "단진자를 이용한 중력가속도 측정",
        "constants": {"g": 9.80665},
        "formulas": ["2 * pi * sqrt(길이 / g)", "4 * pi**2 * 길이 / 주기**2"],
        "columns": ["길이 [m]", f"{MEAS_KEY} [s]"],
        "example": [[0.5, 1.43], [0.8, 1.80], [1.0, 2.01]],
    },
    {
        "key": "freefall", "keys": ["자유낙하", "free fall", "낙하"],
        "title": "자유낙하 실험",
        "constants": {"g": 9.80665},
        "formulas": ["0.5 * g * 시간**2", "sqrt(2 * 높이 / g)"],
        "columns": ["시간 [s]", f"{MEAS_KEY} [m]"],
        "example": [[0.3, 0.45], [0.4, 0.80], [0.5, 1.24]],
    },
    {
        "key": "slit", "keys": ["이중슬릿", "단일슬릿", "회절", "간섭", "double slit", "diffraction", "interference"],
        "title": "빛의 간섭·회절 (슬릿 실험)",
        "constants": {"lam": 632.8},
        "formulas": ["차수 * (lam * 1e-9) * 스크린 거리 / (슬릿 간격 * 1e-3) * 1e3"],
        "columns": ["차수", "스크린 거리 [m]", "슬릿 간격 [mm]", f"{MEAS_KEY} [mm]"],
        "example": [[1.0, 1.0, 0.25, 2.5], [2.0, 1.0, 0.25, 5.1], [3.0, 1.0, 0.25, 7.6]],
    },
    {
        "key": "bragg", "keys": ["브래그", "bragg", "x선", "x-ray", "엑스선"],
        "title": "브래그 회절 (X선 파장 측정)",
        "constants": {"d": 0.2820},
        "formulas": ["2 * d * sin(입사각) / 차수"],
        "columns": ["차수", "입사각 [deg]", f"{MEAS_KEY} [nm]"],
        "example": [[1.0, 7.2, 0.0705], [2.0, 14.5, 0.0708]],
    },
    {
        "key": "michelson", "keys": ["마이컬슨", "michelson"],
        "title": "마이컬슨 간섭계 실험",
        "constants": {"lam": 632.8},
        "formulas": ["무늬 이동 수 * lam * 1e-3 / 2"],
        "columns": ["무늬 이동 수", f"{MEAS_KEY} [μm]"],
        "example": [[20.0, 6.4], [40.0, 12.5], [60.0, 19.1]],
    },
    {
        "key": "ohm", "keys": ["옴의 법칙", "ohm", "저항 측정"],
        "title": "옴의 법칙 검증",
        "constants": {"R": 100.0},
        "formulas": ["전압 / R"],
        "columns": ["전압 [V]", f"{MEAS_KEY} [A]"],
        "example": [[1.0, 0.0102], [2.0, 0.0199], [3.0, 0.0305]],
    },
    {
        "key": "rc", "keys": ["rc 회로", "rc회로", "축전기", "capacitor", "시간 상수", "시간상수"],
        "title": "RC 회로 충·방전 실험",
        "constants": {"V0": 5.0, "R": 1.0e4, "C": 1.0e-4},
        "formulas": ["V0 * exp(-시간 / (R * C))", "V0 * (1 - exp(-시간 / (R * C)))"],
        "columns": ["시간 [s]", f"{MEAS_KEY} [V]"],
        "example": [[0.5, 3.05], [1.0, 1.86], [2.0, 0.68]],
    },
    {
        "key": "hooke", "keys": ["훅", "hooke", "용수철", "스프링"],
        "title": "훅의 법칙 (용수철 상수 측정)",
        "constants": {"k": 25.0},
        "formulas": ["k * 늘어난 길이"],
        "columns": ["늘어난 길이 [m]", f"{MEAS_KEY} [N]"],
        "example": [[0.02, 0.49], [0.04, 1.02], [0.06, 1.47]],
    },
]

_GREEK = {
    "θ": "theta", "φ": "phi", "ϕ": "phi", "λ": "lam", "ω": "omega", "μ": "mu", "ρ": "rho",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "Delta", "ε": "epsilon",
    "σ": "sigma", "τ": "tau", "ν": "nu", "η": "eta", "κ": "kappa", "Ω": "Omega", "π": "pi",
}
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:\s*(?:\*\s*10\s*\^\s*[-+]?\d+|[eE][-+]?\d+))?"
_CONST_RE = re.compile(
    r"(?<![\w.^])([A-Za-z][A-Za-z0-9_]{0,7}(?:\^2)?)\s*=\s*(" + _NUM_RE + r")"
    r"(?![\w.])(?!\s*,\s*\d)(?!\s*[*^(√])(?!\s*(?:pi|sqrt|sin|cos|tan|exp|ln|log)\b)"
)
_CONST_STOP = {"Fig", "fig", "Table", "No", "p", "pp", "Vol", "Eq", "eq", "i", "j"}
_EQ_RE = re.compile(
    r"(?<![\w=<>!*/+\-])([A-Za-z][\w']{0,10}(?:/[A-Za-z]\w{0,3})?(?:\([^()]{1,10}\))?)\s*[=≈]\s*([^=\n≈]+)"
)
_UNIT_TOKENS = r"deg|rad|keV|MeV|eV|kV|mV|V|mA|A|km|cm|mm|nm|μm|um|m|ms|s|g|kg|C|N|mT|T|Hz|kHz|J|K|°C|°|Pa|W"
_UNIT_BY_NAME = {
    "theta": "deg", "phi": "deg", "alpha": "deg", "angle": "deg",
    "V": "V", "U": "V", "I": "A", "t": "s", "T": "s", "lam": "nm", "f": "Hz", "nu": "Hz",
    "m": "kg", "M": "kg", "d": "m", "r": "m", "R": "m", "L": "m", "l": "m", "x": "m", "y": "m",
    "h": "m", "s": "m", "B": "T", "F": "N", "E": "eV", "P": "Pa", "p": "Pa", "Q": "C", "q": "C",
}


def _normalize_text(text: str) -> str:
    t = re.sub(r"10\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: "10^" + m.group(1).translate(_SUPERSCRIPT), text)
    t = t.replace("²", "^2").replace("³", "^3")
    for src, dst in [("×", "*"), ("·", "*"), ("∙", "*"), ("⋅", "*"), ("÷", "/"), ("−", "-"),
                     ("–", "-"), ("＝", "="), ("（", "("), ("）", ")"), ("′", "'")]:
        t = t.replace(src, dst)
    for g, name in _GREEK.items():
        if g == "Δ":
            t = re.sub("Δ(?=\\w)", "Delta", t)
        t = re.sub(re.escape(g) + r"(?=[\d_])", name, t)
        t = t.replace(g, f" {name} ")
    return t


def _parse_number(s: str) -> float:
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\*10\^", "e", s)
    return float(s)


def _extract_constants(norm: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in _CONST_RE.finditer(norm):
        sym, raw = m.group(1).replace("^2", "2"), m.group(2)
        if sym in _CONST_STOP or sym in out:
            continue
        try:
            out[sym] = _parse_number(raw)
        except ValueError:
            continue
    return out


def _clean_rhs(rhs: str) -> str:
    rhs = re.split(r"[가-힣]", rhs, maxsplit=1)[0]
    rhs = re.split(r"\b(?:where|with|for|and|is|the)\b|[,;:]", rhs, maxsplit=1)[0]
    rhs = re.sub(r"\(\s*\d+\s*\)\s*$", "", rhs)
    return re.sub(r"[.\s]+$", "", rhs).strip()


def _formula_to_python(rhs: str) -> str:
    f = _to_python_expr(rhs)
    f = re.sub(r"\b(sin|cos|tan)\s*\*\*\s*2\s*\(?\s*([A-Za-z_]\w*)\s*\)?", r"\1(\2)**2", f)
    f = re.sub(r"\b(sin|cos|tan|sqrt|exp|ln|log)(?=[A-Za-z_])", r"\1 ", f)
    f = re.sub(r"\b(sin|cos|tan|sqrt|exp|ln|log)\s+([A-Za-z_]\w*)", r"\1(\2)", f)
    f = re.sub(r"(\d)\s*(?![eE][-+]?\d)([A-Za-z(])", r"\1*\2", f)
    f = re.sub(r"\)\s*([A-Za-z0-9(])", r")*\1", f)
    f = re.sub(r"([A-Za-z_]\w*)\s*\(",
               lambda m: m.group(0) if m.group(1) in _FUNC_NAMES else m.group(1) + "*(", f)
    f = re.sub(r"([\w)])\s+(?=[\w(])", r"\1*", f)
    return re.sub(r"\s+", "", f)


def _formula_names(pyexpr: str) -> list[str]:
    tree = ast.parse(pyexpr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SyntaxError("허용되지 않는 노드")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNC_NAMES):
            raise SyntaxError("허용되지 않는 함수")
    return [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _FUNC_NAMES]


def _extract_formulas(norm: str, constants: dict[str, float]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in norm.splitlines():
        for seg in re.split(r",\s*(?=[A-Za-z][\w'^/]{0,10}\s*[=≈])", line):
            for m in _EQ_RE.finditer(seg):
                lhs, rhs = m.group(1).strip(), _clean_rhs(m.group(2))
                if lhs in constants or len(rhs) < 3 or not re.search(r"[A-Za-z]", rhs):
                    continue
                if not re.search(r"[+\-*/^(√]", rhs):
                    continue
                try:
                    py = _formula_to_python(rhs)
                    names = list(dict.fromkeys(_formula_names(py)))
                except (SyntaxError, ValueError):
                    continue
                vars_ = [n for n in names if n not in constants]
                if not (1 <= len(vars_) <= 6) or any(len(n) > 12 for n in names) or py in seen:
                    continue
                seen.add(py)
                found.append({"lhs": lhs, "rhs": py, "vars": vars_})
    found.sort(key=lambda d: 0 if len(d["vars"]) <= 4 else 1)
    return found[:8]


def _guess_unit(name: str, norm: str) -> str:
    m = re.search(re.escape(name) + r"\s*[\(\[]\s*(" + _UNIT_TOKENS + r")\s*[\)\]]", norm)
    if m:
        return m.group(1)
    base = re.sub(r"\d+$", "", name)
    return _UNIT_BY_NAME.get(name, _UNIT_BY_NAME.get(base, ""))


def _extract_title(text: str) -> str:
    for line in text.splitlines()[:120]:
        s = line.strip()
        if not (3 <= len(s) <= 60):
            continue
        if re.search(r"(실험|Experiment|측정|Measurement)", s, re.I) and not re.search(
            r"(목적|방법|결과|이론|장치|과정|주의|참고|보고서|Purpose|Method|Result|Theory|Report)", s, re.I
        ):
            s = re.sub(r"^(실험|Exp\.?|Experiment|Lab)?\s*[\dIVX]+\s*[.\):\-]?\s*", "", s, flags=re.I).strip()
            if s:
                return s
    return ""


def _match_template(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    lower = text.lower()
    best, best_hits = None, []
    for tmpl in KNOWN_TEMPLATES:
        hits = [k for k in tmpl["keys"] if k in lower]
        if len(hits) > len(best_hits):
            best, best_hits = tmpl, hits
    return best, best_hits


def analyze_manual(text: str) -> dict[str, Any]:
    """매뉴얼 텍스트 → {key, title, constants, formulas, columns, data, example, desc, extracted}"""
    if not text.strip():
        return {
            "key": None, "title": "새 실험 (매뉴얼을 업로드하세요)", "constants": {}, "formulas": [""],
            "columns": ["변수 1", MEAS_KEY], "data": _nan_rows(2), "example": None,
            "desc": ("매뉴얼이 업로드되지 않아 빈 상태입니다. PDF를 올리면 실험 제목·상수·수식·표 형식을 즉시 추천합니다. "
                     "매뉴얼 없이 사용하려면 사이드바에서 상수를, 표 컬럼 관리에서 변수를 직접 추가하세요."),
            "extracted": [],
        }

    norm = _normalize_text(text)
    tmpl, hits = _match_template(text)
    consts = _extract_constants(norm)
    formulas = _extract_formulas(norm, consts)
    title = _extract_title(text)

    if tmpl:
        s = copy.deepcopy(tmpl)
        s.pop("keys", None)
        s["data"] = _nan_rows(len(s["columns"]))
        for k, v in consts.items():
            s["constants"].setdefault(k, v)
        for f in formulas:
            if f["rhs"] not in s["formulas"]:
                s["formulas"].append(f["rhs"])
        s["desc"] = (
            f"매뉴얼을 「{s['title']}」 실험으로 식별했습니다 (감지 키워드: {', '.join(hits)}). "
            f"본문에서 상수 {len(consts)}개, 수식 {len(formulas)}개를 추가로 추출해 추천 목록에 병합했습니다. "
            "표는 컬럼만 구성되며 값은 비어 있습니다 (예시 값은 '표 컬럼 관리'에서 불러올 수 있음)."
        )
    elif formulas:
        main = formulas[0]
        cols = []
        for v in main["vars"]:
            u = _guess_unit(v, norm)
            cols.append(f"{v} [{u}]" if u else v)
        meas_unit = _guess_unit(main["lhs"], norm)
        cols.append(f"{MEAS_KEY} [{meas_unit}]" if meas_unit else f"{MEAS_KEY} ({main['lhs']})")
        s = {
            "key": None, "title": title or "사용자 정의 실험", "constants": consts,
            "formulas": [f["rhs"] for f in formulas], "columns": cols, "data": _nan_rows(len(cols)), "example": None,
            "desc": (f"알려진 템플릿과 일치하지 않아 매뉴얼 본문에서 직접 추출했습니다 — 수식 {len(formulas)}개, 상수 {len(consts)}개. "
                     f"대표 수식 `{main['lhs']} = {main['rhs']}` 의 변수({', '.join(main['vars'])})로 표를 구성했습니다. 단위와 값은 확인 후 수정하세요."),
        }
    else:
        s = {
            "key": None, "title": title or "사용자 정의 실험", "constants": consts, "formulas": [""],
            "columns": ["변수 1", MEAS_KEY], "data": _nan_rows(2), "example": None,
            "desc": (f"매뉴얼에서 수식을 자동 추출하지 못했습니다 (상수 {len(consts)}개 추출). "
                     "PDF가 이미지 스캔본이면 텍스트가 없을 수 있습니다. 수식과 표 컬럼을 직접 설정해 주세요."),
        }
    s["extracted"] = formulas
    s["formulas"] = [f for f in s["formulas"] if f is not None]
    return s


@st.cache_data(show_spinner=False)
def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 5. 통계·감도 분석 및 리포트 생성
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    formula: str, variables: dict[str, float], constants: dict[str, float], angle_unit: str
) -> list[dict[str, Any]]:
    base, err = safe_eval_formula(formula, variables, constants, angle_unit)
    if err or base is None or base == 0:
        return []
    used = formula_symbols(formula, list(variables), list(constants))
    out = []
    for kind, names, pool in (("표 변수", used["vars"], variables), ("상수", used["consts"], constants)):
        for name in names:
            x = _to_float(pool[name])
            if math.isnan(x):
                continue
            h = abs(x) * 1e-4 if x != 0 else 1e-6
            v_plus, c_plus, v_minus, c_minus = dict(variables), dict(constants), dict(variables), dict(constants)
            (v_plus if kind == "표 변수" else c_plus)[name] = x + h
            (v_minus if kind == "표 변수" else c_minus)[name] = x - h
            f_p, e1 = safe_eval_formula(formula, v_plus, c_plus, angle_unit)
            f_m, e2 = safe_eval_formula(formula, v_minus, c_minus, angle_unit)
            if e1 or e2 or f_p is None or f_m is None:
                continue
            deriv = (f_p - f_m) / (2 * h)
            elasticity = deriv * x / base if x != 0 else float("nan")
            out.append({"kind": kind, "name": name, "value": x, "deriv": deriv, "elasticity": elasticity})
    out.sort(key=lambda d: -abs(d["elasticity"]) if not math.isnan(d["elasticity"]) else 0)
    return out


def compute_stats(theo: np.ndarray, meas: np.ndarray) -> dict[str, Any]:
    n = len(theo)
    stats: dict[str, Any] = {"n": n}
    if n == 0:
        return stats
    resid = theo - meas
    with np.errstate(divide="ignore", invalid="ignore"):
        err_pct = np.where(theo != 0, np.abs(resid) / np.abs(theo) * 100, np.nan)
    has = bool(np.any(~np.isnan(err_pct)))
    stats.update(
        theo_mean=float(np.mean(theo)), meas_mean=float(np.mean(meas)),
        err_pct=err_pct, mean_err=float(np.nanmean(err_pct)) if has else float("nan"),
        max_err=float(np.nanmax(err_pct)) if has else float("nan"),
        max_err_idx=int(np.nanargmax(err_pct)) if has else 0,
        rmse=float(np.sqrt(np.mean(resid ** 2))), resid=resid,
        bias=float(np.mean(resid)), same_sign=bool(n >= 3 and (np.all(resid > 0) or np.all(resid < 0))),
        r=float("nan"), slope=float("nan"), intercept=float("nan"),
    )
    if n >= 2 and np.std(theo) > 0 and np.std(meas) > 0:
        stats["r"] = float(np.corrcoef(theo, meas)[0, 1])
        slope, intercept = np.polyfit(theo, meas, 1)
        stats["slope"], stats["intercept"] = float(slope), float(intercept)
    return stats


def _grade(mean_err: float) -> str:
    if math.isnan(mean_err):
        return "판정 불가"
    if mean_err < 1:
        return "매우 우수 (평균 오차 1% 미만)"
    if mean_err < 5:
        return "우수 (평균 오차 5% 미만)"
    if mean_err < 10:
        return "양호 (평균 오차 10% 미만)"
    return "재검토 필요 (평균 오차 10% 이상)"


def _diagnose(stats: dict[str, Any]) -> list[str]:
    diag: list[str] = []
    if stats.get("n", 0) >= 2 and not math.isnan(stats.get("slope", float("nan"))):
        if abs(stats["slope"] - 1) > 0.05:
            diag.append(f"회귀 기울기 {stats['slope']:.3f}가 1에서 벗어나 **비례(이득) 오차**가 의심됩니다 — 교정 계수, 단위 환산(deg/rad, cm/m), 상수 값을 점검하세요.")
        if stats["theo_mean"] != 0 and abs(stats["intercept"]) > 0.05 * abs(stats["theo_mean"]):
            diag.append(f"회귀 절편 {stats['intercept']:.4g}가 0에서 유의하게 벗어나 **영점(offset) 오차**가 의심됩니다 — 측정기 영점, 배경 보정을 확인하세요.")
    if stats.get("same_sign"):
        direction = "이론값이 측정값보다 항상 큼" if stats["bias"] > 0 else "측정값이 이론값보다 항상 큼"
        diag.append(f"모든 잔차의 부호가 동일합니다 ({direction}) — 무작위 오차만으로는 설명되지 않는 **체계적 편향**이 존재합니다.")
    if not diag:
        diag.append("뚜렷한 비례·영점·부호 편향이 관찰되지 않아 **무작위(통계적) 오차**가 지배적인 것으로 판단됩니다.")
    return diag


def build_report(title: str, formula: str, const_str: str, stats: dict[str, Any], sens: list[dict[str, Any]]) -> dict[str, str]:
    n = stats["n"]
    diag = ["- " + d for d in _diagnose(stats)]
    summary = f"""
### [심층 학술 진단 리포트] {title}

**1. 이론적 틀 (Theoretical Framework)**
- **지배 방정식**: `{formula}`
- **적용 상수**: {const_str or '없음'}
- **데이터 수**: {n}개 행 (유효 계산 기준)
- **이론값 평균 / 측정값 평균**: {stats.get('theo_mean', float('nan')):.4g} / {stats.get('meas_mean', float('nan')):.4g}

**2. 정량 평가 (Quantitative Assessment)**
- **평균 오차율**: {stats.get('mean_err', float('nan')):.3f}%  → 판정: **{_grade(stats.get('mean_err', float('nan')))}**
- **최대 오차율**: {stats.get('max_err', float('nan')):.3f}% (행 #{stats.get('max_err_idx', 0) + 1})
- **RMSE (평균 제곱근 오차)**: {stats.get('rmse', float('nan')):.4g}
- **평균 잔차(이론 − 측정, 편향)**: {stats.get('bias', float('nan')):.4g}
- **상관계수 r**: {stats.get('r', float('nan')):.4f} / **선형 회귀** 측정 = {stats.get('slope', float('nan')):.4f}·이론 + {stats.get('intercept', float('nan')):.4g}

**3. 진단 (Diagnosis)**
{chr(10).join(diag)}
"""
    if sens:
        rows = "\n".join(
            f"| {d['kind']} | `{d['name']}` | {d['value']:.4g} | {d['deriv']:.4g} | {d['elasticity']:+.3f} |" for d in sens
        )
        top = sens[0]
        sens_md = f"""
**감도 분석 (Sensitivity / Uncertainty Propagation)**

대표 행(유효 데이터의 평균값)에서 수치 편미분 ∂f/∂x 를 계산했습니다. **탄성도**는 해당 변수가 1% 변할 때 결과값이 몇 % 변하는지를 의미하며,
1차 불확도 전파식 $u(f)^2=\\sum_i (\\partial f/\\partial x_i)^2\\,u(x_i)^2$ 의 가중치에 해당합니다.

| 종류 | 기호 | 대입값 | ∂f/∂x | 탄성도 |
|---|---|---|---|---|
{rows}

- 결과에 가장 민감한 양은 **`{top['name']}`** (탄성도 {top['elasticity']:+.3f}) 입니다. 이 양의 측정 정밀도가 최종 불확도를 지배합니다.
- 탄성도 절댓값이 1보다 크면 오차가 **증폭**되고, 1보다 작으면 **완화**됩니다.
"""
    else:
        sens_md = "감도 분석을 수행할 유효한 행이 없습니다."

    improve = f"""
**4. 오차 원인 분류 (Error Taxonomy)**
- **기기·교정 오차**: 분해능, 영점 이탈, 이득(gain) 교정 곡선의 비선형성 → 회귀 기울기·절편 진단 참고.
- **기하·정렬 오차**: 검출기/광원/시료 정렬 불량, 유효 입체각·유효 길이의 유한 크기 효과.
- **환경 오차**: 온도·습도·배경 잡음(자연 방사선, 외부 자기장, 진동, 주변광).
- **통계 오차**: 반복 횟수·계수 시간 부족에 따른 무작위 변동 (표본 수 n = {n}).
- **모델 가정 오차**: 이론식의 이상화(질점, 무마찰, 단일 산란, 균일장 등)와 실제 조건의 차이.

**5. 개선 방안 (Recommendations)**
- {'체계적 편향이 관찰되므로, 알려진 참값(표준 시료·기준 선원)으로 **다점 교정**을 먼저 수행하세요.' if stats.get('same_sign') else '무작위 오차가 주도하므로 **반복 측정 횟수를 늘리고 평균·표준오차**를 함께 보고하세요.'}
- 최대 오차가 발생한 행 #{stats.get('max_err_idx', 0) + 1} 의 측정 조건(범위 끝단, 낮은 신호 등)을 재점검하고 필요 시 재측정하세요.
- 감도 분석 상위 변수를 우선 정밀 측정하고 불확도 예산(uncertainty budget)에 반영하세요.
- 배경(background)을 동일 조건에서 별도 측정하여 차감하고, 단위 환산(각도 단위 포함)을 재확인하세요.
"""
    return {"summary": summary, "sensitivity": sens_md, "improve": improve}


# ---------------------------------------------------------------------------
# 6. AI 멘토 지식 베이스
#    각 섹션: id, title, tl(한 줄 요약), keys(질문 매칭 키워드, 소문자), body(마크다운·LaTeX)
#    참고 자료는 실제로 존재하는 안정적 URL(위키백과·HyperPhysics·원논문 DOI·표준 문서)만 사용한다.
# ---------------------------------------------------------------------------

def S(id_: str, title: str, tl: str, keys: list[str], body: str) -> dict[str, Any]:
    return {"id": id_, "title": title, "tl": tl, "keys": [k.lower() for k in keys], "body": body.strip()}


KB: dict[str, dict[str, Any]] = {}

KB["compton"] = {
    "name": "컴프턴 산란",
    "sections": [
        S("principle", "컴프턴 산란의 기본 원리와 공식 유도",
          "광자를 E=hν, p=h/λ 인 입자로 보고 정지 전자와의 탄성 충돌에 에너지·운동량 보존을 적용하면 Δλ=(h/mc)(1−cosθ) 가 나온다.",
          ["원리", "유도", "공식", "이론", "왜", "principle", "derivation", "파장", "에너지", "산란각", "컴프턴 파장", "설명"],
          r"""
광자를 에너지 $E=h\nu=hc/\lambda$, 운동량 $p=E/c=h/\lambda$ 를 갖는 **입자**로 보고, 정지한 자유 전자(정지 에너지 $mc^2=511\ \mathrm{keV}$)와의 **탄성 충돌**로 다룹니다.

- **에너지 보존**: $E_0 + mc^2 = E' + \sqrt{p_e^2c^2 + m^2c^4}$
- **운동량 보존** (입사 방향 $x$, 수직 방향 $y$):
  $\dfrac{E_0}{c} = \dfrac{E'}{c}\cos\theta + p_e\cos\phi,\qquad 0 = \dfrac{E'}{c}\sin\theta - p_e\sin\phi$

두 운동량 식을 제곱하여 더하면 $\phi$ 가 소거되어 $p_e^2c^2 = E_0^2 + E'^2 - 2E_0E'\cos\theta$ 가 되고, 이를 에너지 보존식의 제곱에 대입해 정리하면

$$E'(\theta)=\frac{E_0}{1+\dfrac{E_0}{mc^2}(1-\cos\theta)},\qquad \lambda'-\lambda=\frac{h}{mc}(1-\cos\theta)$$

여기서 $\lambda_C = h/mc = 2.426\times10^{-12}\ \mathrm{m}$ 를 **컴프턴 파장**이라 합니다. 파장 이동 $\Delta\lambda$ 는 입사 파장·산란체 물질·빔 세기와 무관하고 **오직 산란각 $\theta$ 에만** 의존한다는 점이 핵심입니다.

**각도별 특징 (Cs-137, $E_0=661.7$ keV 기준)**
- $\theta=0^\circ$: $\Delta\lambda=0$ → 에너지 변화 없음
- $\theta=90^\circ$: $\Delta\lambda=\lambda_C$ → $E'\approx288$ keV
- $\theta=180^\circ$(후방산란): $\Delta\lambda=2\lambda_C$, 최소 에너지 $E'_{\min}=E_0/(1+2E_0/mc^2)\approx184$ keV
- 전자가 받는 최대 반동 에너지(컴프턴 모서리) $T_{\max}=E_0-E'_{\min}\approx477$ keV

**유도에서 주의할 점**: 전자의 운동에너지는 상대론적으로 다뤄야 합니다(반동 전자 속도가 $c$ 의 수십 % 에 이름). 비상대론적 $\tfrac12 mv^2$ 를 쓰면 공식이 틀립니다. 또 "자유·정지 전자" 가정은 외각 전자에 대해 좋은 근사이며, 속박이 강한 내각 전자는 원자 전체가 반동하여 $m\to M_{atom}$ 이 되므로 $\Delta\lambda\to0$ (이동하지 않은 피크, 레일리/코히런트 산란)이 됩니다.
"""),
        S("momentum_particle", "운동량 보존 검증과 빛의 입자성(광양자설) 증명으로서의 의미",
          "파동론은 파장 이동을 설명할 수 없고, 광자에 운동량 p=h/λ 를 부여한 2체 충돌 모형만이 관측을 정확히 재현한다. 1/E' 대 (1−cosθ) 의 직선성이 그 검증이다.",
          ["운동량", "보존", "입자성", "광양자", "광자", "증명", "검증", "파동", "이중성", "톰슨", "역사", "노벨", "의의", "quantum", "particle", "momentum"],
          r"""
**1) 고전 파동론(톰슨 산란)의 예측** — 전자기파가 전자를 흔들면 전자는 입사파와 같은 진동수로 진동하며 재복사합니다. 따라서 산란광의 파장은 입사광과 **같아야** 하고, 만약 변한다면 빔 세기나 노출 시간에 따라 연속적으로 변해야 합니다. 세기가 약하면 전자가 충분한 에너지를 얻는 데 시간이 걸려야 한다는 예측도 따라옵니다.

**2) 컴프턴(1923)의 관측** — 몰리브덴 $K_\alpha$ X선($\lambda=0.0709$ nm)을 그래파이트에 쏘았을 때 산란광에 **정확히 $(h/mc)(1-\cos\theta)$ 만큼 파장이 길어진 성분**이 나타났습니다. 이동량은 빔 세기·노출 시간·산란체 물질에 무관하고 각도만의 함수였습니다. 이는 파동론으로는 설명이 불가능하며, 광자가 $E=h\nu$ 와 **운동량 $p=h/\lambda$** 를 갖는 알갱이로서 전자와 당구공처럼 충돌한다고 봐야만 설명됩니다.

광전효과(1905)는 "빛 에너지의 양자화"를 보여 주었지만, 에너지 양자화만으로는 광자가 **운동량을 가진 실체**인지 확정할 수 없었습니다. 컴프턴 산란은 운동량 보존이 광자–전자 사이에 성립함을 보여 **광자를 역학적 입자로** 확립한 결정적 실험이며, 컴프턴은 1927년 노벨 물리학상을 받았습니다. 이후 드브로이의 물질파($\lambda=h/p$, 1924)도 같은 관계식을 물질 쪽으로 확장한 것입니다.

**3) '사건 단위' 운동량 보존** — 보어·크라머스·슬레이터(BKS, 1924)는 개별 미시 사건에서는 에너지·운동량이 **통계적으로만** 보존된다고 주장했습니다. 그러나 보테–가이거(1925)의 동시계수 실험과 컴프턴–사이먼의 구름상자 실험은 산란 광자와 반동 전자가 **매 사건마다 동시에, 공식이 예측한 각도 관계 $\cot\phi=(1+E_0/mc^2)\tan(\theta/2)$ 로** 나타남을 보여, 개별 충돌에서 보존 법칙이 엄격히 성립함을 확인했습니다. 이것이 BKS 이론을 폐기시키고 양자역학 성립(1925–26)의 배경이 됩니다.

**4) 이 실험이 두 가지를 '검증'하는 방식**
1. 여러 산란각 $\theta$ 에서 산란 광자의 광전 피크 에너지 $E'$ 를 측정합니다.
2. 공식을 선형화합니다: $\dfrac{1}{E'}=\dfrac{1}{E_0}+\dfrac{1}{mc^2}\,(1-\cos\theta)$. $x=(1-\cos\theta)$, $y=1/E'$ 로 그래프를 그립니다.
3. **직선성 자체**가 "광자–전자 2체 충돌에서 에너지·운동량이 동시에 보존된다"는 검증입니다. 보존 법칙 중 하나라도 어긋나면 $1/E'$ 는 $(1-\cos\theta)$ 에 선형이 될 이유가 없습니다.
4. **기울기의 역수**가 전자 정지 에너지 $mc^2=511$ keV 와 일치하면, 광자 운동량이 $p=E/c$ 임(입자성)이 확인됩니다. 만약 광자에 운동량이 없다고 가정하면 기울기는 0 이어야 합니다. **절편의 역수**는 $E_0$ 를 되돌려 주어 교정 검증까지 겸합니다.
5. 스펙트럼에 이동하지 않은 피크가 함께 보이는 것은 속박 전자(원자 전체 반동)에 같은 공식을 적용한 결과이므로, 이 역시 모형의 일관성을 지지합니다.
"""),
        S("errors", "오차 원인 — 고각도 오차, 다중 산란, 검출기·교정 효과",
          "각도 퍼짐은 dE'/dθ ∝ E'^2 sinθ 로 전파되고, 두꺼운 산란체의 다중 산란은 피크를 고에너지로 밀며, 고각도에서는 단면적 감소로 통계 오차가 커진다.",
          ["오차", "고각도", "다중 산란", "분해능", "교정", "채널", "mca", "배경", "통계", "왜 차이", "error", "편향", "체계"],
          r"""
- **유한 입체각(각도 분해능)**: 선원 콜리메이터·산란체·검출기 크기 때문에 실제 산란각은 $\theta\pm\Delta\theta$ 로 퍼집니다. $\dfrac{dE'}{d\theta}=-\dfrac{E'^2}{mc^2}\sin\theta$ 이므로 각도 오차의 에너지 전파는 $E'^2\sin\theta$ 에 비례하여 **중간 각도(40–70°)에서 가장 크고**, 고각도에서는 $E'$ 가 작아 절대 오차는 줄지만 상대 오차와 계수율 감소가 문제됩니다. 또한 유한 입체각은 $E'(\theta)$ 가 볼록 함수이므로 피크 중심을 **약간 고에너지 쪽으로** 치우치게 합니다.
- **다중 산란**: 두꺼운 산란체에서는 두 번 이상 산란된 광자가 검출됩니다. 작은 각도로 두 번 산란되면 같은 총 각도의 단일 산란보다 에너지가 높으므로 피크가 **고에너지 쪽으로 치우치고 넓어집니다**. 산란체 직경을 줄이거나 저원자번호(Al) 재료를 쓰면 완화됩니다.
- **클라인–니시나 단면적**: 산란 세기는 각도가 커질수록 감소하여 고각도 통계가 나빠집니다. 상대 통계 오차는 $1/\sqrt{N}$ 이므로 계수 시간을 각도별로 늘려 피크 계수 $N$ 을 확보해야 합니다.
- **검출기 에너지 분해능**: NaI(Tl)는 662 keV 에서 FWHM ≈ 7 %. 피크 중심(centroid)을 가우시안 피팅으로 결정하면 채널 판독 오차를 크게 줄일 수 있습니다.
- **MCA 채널–에너지 교정의 비선형성**: 두 점 교정만 하면 중간 에너지에서 수 keV 편향이 생깁니다. Ba-133(356 keV), Na-22(511 keV), Co-60(1173/1332 keV) 등 **다점 교정**을 권장합니다. 회귀 기울기가 1에서 벗어나면 교정 이득 오차, 절편이 0에서 벗어나면 영점 오차를 의심합니다.
- **도플러 넓어짐**: 속박 전자의 운동 때문에 산란 에너지가 퍼집니다(피크 폭 증가, 중심은 거의 불변).
- **배경**: 산란체를 제거한 상태에서 같은 live-time 으로 측정한 스펙트럼을 빼야 직접 투과·실내 산란·자연 방사선 성분을 제거할 수 있습니다.
- **선원 감쇠·전자장치 드리프트**: 장시간 측정 시 PMT 고전압·온도 변화로 이득이 흘러가므로 측정 전후로 교정 피크 위치를 재확인해야 합니다.
"""),
        S("spectrum", "감마 스펙트럼 읽는 법 — 광전 피크·컴프턴 연속·모서리·후방산란 피크",
          "측정값으로 읽는 것은 산란 후 광자의 광전 피크 중심이며, 컴프턴 모서리(477 keV)와 후방산란 피크(184 keV)의 합이 E0 가 되는지로 교정을 검증할 수 있다.",
          ["스펙트럼", "피크", "모서리", "edge", "후방산란", "연속", "광전 피크", "spectrum", "nai", "검출기"],
          r"""
- **광전 피크(전에너지 피크)**: 광자 에너지 전체가 검출기에 흡수된 사건. 산란 실험에서 "측정값"으로 읽는 것은 **산란 후 광자의 광전 피크 중심**입니다.
- **컴프턴 연속체**: 검출기 내부에서 컴프턴 산란 후 광자가 빠져나간 사건으로, 반동 전자 에너지 $T=E_0-E'$ 만 남습니다.
- **컴프턴 모서리**: 검출기 내 $180^\circ$ 산란에 해당하는 최대 반동 에너지 $T_{\max}=\dfrac{2E_0^2/mc^2}{1+2E_0/mc^2}$. Cs-137: 477 keV.
- **후방산란 피크**: 검출기 주변 물질(차폐, 테이블)에서 $180^\circ$ 산란된 광자가 들어온 것. $E'_{\min}\approx184$ keV.
- **교정 자가 검증**: $T_{\max}+E'_{\min}=E_0$ 가 성립해야 합니다. 두 값의 합이 661.7 keV 에서 벗어나면 채널–에너지 교정이 틀린 것입니다.
- **X선 피크(~32 keV)**: Cs-137 붕괴 후 Ba-137 의 K X선. 저에너지 교정점으로 활용 가능합니다.
"""),
        S("cross_section", "산란 세기의 각도 의존 — 톰슨 단면적 vs 클라인–니시나 공식",
          "자유 전자 미분 단면적은 클라인–니시나 공식으로 주어지며 저에너지 한계에서 톰슨 단면적으로 환원된다. 662 keV 에서는 전방 산란이 우세하다.",
          ["단면적", "세기", "강도", "클라인", "니시나", "klein", "nishina", "계수율", "확률", "intensity", "cross section"],
          r"""
자유 전자에 대한 미분 단면적은 QED 로부터 유도된 **클라인–니시나 공식**

$$\frac{d\sigma}{d\Omega}=\frac{r_e^2}{2}\left(\frac{E'}{E_0}\right)^2\left[\frac{E'}{E_0}+\frac{E_0}{E'}-\sin^2\theta\right]$$

로 주어집니다($r_e=e^2/4\pi\varepsilon_0mc^2=2.818$ fm, 고전 전자 반지름). 저에너지 한계 $E_0\ll mc^2$ 에서 $E'\to E_0$ 이면 톰슨 단면적 $\tfrac{r_e^2}{2}(1+\cos^2\theta)$ 로 환원되어 전·후방 대칭이 됩니다. 반면 662 keV 에서는 $(E'/E_0)^2$ 인자 때문에 **전방 산란이 강하게 우세**하고 $90^\circ$ 이상에서는 계수율이 수 배 낮아집니다. 따라서 각도별 측정 시간을 단면적에 반비례하도록 배분하는 것이 통계적으로 효율적입니다. 산란체의 전자 수(밀도×$Z/A$)에 비례하여 세기가 커지므로, 산란체 재질을 바꿔 세기를 비교하면 "전자 1개당 산란"이라는 모형도 검증할 수 있습니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Compton Scattering (공식 유도·계산기)", "http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/comptint.html", "강의노트"),
        ("위키백과(한국어) – 콤프턴 산란", "https://ko.wikipedia.org/wiki/콤프턴_산란", "백과사전"),
        ("Wikipedia – Compton scattering", "https://en.wikipedia.org/wiki/Compton_scattering", "백과사전"),
        ("A. H. Compton, “A Quantum Theory of the Scattering of X-rays by Light Elements”, Phys. Rev. 21, 483 (1923) — 원논문", "https://journals.aps.org/pr/abstract/10.1103/PhysRev.21.483", "논문"),
        ("Wikipedia – Klein–Nishina formula", "https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula", "백과사전"),
        ("Wikipedia – BKS theory (통계적 보존 논쟁과 보테–가이거 실험)", "https://en.wikipedia.org/wiki/BKS_theory", "백과사전"),
        ("Feynman Lectures Vol. III Ch. 1 – Quantum Behavior", "https://www.feynmanlectures.caltech.edu/III_01.html", "교재"),
        ("MIT OCW 8.04 Quantum Physics I (강의 영상·노트)", "https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2016/", "강의"),
    ],
    "yt": ["컴프턴 산란 공식 유도", "Compton scattering derivation", "Compton scattering experiment NaI detector Cs-137", "gamma spectrum Compton edge backscatter peak"],
}

KB["em"] = {
    "name": "전자의 비전하 e/m",
    "sections": [
        S("principle", "e/m 측정 원리 — 로런츠 힘과 헬름홀츠 코일",
          "eV=½mv² 와 evB=mv²/r 에서 v 를 소거하면 e/m=2V/(B²r²) 이고, 헬름홀츠 코일 중심장은 B=(4/5)^{3/2}μ0NI/R 이다.",
          ["원리", "유도", "공식", "이론", "로런츠", "lorentz", "헬름홀츠", "helmholtz", "자기장", "반지름", "왜", "설명"],
          r"""
음극에서 가속 전압 $V$ 로 가속된 전자는 $eV=\tfrac12 mv^2$ 의 속도를 얻습니다. 균일 자기장 $B$ 에 수직으로 입사하면 로런츠 힘 $F=evB$ 가 구심력이 되어 $evB=mv^2/r$, 즉 $r=mv/(eB)$ 로 원운동합니다. 두 식에서 $v$ 를 소거하면

$$\frac{e}{m}=\frac{2V}{B^2r^2}.$$

헬름홀츠 코일(반지름 $R$, 감은 수 $N$, 간격 $=R$) 중심의 자기장은 두 코일 장의 합에서 2차 미분항이 상쇄되어 매우 균일하며

$$B=\left(\frac{4}{5}\right)^{3/2}\frac{\mu_0 N I}{R}\approx 0.7155\,\frac{\mu_0 N I}{R}.$$

참값 $e/m=1.7588\times10^{11}\ \mathrm{C/kg}$. 불확도 전파는 $\dfrac{\delta(e/m)}{e/m}=\sqrt{\left(\dfrac{\delta V}{V}\right)^2+\left(2\dfrac{\delta B}{B}\right)^2+\left(2\dfrac{\delta r}{r}\right)^2}$ 이므로 **반지름과 전류(→B) 측정이 전압보다 두 배 민감**합니다.

**역사적 의의**: J. J. 톰슨(1897)은 음극선의 $e/m$ 이 수소 이온보다 약 1800배 크다는 것을 보여 "원자보다 작은 입자(전자)"의 존재를 확립했습니다. 이후 밀리컨이 $e$ 를 독립 측정하면서 전자 질량 $m=9.109\times10^{-31}$ kg 이 결정되었습니다.
"""),
        S("errors", "오차 원인 — 지자기, 기체 충돌, 시차, 비균일장",
          "지구 자기장은 코일장에 벡터 합으로 더해지고, 관 내 기체 충돌은 전자를 감속시켜 r 을 줄여 e/m 을 과대평가하게 한다.",
          ["오차", "지구 자기장", "지자기", "기체", "충돌", "시차", "parallax", "비균일", "error", "차이", "크게", "작게"],
          r"""
- **지구 자기장(약 50 μT)**: 코일 축을 지자기 수평 성분과 평행하게 맞추거나, 전류 방향을 반전시켜 두 반지름의 평균을 취해 상쇄합니다. 무시하면 $B$ 가 수 % 틀립니다.
- **관 내 기체 충돌**: 전자빔이 보이는 것 자체가 He/Hg 기체 여기 때문입니다. 충돌로 전자가 감속되면 실제 $v$ 가 작아져 $r$ 이 작아지고, 공식은 $V$ 를 그대로 쓰므로 **$e/m$ 이 과대평가**됩니다. 큰 $V$·작은 $r$ 조건일수록 심합니다.
- **궤도 반지름 판독 시차(parallax)**: 거울 눈금 또는 자기 반사 눈금을 사용하고, 빔의 바깥쪽(가장 빠른 전자)을 기준으로 읽습니다.
- **필드 비균일성**: 빔이 코일 중심 평면에서 벗어나면 $B$ 가 감소합니다. 궤도가 클수록 편차가 커집니다.
- **음극 초기 속도·공간전하**: 열전자의 초기 에너지(~0.1 eV)와 접촉 전위가 $V$ 에 더해집니다. $V$ 가 작을 때 상대 영향이 큽니다.
- **점검법**: $r$ 을 고정하고 $V$–$I^2$ 그래프를 그리면 원점을 지나는 직선이어야 하며, 기울기에서 $e/m$ 을 얻습니다. 절편이 0이 아니면 지자기 또는 접촉 전위 효과가 있다는 신호입니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Motion of a Charged Particle in a Magnetic Field", "http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/movchg.html", "강의노트"),
        ("HyperPhysics – Helmholtz Coil", "http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/helmholtz.html", "강의노트"),
        ("Wikipedia – Mass-to-charge ratio", "https://en.wikipedia.org/wiki/Mass-to-charge_ratio", "백과사전"),
        ("Wikipedia – J. J. Thomson (1897 전자 발견)", "https://en.wikipedia.org/wiki/J._J._Thomson", "백과사전"),
        ("Wikipedia – Helmholtz coil", "https://en.wikipedia.org/wiki/Helmholtz_coil", "백과사전"),
    ],
    "yt": ["전자의 비전하 측정 실험", "e/m of electron Helmholtz coil experiment", "Thomson cathode ray e/m"],
}

KB["franck"] = {
    "name": "프랑크–헤르츠",
    "sections": [
        S("principle", "프랑크–헤르츠 실험의 원리 — 원자 에너지 준위의 불연속성",
          "전자 에너지가 수은의 첫 들뜸 에너지 4.9 eV 에 이를 때마다 비탄성 충돌로 전류가 급감하여 약 4.9 V 간격의 극대·극소가 반복된다.",
          ["원리", "유도", "이론", "준위", "불연속", "양자화", "비탄성", "4.9", "봉우리", "피크", "간격", "왜", "설명", "보어"],
          r"""
가열된 음극에서 나온 전자가 가속 전압 $U_a$ 로 가속되며 수은 증기 원자와 충돌합니다. 전자 에너지가 첫 들뜸 에너지 $E_1=4.9\ \mathrm{eV}$ ($6^1S_0\to6^3P_1$) 보다 작으면 충돌은 **탄성**이어서(질량비 $m_e/M_{Hg}\sim10^{-5}$ 로 에너지 전달 무시) 전자는 에너지를 잃지 않고 컬렉터에 도달해 전류가 증가합니다. 에너지가 4.9 eV 에 이르면 **비탄성 충돌**로 에너지를 원자에 넘겨주고, 느려진 전자는 저지 전압($\approx1.5$ V)을 넘지 못해 전류가 급감합니다. 전압을 더 올리면 전자는 재가속되어 두 번, 세 번 비탄성 충돌을 일으키고, 전류 곡선에 **약 4.9 V 간격의 극대·극소**가 반복됩니다.

$$U_{n+1}-U_n=\frac{E_1}{e}\approx4.9\ \mathrm{V},\qquad \lambda=\frac{hc}{E_1}\approx253.7\ \mathrm{nm}$$

들뜬 수은 원자가 방출하는 253.7 nm 자외선을 실제로 관측한 것이 결정적 증거였습니다. 1914년 실험은 보어 모형(1913)이 주장한 **양자화된 에너지 준위**를 광학 분광이 아닌 **전자 충돌**로 독립 확인한 것으로, 1925년 노벨 물리학상을 받았습니다. 원자가 "임의의 에너지"를 받을 수 있다면 전류는 단조 증가해야 하므로, 주기적 감소 자체가 불연속 준위의 직접 증거입니다.
"""),
        S("errors", "오차 원인 — 접촉 전위, 온도(평균 자유 행로), 저지 전압, 봉우리 판독",
          "첫 극소가 4.9 V 가 아닌 것은 접촉 전위차로 곡선 전체가 평행이동하기 때문이며, 물리적 의미는 간격에만 있다.",
          ["오차", "접촉 전위", "온도", "증기압", "평균 자유", "저지", "첫", "위치", "이동", "error", "차이", "높은 전압"],
          r"""
- **첫 극소가 4.9 V 에 있지 않은 이유**: 음극과 양극의 일함수 차이(접촉 전위차, 1–2 V)와 전자 초기 에너지 때문에 곡선 전체가 **평행이동**합니다. 절대 위치가 아닌 **간격**만이 $E_1$ 을 담고 있으므로, 봉우리 번호 $n$ 대 $U_n$ 을 선형 회귀하여 기울기 = 4.9 V 를 얻고 절편은 접촉 전위로 해석합니다.
- **온도**: 수은 증기압이 평균 자유 행로 $\ell$ 을 결정합니다. 온도가 낮으면(적은 원자) 전자가 4.9 eV 를 훨씬 넘겨 더 높은 준위(6³P₂ 5.46 eV, 6¹P₁ 6.7 eV)까지 들뜨게 하여 봉우리가 뭉개지고, 너무 높으면 전류가 작아 신호가 약해집니다. 보통 170–200 °C.
- **저지 전압**: 크면 봉우리 대비가 좋아지지만 전체 전류가 작아지고, 너무 크면 봉우리가 사라집니다.
- **높은 전압에서 간격 증가**: 전압이 높을수록 전자가 4.9 eV 를 넘긴 상태에서 충돌하는 확률이 커져 간격이 4.9 V 보다 조금 커지는 경향이 있습니다(5.0–5.2 V). 낮은 차수 봉우리를 위주로 평가하거나 이 경향을 보고서에 설명해야 합니다.
- **봉우리 위치 판독**: 극대와 극소를 모두 사용하고, 좌우 변곡점의 중점 또는 국소 다항 피팅으로 정점을 결정하면 판독 오차가 줄어듭니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Franck–Hertz Experiment", "http://hyperphysics.phy-astr.gsu.edu/hbase/FrHz.html", "강의노트"),
        ("Wikipedia – Franck–Hertz experiment", "https://en.wikipedia.org/wiki/Franck%E2%80%93Hertz_experiment", "백과사전"),
        ("Wikipedia – Bohr model", "https://en.wikipedia.org/wiki/Bohr_model", "백과사전"),
        ("Nobel Prize 1925 – Franck & Hertz", "https://www.nobelprize.org/prizes/physics/1925/summary/", "노벨재단"),
    ],
    "yt": ["프랑크 헤르츠 실험 원리", "Franck-Hertz experiment explained", "Franck Hertz mercury 4.9 eV curve"],
}

KB["photo"] = {
    "name": "광전효과",
    "sections": [
        S("principle", "광전효과와 아인슈타인 방정식 — 플랑크 상수 측정 원리",
          "eV_s = hν − W 이므로 정지 전압을 진동수에 대해 그리면 기울기 h/e, 절편 −W/e 이다. 세기 무관성·문턱 진동수·지연 없음은 파동론과 모순된다.",
          ["원리", "유도", "아인슈타인", "einstein", "플랑크", "planck", "정지 전압", "저지 전압", "일함수", "문턱", "진동수", "왜", "설명", "이론", "입자성", "광양자"],
          r"""
빛을 진동수 $\nu$ 인 광자의 흐름으로 보면, 금속 표면 전자는 광자 하나를 흡수해 에너지 $h\nu$ 를 얻고 일함수 $W$ 를 지불하여 탈출합니다.

$$K_{\max}=h\nu-W,\qquad eV_s=h\nu-W\ \Rightarrow\ V_s=\frac{h}{e}\,\nu-\frac{W}{e}$$

여기서 $V_s$ 는 광전류를 0으로 만드는 **정지(저지) 전압**입니다. 여러 파장(진동수)에서 $V_s$ 를 측정해 $V_s$–$\nu$ 그래프를 그리면 **기울기 = $h/e$**, **절편 = $-W/e$**, x축 교점 = 문턱 진동수 $\nu_0=W/h$ 입니다. $e=1.602\times10^{-19}$ C 를 곱하면 $h$ 를 얻습니다(참값 $6.626\times10^{-34}$ J·s).

**파동론과의 모순 3가지** — (1) 광전자 최대 에너지가 빛의 **세기와 무관**하고 진동수에만 의존, (2) 문턱 진동수 아래에서는 세기가 아무리 커도 방출 없음, (3) 매우 약한 빛에도 **지연 없이** 즉시 방출. 파동론은 에너지가 연속적으로 축적된다고 보므로 셋 모두를 설명하지 못합니다. 아인슈타인(1905)은 플랑크의 양자 가설을 빛 자체에 적용해 이를 설명했고 1921년 노벨상을 받았습니다. 밀리컨(1916)은 이 식을 반증하려 10년간 정밀 측정했으나 결국 $h$ 값이 플랑크 상수와 일치함을 확인했습니다.

파장 단위 변환에 주의: $\nu=c/\lambda$, $\lambda$ 가 nm 이면 $\nu=2.998\times10^{17}/\lambda_{\mathrm{nm}}$ Hz. 광자 에너지는 $E[\mathrm{eV}]=1239.84/\lambda[\mathrm{nm}]$ 로 빠르게 계산할 수 있습니다.
"""),
        S("errors", "오차 원인 — 역전류, 접촉 전위, 누설광, 절편 판정",
          "양극에서의 광전자 방출(역전류)은 I–V 곡선 꼬리를 만들어 정지 전압 판정을 어렵게 하며, 접촉 전위는 절편만 이동시키고 기울기(h/e)는 보존한다.",
          ["오차", "역전류", "접촉", "누설", "필터", "암전류", "절편", "판정", "error", "차이", "기울기"],
          r"""
- **역전류(reverse current)**: 산란광이 양극(컬렉터)에 닿아 방출된 전자가 반대로 흐르면 I–V 곡선이 0 에 도달하지 않고 음의 꼬리를 만듭니다. 정지 전압은 "전류 0" 이 아니라 **곡선이 포화 역전류에서 꺾이는 점**으로 판정하거나, 꼬리를 직선으로 외삽해 교점을 씁니다.
- **접촉 전위차**: 음극·양극 일함수 차이만큼 모든 $V_s$ 가 평행이동합니다. 따라서 절편에서 얻는 $W$ 는 신뢰도가 낮지만, **기울기 $h/e$ 는 영향받지 않습니다**.
- **필터 대역폭·누설광**: 간섭 필터의 반치폭(~10 nm)이 진동수 불확도가 되고, 형광등 등 주변광은 잡음 전류를 만듭니다. 암실 조건과 암전류 차감이 필요합니다.
- **광전류 판독 시 세기 의존 착시**: 세기를 바꾸면 $V_s$ 가 변해 보이는 것은 대부분 역전류·누설 때문이며, 이상적으로는 불변입니다. 이 불변성 확인 자체가 보고서의 핵심 논거가 됩니다.
- **단위 환산 실수**: nm→Hz 변환, eV↔J 변환(1 eV = 1.602×10⁻¹⁹ J)에서 자릿수 오류가 잦습니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Photoelectric Effect", "http://hyperphysics.phy-astr.gsu.edu/hbase/mod2.html", "강의노트"),
        ("위키백과(한국어) – 광전 효과", "https://ko.wikipedia.org/wiki/광전_효과", "백과사전"),
        ("Wikipedia – Photoelectric effect", "https://en.wikipedia.org/wiki/Photoelectric_effect", "백과사전"),
        ("Wikipedia – Annus Mirabilis papers (아인슈타인 1905 광양자 논문)", "https://en.wikipedia.org/wiki/Annus_mirabilis_papers", "백과사전"),
        ("R. A. Millikan, “A Direct Photoelectric Determination of Planck's h”, Phys. Rev. 7, 355 (1916)", "https://journals.aps.org/pr/abstract/10.1103/PhysRev.7.355", "논문"),
    ],
    "yt": ["광전효과 플랑크 상수 측정 실험", "photoelectric effect stopping potential Planck constant", "Einstein photoelectric equation explained"],
}

KB["millikan"] = {
    "name": "밀리컨 기름방울",
    "sections": [
        S("principle", "밀리컨 기름방울 실험 원리 — 전하 양자화",
          "중력·부력·스토크스 항력·전기력의 평형에서 방울 반지름과 전하를 구하고, 전하가 e 의 정수배로 뭉치는 것을 보인다.",
          ["원리", "유도", "이론", "스토크스", "stokes", "종단 속도", "부력", "항력", "양자화", "기본 전하", "왜", "설명"],
          r"""
반지름 $r$, 밀도 $\rho$ 인 기름방울이 공기(밀도 $\rho_a$, 점성 $\eta$) 속에서 **전기장 없이** 낙하하면 유효 중력 $m'g=\tfrac43\pi r^3(\rho-\rho_a)g$ 와 스토크스 항력 $6\pi\eta r v_f$ 가 평형을 이루어 종단 속도 $v_f$ 에 도달합니다.

$$r=\sqrt{\frac{9\eta v_f}{2(\rho-\rho_a)g}}$$

전압 $V$ 를 걸어 방울이 위로 속도 $v_r$ 로 상승하면 $qE=m'g+6\pi\eta rv_r$ ($E=V/d$) 이므로

$$q=\frac{6\pi\eta r\,(v_f+v_r)\,d}{V}$$

여러 방울, 여러 전하 상태에서 $q$ 를 구하면 값들이 **$e=1.602\times10^{-19}$ C 의 정수배**에 몰립니다. 이것이 전하 양자화의 직접 증거이며, 최대공약수 방식(또는 $q_i/n_i$ 히스토그램)으로 $e$ 를 결정합니다. 작은 방울($r\lesssim1\ \mu$m)은 공기 분자 평균 자유 행로와 비교되므로 **커닝엄 보정** $\eta_{\text{eff}}=\eta/(1+b/(pr))$ ($b\approx8.2\times10^{-3}$ Pa·m)을 적용해야 합니다. 밀리컨(1909–1913)은 이 실험으로 1923년 노벨상을 받았습니다.
"""),
        S("errors", "오차 원인 — 브라운 운동, 점성 온도 의존, 시간 측정, 방울 선택",
          "작은 방울은 브라운 운동으로 속도 판독이 흔들리고, 점성은 온도에 민감하며, 느린 방울을 선택해야 스토크스 법칙이 잘 성립한다.",
          ["오차", "브라운", "점성", "온도", "시간", "선택", "증발", "대류", "error", "차이"],
          r"""
- **브라운 운동**: 작은 방울은 열운동으로 궤적이 흔들려 속도 판독 분산이 큽니다. 여러 번 왕복 측정해 평균합니다.
- **공기 점성의 온도 의존**: $\eta$ 는 1 °C 당 약 0.3 % 변합니다. 실내 온도와 램프 발열을 기록하고 보정합니다. $q\propto\eta^{3/2}$ 이므로 점성 오차는 1.5배로 증폭됩니다.
- **극판 간격 $d$ 와 전압**: $q\propto d/V$. 극판 간격은 마이크로미터로 직접 측정합니다.
- **낙하 거리·시간**: 눈금 배율 교정과 스톱워치 반응 시간(~0.2 s)이 오차원입니다. 낙하 시간이 10 s 이상 되도록 느린 방울을 고르되, 너무 느리면 브라운 운동이 커집니다.
- **증발·대류**: 방울 질량이 시간에 따라 줄어들고, 챔버 내 온도 차가 대류를 만듭니다.
- **밀리컨의 데이터 선택 논란**: 밀리컨이 일부 데이터를 제외한 사실은 과학 윤리 교육의 사례로 자주 언급됩니다. 보고서에는 제외 기준을 사전에 정하고 명시해야 합니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Millikan Oil Drop Experiment", "http://hyperphysics.phy-astr.gsu.edu/hbase/electric/Millikan.html", "강의노트"),
        ("Wikipedia – Oil drop experiment", "https://en.wikipedia.org/wiki/Oil_drop_experiment", "백과사전"),
        ("R. A. Millikan, “On the Elementary Electrical Charge and the Avogadro Constant”, Phys. Rev. 2, 109 (1913)", "https://journals.aps.org/pr/abstract/10.1103/PhysRev.2.109", "논문"),
        ("Wikipedia – Stokes' law", "https://en.wikipedia.org/wiki/Stokes%27_law", "백과사전"),
    ],
    "yt": ["밀리컨 기름방울 실험 원리", "Millikan oil drop experiment explained", "oil drop experiment charge quantization"],
}

KB["pendulum"] = {
    "name": "단진자",
    "sections": [
        S("principle", "단진자 운동 방정식과 주기 공식 유도, 유한 진폭 보정",
          "θ̈+(g/L)sinθ=0 에서 작은 각 근사로 T=2π√(L/g); T² 대 L 그래프의 기울기에서 g=4π²/기울기 를 얻는다.",
          ["원리", "유도", "주기", "공식", "작은 각", "진폭", "보정", "중력가속도", "이론", "왜", "설명", "물리 진자"],
          r"""
길이 $L$ 인 줄에 매달린 질점에 대한 회전 운동 방정식은 $mL^2\ddot\theta=-mgL\sin\theta$, 즉

$$\ddot\theta+\frac{g}{L}\sin\theta=0.$$

**작은 각 근사** $\sin\theta\approx\theta$ 에서 단순 조화 운동이 되어 $T_0=2\pi\sqrt{L/g}$ 입니다. 진폭이 유한하면 정확한 주기는 완전 타원 적분으로 주어지고 급수 전개하면

$$T=T_0\left(1+\frac{\theta_0^2}{16}+\frac{11\theta_0^4}{3072}+\cdots\right)$$

$\theta_0=10^\circ$ 에서 보정은 +0.19 %, $20^\circ$ 에서 +0.77 % 입니다. 정밀 측정에서는 진폭을 기록하고 보정해야 합니다.

**g 결정법**: 여러 $L$ 에서 $T$ 를 재고 $T^2=\dfrac{4\pi^2}{g}L$ 그래프를 그리면 기울기에서 $g=4\pi^2/\text{기울기}$ 를 얻습니다. 절편이 0 이 아니면 길이 측정에 상수 오차(추 중심까지의 거리 누락 등)가 있다는 신호입니다. 구형 추(반지름 $r$)는 물리 진자로서 $L_{\text{eff}}=L+\dfrac{2r^2}{5L}$ 보정이 필요합니다. 표준 중력 $g_0=9.80665$ m/s², 서울 부근 실제값 ≈ 9.799 m/s².
"""),
        S("errors", "오차 원인 — 길이 기준점, 진폭, 공기 저항, 시간 측정",
          "길이는 회전축에서 추의 질량중심까지 재야 하고, 시간은 10주기 이상을 재어 반응 시간 오차를 1/10 로 줄인다.",
          ["오차", "길이", "질량중심", "공기", "저항", "시간", "스톱워치", "반응", "줄 질량", "error", "차이"],
          r"""
- **길이 기준점**: 지지점에서 **추의 질량중심**까지가 $L$ 입니다. 줄 끝까지만 재면 $L$ 이 과소평가되어 $g$ 가 작게 나옵니다.
- **시간 측정**: 사람 반응 시간은 ~0.2 s. 1주기를 재면 10 % 이상 오차지만 **20주기**를 재면 0.5 % 이하로 줄어듭니다. 최하점 통과 순간을 기준으로 재는 것이 정점보다 정확합니다(속도가 최대이므로 위치 판별이 명확).
- **진폭**: 위 보정식 참고. 진폭이 시간에 따라 줄어들므로 초기·말기 진폭의 평균을 씁니다.
- **공기 저항·줄 질량·지지점 마찰**: 주기를 미세하게 늘리고 진폭을 감쇠시킵니다. 무겁고 작은 추, 가벼운 줄이 유리합니다.
- **원뿔 진자화**: 옆으로 흔들리면 주기가 달라집니다. 한 평면에서 놓아야 합니다.
- $g\propto L/T^2$ 이므로 상대 불확도는 $\sqrt{(\delta L/L)^2+(2\delta T/T)^2}$ — **시간 오차가 두 배로 전파**됩니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Simple Pendulum", "http://hyperphysics.phy-astr.gsu.edu/hbase/pend.html", "강의노트"),
        ("Wikipedia – Pendulum (mechanics) (유한 진폭 급수 포함)", "https://en.wikipedia.org/wiki/Pendulum_(mechanics)", "백과사전"),
        ("Wikipedia – Gravity of Earth (위치별 g 값)", "https://en.wikipedia.org/wiki/Gravity_of_Earth", "백과사전"),
    ],
    "yt": ["단진자 중력가속도 측정 실험 오차", "simple pendulum period derivation large amplitude", "pendulum experiment measure g"],
}

KB["freefall"] = {
    "name": "자유낙하",
    "sections": [
        S("principle", "자유낙하 운동학과 g 측정",
          "정지 출발 시 h=½gt², v=gt, v²=2gh; h–t² 그래프의 기울기 ½g 에서 g 를 얻는다.",
          ["원리", "유도", "공식", "운동학", "등가속도", "이론", "왜", "설명", "중력가속도"],
          r"""
공기 저항을 무시하면 낙하체는 등가속도 $g$ 로 운동합니다. 정지 상태에서 놓으면

$$h=\tfrac12gt^2,\qquad v=gt,\qquad v^2=2gh.$$

여러 높이에서 낙하 시간을 재고 $h$–$t^2$ 그래프를 그리면 기울기가 $g/2$ 입니다. 초기 속도가 0 이 아니면 $h=v_0t+\tfrac12gt^2$ 이므로 $h/t$ 대 $t$ 그래프(기울기 $g/2$, 절편 $v_0$)를 쓰면 초기 속도 효과를 분리할 수 있습니다. 갈릴레이(1600년경)의 경사면 실험이 "낙하 거리 ∝ 시간²" 을 처음 보였고, 질량과 무관한 낙하는 등가 원리의 출발점입니다.

공기 저항이 있으면 $m\dot v=mg-\tfrac12C_d\rho Av^2$ 이 되어 종단 속도 $v_t=\sqrt{2mg/(C_d\rho A)}$ 로 수렴합니다. 강구(직경 2 cm)의 경우 $v_t\sim 70$ m/s 로 실험실 낙하(1–2 m, $v\lesssim6$ m/s)에서는 저항 효과가 0.1 % 미만이지만, 탁구공은 수 % 에 이릅니다.
"""),
        S("errors", "오차 원인 — 반응 시간, 전자석 잔류 자기, 높이 기준",
          "수동 스톱워치 반응 시간(~0.2 s)이 지배적이므로 전자 타이머·포토게이트가 필수이며, 전자석 해제 지연은 t 를 과대평가시켜 g 를 작게 만든다.",
          ["오차", "반응 시간", "타이머", "포토게이트", "전자석", "지연", "높이", "공기", "error", "차이"],
          r"""
- **반응 시간**: 사람 손 측정은 0.2 s 수준으로 1 m 낙하(0.45 s)에서 수십 % 오차. 전자석 해제와 연동된 타이머·포토게이트 사용이 필수입니다.
- **전자석 잔류 자기·해제 지연**: 전류를 끊어도 수 ms 뒤에 구가 떨어지면 $t$ 가 과대평가되어 **$g$ 가 작게** 나옵니다. $h/t$–$t$ 그래프의 절편(음의 $v_0$)이나 시간 오프셋 $t_0$ 를 피팅 파라미터로 넣어 보정합니다.
- **높이 기준**: 구의 **밑면**(포토게이트 차단 시점)과 정지 위치의 정합. 구 직경만큼의 상수 오차가 흔합니다.
- **연직 정렬·측면 충돌**: 낙하 경로가 기울면 유효 높이가 늘어납니다.
- $g=2h/t^2$ 이므로 시간 오차가 두 배로 전파됩니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Trajectories / Free Fall", "http://hyperphysics.phy-astr.gsu.edu/hbase/traj.html", "강의노트"),
        ("Wikipedia – Free fall", "https://en.wikipedia.org/wiki/Free_fall", "백과사전"),
        ("Wikipedia – Equations for a falling body", "https://en.wikipedia.org/wiki/Equations_for_a_falling_body", "백과사전"),
    ],
    "yt": ["자유낙하 실험 중력가속도 측정", "free fall experiment photogate measure g", "Galileo inclined plane experiment"],
}

KB["slit"] = {
    "name": "간섭·회절",
    "sections": [
        S("principle", "영의 이중슬릿 간섭과 단일슬릿 회절 — 조건식 유도",
          "경로차 d sinθ = mλ 가 밝은 무늬 조건, 작은 각에서 y_m = mλL/d; 단일슬릿 폭 a 의 회절 포락선이 세기를 변조한다.",
          ["원리", "유도", "간섭", "회절", "경로차", "무늬", "슬릿", "파동성", "이론", "왜", "설명", "포락선", "세기"],
          r"""
간격 $d$ 인 두 슬릿에서 나온 파동이 각도 $\theta$ 방향에서 만날 때 경로차는 $d\sin\theta$ 입니다. 보강 간섭(밝은 무늬) 조건은

$$d\sin\theta_m=m\lambda\quad(m=0,\pm1,\pm2,\dots),\qquad y_m\approx\frac{m\lambda L}{d}\ (\theta\ll1)$$

이며 스크린 거리 $L$ 에서 무늬 간격은 $\Delta y=\lambda L/d$ 입니다. 각 슬릿의 폭 $a$ 가 유한하므로 세기는 단일슬릿 회절 포락선으로 변조됩니다:

$$I(\theta)=I_0\cos^2\!\left(\frac{\pi d\sin\theta}{\lambda}\right)\,\mathrm{sinc}^2\!\left(\frac{\pi a\sin\theta}{\lambda}\right)$$

단일슬릿의 어두운 무늬 조건은 $a\sin\theta=m\lambda$ 이고, $d/a$ 가 정수일 때 그 차수의 간섭 무늬가 **사라지는 차수(missing order)** 가 생깁니다. 영(1801)의 실험은 빛의 파동성을, 광자·전자 하나씩 보내는 현대판 실험은 파동–입자 이중성을 보여 줍니다. 레이저(632.8 nm He-Ne)는 공간·시간 결맞음이 좋아 무늬가 선명합니다.
"""),
        S("errors", "오차 원인 — 거리 측정, 무늬 중심 판독, 작은 각 근사",
          "L 이 크면 무늬는 넓어져 판독은 쉬워지지만 밝기가 줄고, 여러 차수 간격을 한 번에 재어 나누면 판독 오차가 1/N 로 줄어든다.",
          ["오차", "거리", "판독", "중심", "작은 각", "근사", "차수", "error", "차이"],
          r"""
- **무늬 중심 판독**: 개별 무늬 대신 $N$ 개 무늬에 걸친 거리를 재고 $N$ 으로 나누면 판독 오차가 $1/N$ 로 줄어듭니다. 사진 촬영 후 세기 프로파일에서 피크를 찾는 방법이 정밀합니다.
- **스크린 거리 $L$**: 슬릿 위치(마운트가 아닌 슬릿 판)에서 스크린까지. $\lambda\propto1/L$ 로 전파.
- **작은 각 근사 붕괴**: $\theta>5^\circ$ 이면 $\tan\theta$ 와 $\sin\theta$ 차이가 0.4 % 이상. 정확히는 $\sin\theta_m=y_m/\sqrt{y_m^2+L^2}$ 을 써야 합니다.
- **슬릿 간격 공차·정렬**: 제조 공차(±5 %)와 슬릿–빔 수직 정렬 불량은 계통 오차가 됩니다.
- **포락선 영향**: 회절 포락선 최소 근처 무늬는 위치가 약간 치우칠 수 있어 중앙 부근 차수를 쓰는 것이 안전합니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Double Slit Interference / Fraunhofer Diffraction", "http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/slits.html", "강의노트"),
        ("Wikipedia – Double-slit experiment", "https://en.wikipedia.org/wiki/Double-slit_experiment", "백과사전"),
        ("Wikipedia – Diffraction", "https://en.wikipedia.org/wiki/Diffraction", "백과사전"),
        ("Feynman Lectures Vol. I Ch. 30 – Diffraction", "https://www.feynmanlectures.caltech.edu/I_30.html", "교재"),
    ],
    "yt": ["이중슬릿 간섭 실험 파장 측정", "Young double slit experiment derivation", "single slit diffraction intensity envelope"],
}

KB["bragg"] = {
    "name": "브래그 회절",
    "sections": [
        S("principle", "브래그 법칙 유도와 X선 파장·결정 격자 측정",
          "인접 원자면에서 반사된 X선의 경로차 2d sinθ 가 nλ 일 때 보강 간섭; NaCl (200)면 d=0.282 nm 로 Mo/Cu K선 파장을 결정한다.",
          ["원리", "유도", "브래그", "bragg", "격자", "결정", "면 간격", "x선", "파장", "이론", "왜", "설명"],
          r"""
결정 내 간격 $d$ 인 평행한 원자면에 X선이 각 $\theta$(면과 이루는 각, 입사각의 여각 아님에 주의)로 입사하면, 인접 면에서 반사된 두 파의 경로차는 $2d\sin\theta$ 입니다. 보강 간섭 조건이 **브래그 법칙**입니다.

$$2d\sin\theta_n=n\lambda\quad(n=1,2,\dots)$$

$n=1,2$ 피크 각도에서 $\lambda$ 를 구하거나, 알려진 $\lambda$(Mo $K_\alpha$ 0.0711 nm, Cu $K_\alpha$ 0.1542 nm)로 $d$ 를 구합니다. NaCl 의 (200) 면 간격 $d=a/2=0.2820$ nm. 회절계는 검출기를 $2\theta$ 로 움직이므로 판독값을 절반으로 나눠야 하는 경우가 많습니다. 브래그 부자(1913)는 이 법칙으로 결정 구조 해석을 열어 1915년 노벨상을 받았고, 이후 DNA 구조(1953)까지 X선 결정학의 토대가 되었습니다. $K_\beta$ 선(Mo 0.0632 nm)이 각 차수마다 $K_\alpha$ 앞쪽에 작은 피크로 나타나므로 혼동하지 않아야 합니다.
"""),
        S("errors", "오차 원인 — 영점 각도, 결정 정렬, 차수 중첩",
          "각도 영점 오프셋은 sinθ 에 비례해 계통적으로 전파되므로 여러 차수의 sinθ 대 n 회귀에서 절편으로 검출·보정한다.",
          ["오차", "영점", "정렬", "각도", "차수", "중첩", "error", "차이"],
          r"""
- **각도 영점 오프셋 $\delta$**: 실제 각은 $\theta+\delta$. $\sin\theta_n$ 대 $n$ 을 회귀하면 기울기 $\lambda/2d$, 절편으로 오프셋을 검출·보정할 수 있습니다.
- **결정 정렬**: 결정면이 회전축과 평행하지 않으면 유효 $d$ 가 커집니다. 저차 강한 피크로 정렬 후 측정합니다.
- **각도 분해능**: 콜리메이터 슬릿 폭이 피크 폭을 결정. 단계 크기를 피크 폭의 1/5 이하로.
- **$K_\alpha/K_\beta$ 및 고차 중첩**: 차수·선 종류를 잘못 배정하면 파장이 정수비로 틀립니다. 두 피크의 세기비(~5:1)로 판별합니다.
- **계수 통계**: 고차 피크는 약하므로 계수 시간을 늘려 $\sqrt N$ 오차를 줄입니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Bragg's Law", "http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/bragg.html", "강의노트"),
        ("Wikipedia – Bragg's law", "https://en.wikipedia.org/wiki/Bragg%27s_law", "백과사전"),
        ("Wikipedia – X-ray crystallography", "https://en.wikipedia.org/wiki/X-ray_crystallography", "백과사전"),
    ],
    "yt": ["브래그 회절 법칙 유도", "Bragg's law X-ray diffraction explained", "X-ray diffraction NaCl crystal experiment"],
}

KB["michelson"] = {
    "name": "마이컬슨 간섭계",
    "sections": [
        S("principle", "마이컬슨 간섭계 원리 — 거울 이동과 무늬 계수로 파장 측정",
          "거울을 Δd 이동하면 광경로차가 2Δd 변하므로 지나간 무늬 수 N 으로 λ = 2Δd/N 을 얻는다.",
          ["원리", "유도", "간섭계", "무늬", "경로차", "파장", "결맞음", "이론", "왜", "설명", "마이컬슨 몰리", "ligo"],
          r"""
빔스플리터가 빛을 두 팔로 나눈 뒤 다시 합치면, 두 팔의 광경로차 $\Delta=2(d_1-d_2)$ 에 따라 중심 무늬가 밝음($\Delta=m\lambda$)과 어두움을 반복합니다. 거울 하나를 $\Delta d$ 만큼 이동하면 경로차는 $2\Delta d$ 변하므로 지나간 무늬 수 $N$ 으로

$$\lambda=\frac{2\,\Delta d}{N}$$

를 얻습니다. 반대로 알려진 $\lambda$ 로 $\Delta d$ 를 nm 수준으로 측정할 수 있고, 팔 하나에 셀을 넣어 기체 굴절률($n-1=N\lambda/2\ell$), 유리판 두께 등도 측정합니다. 원형 무늬는 등경사 간섭(거울이 평행), 직선 무늬는 등두께 간섭(거울이 기울어짐)에 해당합니다. 마이컬슨–몰리(1887)는 이 장치로 에테르 바람을 찾지 못해 특수상대론의 실험적 토대를 놓았고, LIGO(2015)는 팔 길이 4 km 마이컬슨 간섭계로 중력파를 검출했습니다. 무늬가 보이려면 경로차가 광원의 **결맞음 길이** 이내여야 합니다(He-Ne 레이저 수십 cm, 나트륨 램프 수 mm).
"""),
        S("errors", "오차 원인 — 레버 비율 교정, 백래시, 계수 오류, 진동",
          "마이크로미터 눈금은 레버 비율(1:5 등)을 거쳐 거울에 전달되므로 비율 교정이 계통 오차를 지배하고, 한 방향으로만 돌려 백래시를 피한다.",
          ["오차", "레버", "백래시", "마이크로미터", "계수", "진동", "온도", "error", "차이"],
          r"""
- **레버 비율 교정**: 마이크로미터 1 눈금이 거울 이동으로 얼마인지(제조사 명시값 ±수 %)가 결과에 직접 비례합니다. 알려진 파장으로 역교정하면 계통 오차를 줄일 수 있습니다.
- **백래시**: 마이크로미터 회전 방향을 바꾸면 수 μm 공백이 생깁니다. 항상 한 방향으로 돌리고, 시작 전에 충분히 돌려 접촉시킵니다.
- **무늬 계수 오류**: 20–50개씩 여러 세트로 재고 평균. 카메라·포토다이오드로 자동 계수하면 정밀도가 크게 향상됩니다.
- **진동·공기 흐름·온도**: 무늬가 흔들리고 경로차가 드리프트. 광학 테이블, 덮개, 측정 시간 단축.
- $\lambda\propto\Delta d/N$ 이므로 $N$ 을 늘릴수록 계수 1개 오차의 상대 기여가 $1/N$ 로 줄어듭니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Michelson Interferometer", "http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/michel.html", "강의노트"),
        ("Wikipedia – Michelson interferometer", "https://en.wikipedia.org/wiki/Michelson_interferometer", "백과사전"),
        ("Wikipedia – Michelson–Morley experiment", "https://en.wikipedia.org/wiki/Michelson%E2%80%93Morley_experiment", "백과사전"),
        ("LIGO Caltech – 간섭계로 중력파를 검출하는 원리", "https://www.ligo.caltech.edu/page/what-is-interferometer", "기관"),
    ],
    "yt": ["마이컬슨 간섭계 실험 파장 측정", "Michelson interferometer explained fringe counting", "LIGO interferometer how it works"],
}

KB["ohm"] = {
    "name": "옴의 법칙",
    "sections": [
        S("principle", "옴의 법칙 — 거시적 V=IR 과 미시적 J=σE, 온도 의존성",
          "V=IR 은 드루드 모형의 J=σE 를 도체 형상에 적분한 결과이며, 금속은 R=R0(1+αΔT) 로 온도 상승 시 저항이 커진다.",
          ["원리", "유도", "옴", "ohm", "저항", "전도도", "드루드", "온도", "비옴", "이론", "왜", "설명"],
          r"""
드루드 모형에서 전자는 전기장 $E$ 에 의해 가속되다가 평균 시간 $\tau$ 마다 충돌하여 표류 속도 $v_d=-e\tau E/m$ 을 갖고, 전류 밀도는 $J=-nev_d=\dfrac{ne^2\tau}{m}E\equiv\sigma E$ 입니다. 길이 $\ell$, 단면 $A$ 인 균일 도체에 적분하면

$$V=IR,\qquad R=\frac{\ell}{\sigma A}=\rho\frac{\ell}{A}.$$

$I$–$V$ 그래프가 원점을 지나는 직선이면 "옴성(ohmic)" 이며 기울기의 역수가 $R$ 입니다. 금속은 온도가 오르면 격자 진동으로 $\tau$ 가 줄어 $R=R_0(1+\alpha\Delta T)$ ($\alpha_{Cu}\approx3.9\times10^{-3}$/K) 로 커지고, 반도체·열전구 필라멘트·다이오드는 비선형(비옴성)입니다. 전구 필라멘트의 $I$–$V$ 곡선이 위로 볼록하게 꺾이는 것은 자기 발열로 저항이 커지기 때문입니다.
"""),
        S("errors", "오차 원인 — 계기 부하 효과, 리드 저항, 자기 발열",
          "전압계 내부저항이 유한하면 병렬 분류로 전류계가 과대 전류를 읽고, 전류계 내부저항은 직렬로 더해져 전압을 나눠 갖는다.",
          ["오차", "내부저항", "부하", "리드", "발열", "접촉", "error", "차이"],
          r"""
- **전압계 부하 효과**: 내부저항 $R_V$ 가 $R$ 과 병렬로 들어가 전류계가 $I_R+I_V$ 를 읽습니다. $R\ll R_V$ 일 때만 무시 가능. 반대로 전류계 내부저항 $R_A$ 는 직렬로 더해져 전압계 연결 위치에 따라 $V_R+IR_A$ 를 읽게 됩니다. 두 연결법의 차이를 이용해 계기 저항을 추정할 수 있습니다.
- **리드·접촉 저항**: 수십 mΩ. 작은 저항 측정 시 4단자법(켈빈 접속) 필요.
- **자기 발열**: $P=I^2R$ 로 온도가 올라 $R$ 이 변합니다. 짧은 시간·작은 전류 사용 후 냉각.
- **계기 정확도·분해능**: 디지털 멀티미터는 "% of reading + digits" 사양을 불확도로 반영합니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Ohm's Law", "http://hyperphysics.phy-astr.gsu.edu/hbase/electric/ohmlaw.html", "강의노트"),
        ("Wikipedia – Ohm's law", "https://en.wikipedia.org/wiki/Ohm%27s_law", "백과사전"),
        ("Wikipedia – Drude model", "https://en.wikipedia.org/wiki/Drude_model", "백과사전"),
    ],
    "yt": ["옴의 법칙 실험 오차 원인", "Ohm's law experiment voltmeter loading effect", "Drude model conductivity"],
}

KB["rc"] = {
    "name": "RC 회로",
    "sections": [
        S("principle", "RC 충·방전 미분방정식과 시간 상수",
          "KVL 에서 R dq/dt + q/C = V0 를 풀면 충전 V0(1−e^{−t/τ}), 방전 V0 e^{−t/τ}, τ=RC; ln(V/V0) 대 t 의 기울기 −1/τ 로 τ 를 구한다.",
          ["원리", "유도", "시간 상수", "충전", "방전", "지수", "미분방정식", "반감기", "이론", "왜", "설명"],
          r"""
저항 $R$ 과 축전기 $C$ 가 전원 $V_0$ 에 직렬 연결되면 키르히호프 전압 법칙은 $R\dfrac{dq}{dt}+\dfrac{q}{C}=V_0$ 입니다. 초기 조건 $q(0)=0$ 으로 풀면

$$V_C(t)=V_0\left(1-e^{-t/\tau}\right)\ \text{(충전)},\qquad V_C(t)=V_0\,e^{-t/\tau}\ \text{(방전)},\qquad \tau=RC.$$

$t=\tau$ 에서 방전 전압은 $V_0/e\approx0.368V_0$, 반감기는 $t_{1/2}=\tau\ln2$ 입니다. 데이터는 $\ln(V/V_0)=-t/\tau$ 로 **선형화**하여 기울기에서 $\tau$ 를 구하는 것이 정확하며, 전압을 여러 자릿수에 걸쳐 측정할 수 있어 지수 법칙 검증에도 좋습니다. 방전 전류 $I=-(V_0/R)e^{-t/\tau}$ 도 같은 시간 상수를 갖습니다. 회로에 흐르는 에너지 중 절반은 저항에서 열로 사라지고 절반만 축전기에 저장됩니다($\tfrac12CV_0^2$) — 이는 $R$ 값과 무관한 흥미로운 결과입니다.
"""),
        S("errors", "오차 원인 — 계측기 입력 임피던스, 부품 공차, 누설",
          "오실로스코프·멀티미터 입력저항(1–10 MΩ)이 C 와 병렬로 들어가 유효 R 이 줄어 τ 가 작아지며, 전해 축전기 공차는 ±20 % 에 이른다.",
          ["오차", "임피던스", "입력저항", "공차", "누설", "전해", "error", "차이"],
          r"""
- **계측기 입력 임피던스**: 전압계(10 MΩ)나 오실로스코프(1 MΩ)가 축전기와 병렬이면 방전 경로가 추가되어 $R_{\text{eff}}=R\parallel R_{in}$, $\tau$ 가 **작아집니다**. $R$ 이 100 kΩ 이상이면 무시할 수 없습니다.
- **부품 공차**: 전해 축전기 ±20 %, 세라믹 ±10 %. 측정된 $\tau$ 와 표시값 $RC$ 의 차이는 대부분 여기서 옵니다. $R$ 을 별도 측정하면 $C$ 의 실제값을 역산할 수 있습니다.
- **누설 전류·유전 흡수**: 전해 축전기는 자체 방전이 있어 긴 $\tau$ 측정에서 곡선이 지수보다 빨리 떨어집니다.
- **시간 기준·샘플링**: 스위치 접점 바운스와 샘플링 지연은 $t$ 오프셋을 만듭니다. $\ln V$ 회귀의 절편이 $\ln V_0$ 와 다르면 오프셋 신호입니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Capacitor Discharge", "http://hyperphysics.phy-astr.gsu.edu/hbase/electric/capdis.html", "강의노트"),
        ("Wikipedia – RC circuit", "https://en.wikipedia.org/wiki/RC_circuit", "백과사전"),
        ("Wikipedia – RC time constant", "https://en.wikipedia.org/wiki/RC_time_constant", "백과사전"),
    ],
    "yt": ["RC 회로 충방전 시간상수 실험", "RC circuit charging discharging derivation", "RC time constant oscilloscope measurement"],
}

KB["hooke"] = {
    "name": "훅의 법칙",
    "sections": [
        S("principle", "훅의 법칙과 탄성 한계, 단순 조화 운동과의 연결",
          "F=−kx 는 퍼텐셜 최소 근처의 2차 근사이며, 탄성 한계 내에서만 성립한다. 진동 주기 T=2π√(m/k) 로도 k 를 구할 수 있다.",
          ["원리", "유도", "훅", "hooke", "용수철", "탄성", "복원력", "퍼텐셜", "진동", "주기", "이론", "왜", "설명"],
          r"""
용수철에 힘 $F$ 를 가하면 늘어난 길이 $x$ 는 힘에 비례합니다: $F=kx$ (복원력은 $-kx$). 이는 임의의 퍼텐셜 $U(x)$ 를 안정 평형점 근처에서 테일러 전개할 때 1차 항이 0 이고 2차 항 $\tfrac12U''(0)x^2$ 만 남는다는 일반적 사실의 표현으로, 원자 간 결합·분자 진동·고체 탄성까지 같은 논리가 적용됩니다. 저장 에너지는 $U=\tfrac12kx^2$.

**정적 측정**: 질량 $m$ 을 달아 $mg$ 대 $x$ 그래프의 기울기에서 $k$. 절편이 0 이 아니면 초기 장력(코일이 밀착된 용수철)이나 영점 오차를 뜻합니다.
**동적 측정**: $T=2\pi\sqrt{(m+m_s/3)/k}$ — 용수철 자체 질량 $m_s$ 의 1/3 이 유효 질량으로 더해집니다. $T^2$ 대 $m$ 의 기울기 $4\pi^2/k$ 에서 $k$ 를, 절편에서 $m_s/3$ 을 얻습니다. 두 방법의 $k$ 가 일치하는지 비교하면 좋은 검증이 됩니다.
직렬 연결 $1/k=1/k_1+1/k_2$, 병렬 $k=k_1+k_2$. **탄성 한계**를 넘으면 영구 변형이 생기고 선형성이 깨지므로 하중 범위를 제한해야 합니다.
"""),
        S("errors", "오차 원인 — 초기 장력, 영점, 시차, 탄성 한계 초과",
          "초기 장력이 있는 용수철은 작은 하중에서 늘어나지 않아 그래프가 원점을 지나지 않으며, 기울기만 사용하면 이 영향을 피할 수 있다.",
          ["오차", "초기 장력", "영점", "시차", "탄성 한계", "히스테리시스", "error", "차이"],
          r"""
- **초기 장력**: 밀착 코일 용수철은 일정 하중까지 늘어나지 않습니다. $F$–$x$ 그래프의 절편이 이를 나타내며 기울기(=$k$)는 영향받지 않습니다.
- **영점·시차 판독**: 눈금과 지시침 사이 거리로 시차가 생깁니다. 눈높이를 맞추고 거울 눈금을 씁니다.
- **탄성 한계 초과·히스테리시스**: 하중을 올릴 때와 내릴 때 $x$ 가 다르면 영구 변형 신호입니다. 왕복 측정으로 확인합니다.
- **추 질량 공차·용수철 자중**: 추 표시값(±1 %)을 저울로 재확인하고, 수직으로 걸린 용수철의 자중 처짐은 절편에 포함됩니다.
"""),
    ],
    "refs": [
        ("HyperPhysics – Hooke's Law", "http://hyperphysics.phy-astr.gsu.edu/hbase/permot2.html", "강의노트"),
        ("Wikipedia – Hooke's law", "https://en.wikipedia.org/wiki/Hooke%27s_law", "백과사전"),
        ("Wikipedia – Effective mass (spring–mass system)", "https://en.wikipedia.org/wiki/Effective_mass_(spring%E2%80%93mass_system)", "백과사전"),
    ],
    "yt": ["훅의 법칙 용수철 상수 측정 실험", "Hooke's law experiment spring constant", "spring mass oscillation effective mass"],
}

# 범용 방법론 (모든 실험 공통)
KB["_general"] = {
    "name": "실험 방법론 공통",
    "sections": [
        S("uncertainty", "측정 불확도 — A형/B형 평가와 불확도 전파",
          "표준불확도는 반복 측정(A형, s/√n)과 사양·분해능(B형)으로 평가하고, 결과 불확도는 편미분 제곱합으로 전파하며 k=2 확장불확도로 보고한다.",
          ["불확도", "uncertainty", "표준편차", "신뢰", "전파", "propagation", "오차 막대", "유효", "k=2", "확장"],
          r"""
**용어**: '오차'는 참값과의 차이(알 수 없음), '불확도'는 측정값 주변에 참값이 있을 것으로 기대되는 구간의 폭(평가 가능)입니다. GUM(측정 불확도 표현 지침)이 국제 표준입니다.

- **A형(통계적) 평가**: $n$ 회 반복 측정의 표준편차 $s$ 에서 평균의 표준불확도 $u=s/\sqrt n$.
- **B형 평가**: 계기 사양, 분해능, 교정 성적서 등에서. 분해능 $\delta$ 인 디지털 계기는 균등분포 가정으로 $u=\delta/\sqrt{12}\approx0.29\delta$; 제조사 "±a" 사양은 $u=a/\sqrt3$.
- **합성 표준불확도(전파식)**: $f(x_1,\dots,x_n)$ 에 대해
  $$u_c(f)^2=\sum_i\left(\frac{\partial f}{\partial x_i}\right)^2u(x_i)^2\quad(\text{입력량이 독립일 때})$$
  곱·나눗셈·거듭제곱 형태 $f=x^ay^b$ 이면 상대 불확도로 $\left(\dfrac{u_f}{f}\right)^2=a^2\left(\dfrac{u_x}{x}\right)^2+b^2\left(\dfrac{u_y}{y}\right)^2$ — 지수가 클수록 그 변수의 기여가 증폭됩니다. 이 시스템의 '감도 분석' 탭 탄성도가 정확히 이 계수 $a, b$ 에 해당합니다.
- **확장 불확도**: $U=k\,u_c$, 보통 $k=2$ (약 95 % 신뢰수준). 보고 형식: "$g=(9.79\pm0.04)\ \mathrm{m/s^2}$ ($k=2$)".
- **참값과의 비교**: $|x-x_{ref}|\le U$ 이면 "불확도 내에서 일치". $|x-x_{ref}|/u_c$ (z-값)이 2–3 을 넘으면 계통 오차를 찾아야 합니다. 오차율(%)만 보고하고 불확도를 빼먹으면 "일치 여부"를 판단할 수 없습니다.
"""),
        S("regression", "최소제곱 선형 회귀 — 기울기·절편과 그 불확도, 선형화 전략",
          "y=mx+b 의 최소제곱 해와 σ_m, σ_b 공식을 쓰고, 비선형 법칙은 로그·역수 변환으로 선형화하여 기울기에서 물리량을 얻는다.",
          ["회귀", "최소제곱", "기울기", "절편", "그래프", "직선", "선형화", "regression", "fit", "피팅", "r값", "상관"],
          r"""
데이터 $(x_i,y_i)$, $i=1..n$ 에 직선 $y=mx+b$ 를 맞추면(잔차 제곱합 최소화) $\Delta=n\sum x_i^2-(\sum x_i)^2$ 일 때

$$m=\frac{n\sum x_iy_i-\sum x_i\sum y_i}{\Delta},\qquad b=\frac{\sum x_i^2\sum y_i-\sum x_i\sum x_iy_i}{\Delta}$$

$$\sigma_y^2=\frac{1}{n-2}\sum(y_i-mx_i-b)^2,\qquad \sigma_m=\sigma_y\sqrt{\frac{n}{\Delta}},\qquad \sigma_b=\sigma_y\sqrt{\frac{\sum x_i^2}{\Delta}}$$

- **왜 회귀가 개별 계산 평균보다 좋은가**: 절편이 상수 계통 오차(영점, 접촉 전위, 길이 오프셋)를 흡수해 기울기가 깨끗해지고, 불확도를 데이터 산포에서 직접 추정할 수 있습니다.
- **선형화**: 지수 법칙 $y=Ae^{-t/\tau}$ → $\ln y=\ln A-t/\tau$; 거듭제곱 $y=ax^n$ → $\log y=\log a+n\log x$; 컴프턴 $1/E'$ 대 $(1-\cos\theta)$; 진자 $T^2$ 대 $L$. 변환 후 오차 분포가 바뀌므로 엄밀히는 가중 최소제곱($w_i=1/\sigma_i^2$)을 써야 합니다.
- **$r$ 의 한계**: 상관계수가 0.999 라도 계통 편향(기울기≠1, 절편≠0)은 드러나지 않습니다. 반드시 **잔차 그래프**를 그려 무작위성을 확인하세요 — 잔차에 곡률·추세가 보이면 모형이 틀렸다는 뜻입니다.
- **절편을 0으로 강제할지**: 물리적으로 원점을 지나야 하더라도 먼저 자유 절편으로 피팅해 절편이 0 과 통계적으로 일치하는지 확인하는 것이 계통 오차 진단에 유리합니다.
"""),
        S("systematic", "계통 오차 vs 우연 오차 — 구별법과 제거 전략",
          "우연 오차는 반복·평균으로 줄지만 계통 오차는 줄지 않으므로 교정·영위법·반전·배경 차감 같은 설계로 제거해야 한다.",
          ["계통", "체계", "우연", "무작위", "systematic", "random", "편향", "bias", "정확도", "정밀도", "교정", "calibration", "영점", "배경"],
          r"""
- **우연(무작위) 오차**: 반복할 때마다 부호가 바뀌며 평균하면 $1/\sqrt n$ 로 줄어듭니다. 판독 흔들림, 계수 통계, 전자 잡음.
- **계통 오차**: 같은 방향으로 일관되게 치우치며 반복 측정으로 줄지 **않습니다**. 영점 이탈, 이득 오차, 단위 환산 실수, 모형 가정 위배(공기 저항 무시 등), 관찰자 편향.
- **정밀도(precision)** 는 우연 오차의 크기, **정확도(accuracy)** 는 계통 오차까지 포함한 참값 근접도입니다. 정밀하지만 부정확한 데이터(작은 산포, 큰 편향)가 가장 위험합니다.

**진단 신호**: (1) 잔차 부호가 모두 같음, (2) 측정 vs 이론 회귀 기울기 ≠ 1 → 비례 오차(교정 계수·단위), 절편 ≠ 0 → 영점 오차, (3) 잔차가 독립 변수에 따라 추세를 가짐 → 모형 누락 항.

**제거 전략**: **교정**(알려진 표준으로 다점 교정), **영위법**(브리지·보상법으로 계기 눈금 의존 제거), **반전·대칭 측정**(전류 반전, 좌우 측정으로 지자기·오프셋 상쇄), **배경 차감**(동일 조건 blank 측정), **차분 측정**(절대값 대신 차이를 재어 공통 오프셋 제거), **모형 보정**(진폭 보정, 부력 보정 등 알려진 효과 계산 보정).
"""),
        S("statistics", "계수 통계·포아송 분포·χ² 적합도",
          "방사선 계수의 표준편차는 √N 이므로 1 % 정밀도에 10⁴ 계수가 필요하고, 배경 차감 시 불확도는 √(N+B) 로 커진다. 축소 χ² ≈ 1 이면 모형과 오차 추정이 정합적이다.",
          ["포아송", "poisson", "계수", "통계", "카이", "chi", "히스토그램", "정규분포", "가우스", "분포", "표본", "n수"],
          r"""
- **포아송 통계**: 일정 시간 동안 독립 사건이 평균 $\mu$ 회 일어나면 $P(N)=\mu^Ne^{-\mu}/N!$, 분산 $=\mu$. 계수 $N$ 의 표준편차는 $\sqrt N$, 상대 불확도 $1/\sqrt N$. 1 % 정밀도 → $N=10^4$, 0.1 % → $10^6$. $\mu\gtrsim20$ 이면 정규분포로 근사.
- **배경 차감**: 신호 $S=N_{tot}-N_{bg}$ 의 불확도는 $\sqrt{N_{tot}+N_{bg}}$ — 배경이 클수록 불확도가 커지므로 차폐가 중요하고, 배경 측정 시간을 충분히 길게 해야 합니다.
- **계수율**: $R=N/t$, $u_R=\sqrt N/t$. 사각 시간(dead time) $\tau_d$ 보정: $R_{true}=R/(1-R\tau_d)$.
- **정규분포와 표준오차**: 반복 측정 히스토그램이 가우스형이면 68 % 가 $\pm1\sigma$, 95 % 가 $\pm2\sigma$ 안에. 평균의 표준오차는 $\sigma/\sqrt n$.
- **χ² 적합도**: $\chi^2=\sum_i\dfrac{(y_i-f(x_i))^2}{\sigma_i^2}$, 자유도 $\nu=n-p$ ($p$: 피팅 파라미터 수). **축소 χ²** $=\chi^2/\nu\approx1$ 이면 모형·오차 추정이 정합적, $\gg1$ 이면 모형 부적합 또는 오차 과소평가, $\ll1$ 이면 오차 과대평가.
- **이상치 처리**: 사전에 기준(예: 3σ, Chauvenet)을 정하고 제외 사실과 이유를 보고서에 명시합니다.
"""),
        S("calibration", "계기 교정과 채널–에너지(또는 눈금–물리량) 변환",
          "두 점 교정은 비선형성을 잡지 못하므로 3점 이상으로 다항 교정하고, 측정 전후 교정 피크를 재확인해 드리프트를 감시한다.",
          ["교정", "calibration", "채널", "mca", "눈금", "변환", "드리프트", "이득", "gain", "표준 시료", "기준 선원", "선형성"],
          r"""
- **교정 함수**: 계기 판독 $c$(채널, 눈금)와 물리량 $E$ 사이 $E=a+bc(+dc^2)$. 두 점 교정은 $d=0$ 을 강제하므로 중간 구간에서 비선형 편향이 남습니다. 3점 이상으로 최소제곱 교정하고 잔차를 확인합니다.
- **교정점 선택**: 측정 범위를 **포괄**하도록(내삽이 외삽보다 안전). 감마: Cs-137 662, Na-22 511/1275, Co-60 1173/1332, Ba-133 356 keV. 광학: He-Ne 632.8 nm, Hg 546.1/435.8 nm.
- **드리프트**: 온도·고전압·전원 변동으로 이득이 시간에 따라 흘러갑니다. 측정 전후로 교정 피크를 재확인하고, 차이가 크면 시간 가중 보정 또는 재측정.
- **추적성(traceability)**: 교정 표준의 불확도가 결과 불확도 예산의 B형 항목으로 들어갑니다.
- **비례/영점 오차의 분리**: 이 시스템의 측정 vs 이론 회귀에서 기울기 $\ne1$ 은 이득(비례) 오차, 절편 $\ne0$ 은 영점 오차로 읽습니다.
"""),
        S("reporting", "유효숫자·결과 표기·보고서 작성 체크리스트",
          "불확도는 유효숫자 1–2자리로 반올림하고 측정값은 그 자릿수에 맞추며, 보고서는 목적–이론–방법–결과–분석–결론의 논리 흐름과 그래프·표 규약을 지켜야 한다.",
          ["유효숫자", "표기", "반올림", "보고서", "리포트", "결론", "작성", "그래프 그리", "표 작성", "단위", "report"],
          r"""
- **유효숫자 규칙**: 불확도를 먼저 1–2 자리 유효숫자로 반올림하고, 측정값을 **같은 자릿수**까지 맞춥니다. 예: $9.7832\pm0.0417 \to 9.78\pm0.04$. 계산 중간에는 자릿수를 넉넉히 유지하고 최종 결과에서만 반올림합니다.
- **단위와 접두어**: SI 단위, 지수 표기($1.60\times10^{-19}$ C)와 접두어(keV, nm)를 일관되게. 표 머리글에 단위를 쓰고 셀에는 숫자만.
- **그래프 규약**: 축 이름+단위, 오차 막대, 피팅선과 식·파라미터·$r$ 표기, 독립 변수를 x축에. 선형화 그래프는 변환 변수를 축 이름에 명시(예: "$1/E'$ (keV⁻¹)").
- **보고서 흐름**: 목적 → 이론(핵심 식 유도, 가정 명시) → 장치·방법(재현 가능하게) → 데이터(원자료 표) → 분석(계산·그래프·불확도) → 결과와 참값 비교(불확도 내 일치 여부) → 오차 원인(계통/우연 구분, 정량적 크기 추정) → 개선 방안 → 결론.
- **좋은 '오차 분석'의 기준**: "인간 오차" 같은 모호한 표현 대신, 각 원인이 **어느 방향으로 얼마나** 결과를 바꿨을지 추정하고 관측된 편향과 부합하는지 논증합니다.
"""),
        S("approach", "낯선 질문에 접근하는 방법 — 차원 분석·극한 검토·크기 추정",
          "공식을 이해하려면 차원 일치, 극한(θ→0, m→∞ 등)에서의 거동, 자릿수 추정을 먼저 확인하는 것이 가장 빠른 검증법이다.",
          ["방법", "접근", "검토", "차원", "극한", "추정", "확인", "이해", "어떻게"],
          r"""
어떤 물리 공식이나 실험 결과를 이해·검증할 때 다음 세 가지 점검은 언제나 유효합니다.

1. **차원(단위) 분석**: 양변의 단위가 일치하는지. 예: $e/m=2V/(B^2r^2)$ → $\mathrm{V/(T^2m^2)}=\mathrm{C/kg}$ 확인.
2. **극한 검토**: 매개변수를 극단으로 보냈을 때 물리적으로 타당한지. 컴프턴 공식에서 $\theta\to0$ 이면 $E'\to E_0$, $m\to\infty$ 이면 이동 없음(속박 전자); 진자 $L\to0$ 이면 $T\to0$; RC 에서 $t\gg\tau$ 이면 $V\to0$.
3. **자릿수 추정**: 대입 전에 대략 값을 예상해 계산 실수를 잡습니다(예: 662 keV 후방산란 ≈ 180 keV 근방).

그리고 실험 데이터 해석에서는 항상 **(a) 측정 vs 이론 그래프, (b) 잔차 그래프, (c) 불확도 대비 편차** 세 가지를 그려 본 뒤 결론을 내립니다.
"""),
    ],
    "refs": [
        ("BIPM – JCGM 100:2008 GUM (측정 불확도 표현 지침, 무료 PDF)", "https://www.bipm.org/en/committees/jc/jcgm/publications", "표준문서"),
        ("NIST Technical Note 1297 – Guidelines for Evaluating and Expressing Uncertainty", "https://www.nist.gov/pml/nist-technical-note-1297", "표준문서"),
        ("NIST/SEMATECH e-Handbook of Statistical Methods", "https://www.itl.nist.gov/div898/handbook/", "핸드북"),
        ("Wikipedia – Propagation of uncertainty", "https://en.wikipedia.org/wiki/Propagation_of_uncertainty", "백과사전"),
        ("Wikipedia – Simple linear regression (기울기·절편 불확도 공식)", "https://en.wikipedia.org/wiki/Simple_linear_regression", "백과사전"),
        ("Wikipedia – Poisson distribution", "https://en.wikipedia.org/wiki/Poisson_distribution", "백과사전"),
        ("Wikipedia – Reduced chi-squared statistic", "https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic", "백과사전"),
        ("J. R. Taylor, *An Introduction to Error Analysis*, 2nd ed. (University Science Books, 1997) — 학부 실험 오차론 표준 교재", "https://en.wikipedia.org/wiki/John_R._Taylor", "교재"),
    ],
    "yt": ["측정 불확도 전파 계산 방법", "uncertainty propagation partial derivatives explained", "least squares linear regression uncertainty slope intercept", "systematic vs random error physics lab"],
}


# ---------------------------------------------------------------------------
# 7. AI 멘토 엔진
# ---------------------------------------------------------------------------

def _q(url_base: str, query: str) -> str:
    return url_base + urllib.parse.quote_plus(query)


def _yt(query: str) -> str:
    return _q("https://www.youtube.com/results?search_query=", query)


def detect_topic(prompt: str, title: str, manual_text: str) -> str | None:
    """질문(가중 3) > 제목(가중 2) > 매뉴얼(가중 1) 순으로 실험 주제를 판별한다."""
    best, best_score = None, 0
    p, t, m = prompt.lower(), title.lower(), manual_text[:4000].lower()
    for tmpl in KNOWN_TEMPLATES:
        score = sum(3 for k in tmpl["keys"] if k in p) + sum(2 for k in tmpl["keys"] if k in t) + sum(1 for k in tmpl["keys"] if k in m)
        if score > best_score:
            best, best_score = tmpl["key"], score
    return best


def _score_section(prompt_l: str, sec: dict[str, Any]) -> float:
    return float(sum(len(k) for k in sec["keys"] if k in prompt_l))


def select_sections(prompt: str, topic: str | None) -> list[tuple[str, dict[str, Any], float]]:
    """(주제키, 섹션, 점수) 상위 3개. 실험 주제 섹션은 1.3배 가중."""
    p = prompt.lower()
    scored: list[tuple[str, dict[str, Any], float]] = []
    if topic and topic in KB:
        for sec in KB[topic]["sections"]:
            scored.append((topic, sec, _score_section(p, sec) * 1.3))
    for sec in KB["_general"]["sections"]:
        scored.append(("_general", sec, _score_section(p, sec)))
    # 다른 실험 주제가 질문에 직접 언급된 경우
    for key, kb in KB.items():
        if key in (topic, "_general"):
            continue
        if any(k in p for k in next((t["keys"] for t in KNOWN_TEMPLATES if t["key"] == key), [])):
            for sec in kb["sections"]:
                scored.append((key, sec, _score_section(p, sec) * 1.2 + 2))
    scored = [s for s in scored if s[2] > 0]
    scored.sort(key=lambda s: -s[2])
    picked, seen = [], set()
    for key, sec, sc in scored:
        if (key, sec["id"]) in seen:
            continue
        seen.add((key, sec["id"]))
        picked.append((key, sec, sc))
        if len(picked) == 3:
            break
    if not picked and topic and topic in KB:
        picked.append((topic, KB[topic]["sections"][0], 0.0))
    return picked


def _stats_md(ctx: dict[str, Any]) -> str:
    stats = ctx.get("stats")
    if not stats or stats.get("n", 0) == 0:
        return ("아직 분석을 실행하지 않아 수치 비교는 생략합니다. 위의 **정밀 분석 실행** 버튼을 누르면 "
                "평균 오차율·회귀 기울기·감도 분석 결과를 근거로 더 구체적으로 답할 수 있습니다.")
    lines = [
        f"- 유효 데이터 {stats['n']}개, 평균 오차율 **{stats['mean_err']:.2f} %**, 최대 오차 **{stats['max_err']:.2f} %** (행 #{stats['max_err_idx'] + 1}), RMSE {stats['rmse']:.4g}",
        f"- 평균 잔차(이론−측정) {stats['bias']:+.4g}" + (" — **모든 잔차 부호가 동일** → 계통 편향 존재" if stats.get("same_sign") else " — 잔차 부호 혼재 → 우연 오차가 상당 부분"),
    ]
    if not math.isnan(stats.get("slope", float("nan"))):
        lines.append(f"- 측정 = {stats['slope']:.4f}·이론 + {stats['intercept']:.4g}, r = {stats['r']:.4f}")
    lines += ["- 진단: " + d for d in _diagnose(stats)]
    sens = ctx.get("sens") or []
    if sens:
        lines.append("- 감도(탄성도) 상위: " + ", ".join(f"`{d['name']}` {d['elasticity']:+.2f}" for d in sens[:4]) +
                     f" → **`{sens[0]['name']}`** 의 측정 정밀도가 최종 불확도를 지배")
    return "\n".join(lines)


def _hook_compton(ctx: dict[str, Any]) -> str:
    """사용자 데이터로 1/E' 대 (1−cosθ) 선형화 회귀를 즉석 수행하여 mc², E0 를 추정한다."""
    consts = ctx.get("constants_dict") or {}
    out = []
    E0, mc2 = consts.get("E0"), consts.get("mc2")
    if E0 and mc2:
        e_back = E0 / (1 + 2 * E0 / mc2)
        out.append(f"- 현재 상수(E0={E0:.4g} keV, mc²={mc2:.4g} keV)에서 예측: 후방산란(180°) 에너지 **{e_back:.1f} keV**, "
                   f"컴프턴 모서리 **{E0 - e_back:.1f} keV**, 90° 산란 에너지 **{E0 / (1 + E0 / mc2):.1f} keV**. "
                   f"컴프턴 파장 λ_C = h/mc = 2.426 pm.")
    df = ctx.get("df")
    if isinstance(df, pd.DataFrame):
        ang_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["각", "theta", "angle"])), None)
        meas_col = _find_meas_col(df)
        if ang_col and meas_col:
            d = df[[ang_col, meas_col]].apply(pd.to_numeric, errors="coerce").dropna()
            d = d[(d[meas_col] > 0)]
            if len(d) >= 2:
                x = 1 - np.cos(np.radians(d[ang_col].to_numpy(float)))
                y = 1.0 / d[meas_col].to_numpy(float)
                if np.std(x) > 0:
                    slope, intercept = np.polyfit(x, y, 1)
                    mc2_est = 1 / slope if slope != 0 else float("nan")
                    e0_est = 1 / intercept if intercept != 0 else float("nan")
                    r = float(np.corrcoef(x, y)[0, 1]) if len(d) >= 3 else float("nan")
                    dev = abs(mc2_est - 511) / 511 * 100
                    out.append(
                        f"- **당신의 표 데이터로 즉석 선형화 회귀** ($1/E'$ 대 $1-\\cos\\theta$, {len(d)}점): "
                        f"기울기 → **mc² 추정 = {mc2_est:.1f} keV** (참값 511 keV, 편차 {dev:.1f} %), "
                        f"절편 → E0 추정 = {e0_est:.1f} keV" + (f", r = {r:.4f}" if not math.isnan(r) else "") + ". "
                        + ("직선성과 mc² 일치가 양호하여 **광자 운동량 p=E/c 와 2체 충돌 보존 법칙이 데이터로 지지됩니다**."
                           if dev < 10 else
                           "mc² 편차가 커서 교정(이득/영점) 또는 각도 정의를 점검해야 합니다. 절편(E0 추정)이 661.7 keV 와 다르면 채널–에너지 교정 문제일 가능성이 큽니다.")
                    )
    return "\n".join(out)


def _hook_pendulum(ctx: dict[str, Any]) -> str:
    df = ctx.get("df")
    if not isinstance(df, pd.DataFrame):
        return ""
    len_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["길이", "length", "l ["])), None)
    meas_col = _find_meas_col(df)
    if not (len_col and meas_col):
        return ""
    d = df[[len_col, meas_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 2 or np.std(d[len_col]) == 0:
        return ""
    slope, intercept = np.polyfit(d[len_col].to_numpy(float), d[meas_col].to_numpy(float) ** 2, 1)
    if slope <= 0:
        return ""
    g = 4 * math.pi ** 2 / slope
    return (f"- **당신의 표 데이터로 즉석 회귀** ($T^2$ 대 $L$, {len(d)}점): 기울기 → **g 추정 = {g:.3f} m/s²** "
            f"(표준값 9.807, 편차 {abs(g - 9.80665) / 9.80665 * 100:.2f} %), 절편 {intercept:+.4g} s²"
            + (" — 절편이 0 에서 크게 벗어나면 길이 기준점(질량중심) 오차를 의심하세요." if abs(intercept) > 0.05 else "."))


TOPIC_HOOKS = {"compton": _hook_compton, "pendulum": _hook_pendulum}


def _refs_md(prompt: str, topic: str | None, used_keys: list[str], title: str) -> str:
    refs: list[tuple[str, str, str]] = []
    yts: list[str] = []
    for key in dict.fromkeys(([topic] if topic else []) + used_keys + ["_general"]):
        if key in KB:
            refs += KB[key]["refs"]
            yts += KB[key]["yt"]
    seen, uniq = set(), []
    for r in refs:
        if r[1] not in seen:
            seen.add(r[1])
            uniq.append(r)
    md = ["**📚 참고 자료 (읽을 것)**"]
    md += [f"- [{t}]({u}) — {k}" for t, u, k in uniq[:10]]
    md.append("\n**🎬 함께 보면 좋은 영상 (YouTube 검색 링크)**")
    q_short = re.sub(r"\s+", " ", prompt.strip())[:60]
    yt_queries = list(dict.fromkeys(yts[:4] + [f"{title} {q_short}"]))
    md += [f"- [▶ {q}]({_yt(q)})" for q in yt_queries]
    md.append("\n**🔎 더 찾아보기**")
    md.append(f"- [Google Scholar: {title} {q_short}]({_q('https://scholar.google.com/scholar?q=', title + ' ' + q_short)})")
    md.append(f"- [위키백과 검색: {q_short}]({_q('https://ko.wikipedia.org/w/index.php?search=', q_short)})")
    md.append("\n<span class='small-note'>영상은 특정 링크가 삭제·변경될 수 있어 검색 결과 링크로 제공합니다. 문서 링크는 위키백과·HyperPhysics·원논문 DOI·표준 문서 등 안정적인 출처만 사용했습니다.</span>")
    return "\n".join(md)


def builtin_answer(prompt: str, ctx: dict[str, Any]) -> str:
    topic = ctx.get("topic")
    title = ctx["title"]
    picked = select_sections(prompt, topic)
    used_keys = [k for k, _, _ in picked]

    parts = [f"### 🎯 질문: {prompt.strip()}"]
    if picked:
        parts.append("**한 줄 핵심**\n" + "\n".join(f"- {sec['tl']}" for _, sec, _ in picked))
        parts.append("---\n### 📘 원리 설명")
        for key, sec, _ in picked:
            label = KB[key]["name"]
            parts.append(f"#### {sec['title']}  <span class='small-note'>[{label}]</span>\n\n{sec['body']}")
    else:
        parts.append(
            "내장 지식 엔진에서 이 질문과 직접 대응하는 원리 섹션을 찾지 못했습니다. 아래에 현재 실험 데이터 기반 진단과 참고 자료를 제공합니다. "
            "**임의의 질문에 대한 실시간 답변**이 필요하면 사이드바 'AI 멘토 엔진' 에서 외부 LLM API 를 연결하세요."
        )

    parts.append("---\n### 🧪 현재 실험 데이터와의 연결")
    parts.append(f"- 실험: **{title}**, 수식: `{ctx['formula'] or '(미설정)'}`, 상수: {ctx.get('const_str') or '없음'}")
    parts.append(_stats_md(ctx))
    hook = TOPIC_HOOKS.get(topic or "")
    if hook:
        h = hook(ctx)
        if h:
            parts.append(h)

    parts.append("---\n### ✅ 보고서에 반영할 때 체크리스트")
    parts.append(
        "- 핵심 공식의 **유도 가정**(자유·정지 전자, 작은 각, 무마찰 등)을 명시하고, 데이터가 그 가정 범위 안에 있는지 논증했는가\n"
        "- 그래프는 **선형화**하여 기울기·절편의 물리적 의미와 불확도를 함께 제시했는가\n"
        "- 오차 원인을 계통/우연으로 구분하고 **방향과 크기**를 정량 추정했는가 (위 진단 결과와 부합하는지)\n"
        "- 참값과의 비교는 오차율(%)뿐 아니라 **확장 불확도(k=2) 내 일치 여부**로 판단했는가"
    )
    parts.append("---\n" + _refs_md(prompt, topic, used_keys, title))
    follow = {
        "compton": ["1/E' 대 (1−cosθ) 그래프의 기울기에서 전자 질량을 구하는 방법을 자세히 설명해줘", "다중 산란이 피크를 고에너지로 이동시키는 이유는?", "클라인–니시나 공식과 톰슨 산란의 차이는?"],
        "em": ["지구 자기장을 상쇄하는 실험적 방법은?", "e/m 이 참값보다 크게 나온 이유는?", "V 대 I² 그래프로 e/m 을 구하는 방법"],
        "franck": ["첫 번째 극소가 4.9 V 가 아닌 이유는?", "온도가 곡선 모양에 미치는 영향은?", "봉우리 간격이 높은 전압에서 커지는 이유"],
        "photo": ["역전류가 정지 전압 판정에 미치는 영향은?", "빛의 세기를 바꿔도 정지 전압이 변하지 않는 이유", "V_s–ν 그래프의 절편으로 일함수를 구하는 방법"],
    }.get(topic or "", ["이 실험의 불확도를 어떻게 정량적으로 평가하지?", "회귀 기울기가 1에서 벗어난 이유는?", "계통 오차와 우연 오차를 어떻게 구분하지?"])
    parts.append("---\n**❓ 이어서 물어볼 만한 질문**\n" + "\n".join(f"- {f}" for f in follow))
    return "\n\n".join(parts)


# ---- 외부 LLM 연동 ----------------------------------------------------------

LLM_PROVIDERS = {
    "내장 지식 엔진 (오프라인)": None,
    "OpenAI 호환 API (GPT 등)": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "Anthropic Claude": {"model": "claude-sonnet-4-20250514", "base_url": "https://api.anthropic.com"},
    "Google Gemini": {"model": "gemini-2.5-flash", "base_url": "https://generativelanguage.googleapis.com"},
}


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # noqa: BLE001
        return ""


def _http_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 180) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def call_llm(cfg: dict[str, Any], system: str, messages: list[dict[str, str]]) -> str:
    provider, key, model, base = cfg["provider"], cfg["api_key"], cfg["model"], cfg["base_url"].rstrip("/")
    if provider.startswith("OpenAI"):
        data = _http_json(f"{base}/chat/completions",
                          {"model": model, "messages": [{"role": "system", "content": system}] + messages, "temperature": 0.4, "max_tokens": 4000},
                          {"Authorization": f"Bearer {key}"})
        return data["choices"][0]["message"]["content"]
    if provider.startswith("Anthropic"):
        data = _http_json(f"{base}/v1/messages",
                          {"model": model, "system": system, "messages": messages, "max_tokens": 4000, "temperature": 0.4},
                          {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(part.get("text", "") for part in data["content"])
    if provider.startswith("Google"):
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
        data = _http_json(f"{base}/v1beta/models/{model}:generateContent?key={urllib.parse.quote(key)}",
                          {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
                           "generationConfig": {"temperature": 0.4, "maxOutputTokens": 4000}}, {})
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    raise RuntimeError("알 수 없는 제공자")


def build_llm_system(ctx: dict[str, Any], prompt: str) -> str:
    topic = ctx.get("topic")
    picked = select_sections(prompt, topic)
    knowledge = "\n\n".join(f"[{KB[k]['name']} / {sec['title']}]\n{sec['body']}" for k, sec, _ in picked)
    df = ctx.get("df")
    table_csv = df.head(40).to_csv(index=False) if isinstance(df, pd.DataFrame) else "(없음)"
    stats = ctx.get("stats")
    stats_txt = _stats_md(ctx) if stats else "(분석 미실행)"
    hook = TOPIC_HOOKS.get(topic or "")
    hook_txt = hook(ctx) if hook else ""
    manual = (ctx.get("manual_text") or "")[:6000]
    return f"""당신은 대학 물리실험을 지도하는 교수급 멘토입니다. 학생의 질문에 한국어로, 학부 3~4학년~대학원 수준의 깊이로 답하되 친절하고 명확하게 설명합니다.

[현재 실험 컨텍스트]
- 실험 제목: {ctx['title']}
- 사용 수식(파이썬 표기): {ctx['formula'] or '(미설정)'}
- 적용 상수: {ctx.get('const_str') or '없음'}
- 데이터 표(CSV, 최대 40행):
{table_csv}
- 최근 분석 통계·진단:
{stats_txt}
{hook_txt}

[매뉴얼 본문 발췌]
{manual or '(업로드된 매뉴얼 없음)'}

[참고 지식 (검증된 내용, 자유롭게 활용)]
{knowledge or '(해당 없음)'}

[답변 규칙]
1. 구조: ### 핵심 답변 → ### 원리와 유도(수식은 LaTeX $...$/$$...$$) → ### 현재 데이터와의 연결(위 통계 수치를 반드시 인용) → ### 오차·개선 관점 → ### 보고서 작성 팁 → ### 이어서 생각해볼 질문.
2. 길이는 충분히 길고 상세하게(최소 700자 이상). 공식은 반드시 유도 과정과 가정을 함께 제시.
3. 사실이 불확실하면 그렇다고 명시. URL 은 절대 생성하지 마세요(참고 자료 링크는 시스템이 별도로 붙입니다).
4. 학생의 질문 표현을 그대로 반복하지 말고 바로 내용으로 들어가세요."""


def mentor_answer(prompt: str, ctx: dict[str, Any], llm_cfg: dict[str, Any] | None, history: list[dict[str, str]]) -> tuple[str, str]:
    """(답변 마크다운, 사용 엔진 이름)"""
    if llm_cfg and llm_cfg.get("api_key"):
        try:
            msgs = [{"role": m["role"], "content": m["content"][:3000]} for m in history[-8:] if m["role"] in ("user", "assistant")]
            msgs.append({"role": "user", "content": prompt})
            text = call_llm(llm_cfg, build_llm_system(ctx, prompt), msgs)
            picked = select_sections(prompt, ctx.get("topic"))
            text += "\n\n---\n" + _refs_md(prompt, ctx.get("topic"), [k for k, _, _ in picked], ctx["title"])
            return text, f"{llm_cfg['provider']} · {llm_cfg['model']}"
        except Exception as exc:  # noqa: BLE001
            fallback = builtin_answer(prompt, ctx)
            return (f"> ⚠️ 외부 LLM 호출 실패 ({exc}) — 내장 지식 엔진으로 답변합니다.\n\n" + fallback), "내장 지식 엔진 (폴백)"
    return builtin_answer(prompt, ctx), "내장 지식 엔진"


# ---------------------------------------------------------------------------
# 8. UI
# ---------------------------------------------------------------------------

def _apply_suggestion() -> None:
    sugg = st.session_state.get("suggestion")
    if not sugg:
        return
    st.session_state.input_df = pd.DataFrame(sugg["data"], columns=sugg["columns"]).astype(float)
    for k, v in sugg["constants"].items():
        st.session_state.custom_constants[k] = float(v)
        st.session_state[f"const_{k}"] = float(v)
    st.session_state["exp_title_input"] = sugg["title"]
    first = sugg["formulas"][0] if sugg["formulas"] else ""
    st.session_state["formula_radio"] = first
    st.session_state["formula_input"] = first
    st.session_state.table_version += 1


def _sync_formula_from_radio() -> None:
    st.session_state["formula_input"] = st.session_state.get("formula_radio", "")


def _render_sidebar() -> tuple[str, dict[str, Any] | None]:
    st.sidebar.header("⚙️ 장비 상수 및 계산 설정")
    angle_unit = st.sidebar.selectbox("삼각함수 각도 단위 (sin/cos/tan 입력값)", ["deg", "rad"], key="angle_unit",
                                      help="표의 각도 컬럼 단위와 일치시켜 주세요.")

    with st.sidebar.expander("➕ 새 상수 추가"):
        new_name = st.text_input("기호 (예: h, c, R, mu0)", key="new_const_name")
        new_val = st.text_input("값 (예: 6.626e-34)", key="new_const_val")
        if st.button("상수 추가", key="add_const_btn"):
            name = new_name.strip()
            if not re.fullmatch(r"[^\W\d]\w*", name):
                st.warning("기호는 문자로 시작하고 공백 없이 입력해야 합니다.")
            elif name in st.session_state.custom_constants:
                st.warning("이미 존재하는 기호입니다.")
            elif math.isnan(_to_float(new_val.strip())):
                st.warning("값이 올바른 숫자가 아닙니다.")
            else:
                st.session_state.custom_constants[name] = float(new_val)
                st.session_state[f"const_{name}"] = float(new_val)
                st.rerun()

    with st.sidebar.expander("➖ 상수 삭제"):
        if st.session_state.custom_constants:
            del_name = st.selectbox("삭제할 상수", list(st.session_state.custom_constants), key="del_const_sel")
            if st.button("삭제", key="del_const_btn"):
                st.session_state.custom_constants.pop(del_name, None)
                st.session_state.pop(f"const_{del_name}", None)
                st.rerun()
        else:
            st.caption("등록된 상수가 없습니다.")

    st.sidebar.divider()
    st.sidebar.markdown("**활성 상수 (직접 수정 가능)**")
    if not st.session_state.custom_constants:
        st.sidebar.caption("상수가 없습니다. 매뉴얼을 업로드하거나 위에서 추가하세요.")
    for k in list(st.session_state.custom_constants):
        wkey = f"const_{k}"
        if wkey not in st.session_state:
            st.session_state[wkey] = float(st.session_state.custom_constants[k])
        st.session_state.custom_constants[k] = float(st.sidebar.number_input(k, key=wkey, format="%.6g", step=None))

    st.sidebar.divider()
    st.sidebar.header("🤖 AI 멘토 엔진")
    provider = st.sidebar.selectbox("답변 엔진", list(LLM_PROVIDERS), key="llm_provider",
                                    help="내장 엔진은 오프라인 지식 베이스 기반입니다. 임의 질문에 대한 실시간 생성 답변은 외부 API 키가 필요합니다.")
    llm_cfg: dict[str, Any] | None = None
    if LLM_PROVIDERS[provider]:
        defaults = LLM_PROVIDERS[provider]
        api_key = st.sidebar.text_input("API Key", type="password", key="llm_api_key",
                                        value=st.session_state.get("llm_api_key", _secret("LLM_API_KEY")))
        model = st.sidebar.text_input("모델명", key=f"llm_model_{provider}", value=defaults["model"])
        base_url = defaults["base_url"]
        if provider.startswith("OpenAI"):
            base_url = st.sidebar.text_input("Base URL (OpenAI 호환 서버 사용 시 변경)", key="llm_base_url", value=defaults["base_url"])
        if not api_key:
            st.sidebar.warning("API 키가 없으면 내장 지식 엔진으로 답변합니다.")
        llm_cfg = {"provider": provider, "api_key": api_key.strip(), "model": model.strip(), "base_url": base_url}
        st.sidebar.caption("키는 이 세션 메모리에만 머물며 저장되지 않습니다. 새로고침 시 사라집니다.")

    st.sidebar.divider()
    c1, c2 = st.sidebar.columns(2)
    if c1.button("🧹 보드·대화 초기화", use_container_width=True):
        st.session_state.history = []
        st.session_state.chat_messages = []
        st.rerun()
    if c2.button("♻️ 전체 초기화", use_container_width=True, help="브라우저 새로고침과 동일하게 모든 입력·상수·표·대화를 지웁니다."):
        st.session_state.clear()
        st.rerun()
    return angle_unit, llm_cfg


def main() -> None:
    # ---- 1. 매뉴얼 업로드 및 자동 분석 ---------------------------------
    st.header("📄 1. 매뉴얼 업로드 & AI 자동 분석")
    uploaded_files = st.file_uploader(
        "실험 매뉴얼 PDF를 업로드하세요. 파일이 바뀌면(추가·교체·삭제) 실험 제목·상수·수식·표 형식이 즉시 다시 추천됩니다.",
        type=["pdf"], accept_multiple_files=True,
    )

    text_parts: list[str] = []
    sig_src = ""
    if uploaded_files:
        for f in uploaded_files:
            data = f.getvalue()
            sig_src += f"{f.name}:{len(data)};"
            try:
                text_parts.append(_extract_pdf_text(data))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"PDF 읽기 오류 ({f.name}): {exc}")
    manual_text = "\n".join(text_parts)
    sig = hashlib.md5((sig_src + manual_text).encode("utf-8", "ignore")).hexdigest()

    if sig != st.session_state.manual_sig:
        st.session_state.manual_sig = sig
        st.session_state.manual_text = manual_text
        with st.spinner("매뉴얼을 분석해 수식·상수·표 형식을 추천하는 중..."):
            st.session_state.suggestion = analyze_manual(manual_text)
        if st.session_state.get("auto_apply", True):
            _apply_suggestion()

    sugg: dict[str, Any] = st.session_state.suggestion
    if uploaded_files:
        st.success(f"매뉴얼 {len(uploaded_files)}개 파일 분석 완료 (텍스트 {len(manual_text):,}자).")
        with st.expander("📃 추출된 매뉴얼 텍스트 미리보기"):
            st.text(manual_text[:4000] + ("\n... (이하 생략)" if len(manual_text) > 4000 else ""))
            if not manual_text.strip():
                st.warning("텍스트가 추출되지 않았습니다. 스캔 이미지 PDF는 텍스트 레이어가 없어 분석할 수 없습니다.")

    angle_unit, llm_cfg = _render_sidebar()

    # ---- 2. 실험 설정 및 수식 선택 -------------------------------------
    st.divider()
    st.header("🛠️ 2. 실험 설정 및 수식 선택")
    st.markdown(f"<div class='ai-box'><b>🤖 AI 매뉴얼 분석 및 추천 설정</b><br>{sugg['desc']}</div>", unsafe_allow_html=True)
    if sugg.get("extracted"):
        with st.expander(f"🔎 매뉴얼 본문에서 추출한 수식 {len(sugg['extracted'])}개 보기"):
            st.dataframe(pd.DataFrame([{"좌변": f["lhs"], "우변(파이썬 표기)": f["rhs"], "변수": ", ".join(f["vars"])} for f in sugg["extracted"]]),
                         use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.checkbox("매뉴얼 변경 시 추천 설정 자동 적용", value=True, key="auto_apply")
    with c2:
        st.button("🔄 추천 설정 지금 다시 적용 (표·상수·수식 초기화)", on_click=_apply_suggestion, use_container_width=True)

    st.session_state.setdefault("exp_title_input", sugg["title"])
    exp_name = st.text_input("실험 제목 (수정 가능)", key="exp_title_input", placeholder="예: 컴프턴 산란 실험")

    options = [f for f in sugg["formulas"] if f] or [""]
    if st.session_state.get("formula_radio") not in options:
        st.session_state["formula_radio"] = options[0]
    st.markdown("**💡 AI 추천 수식 (선택하면 아래 입력란에 반영됩니다)**")
    if options != [""]:
        st.radio("매뉴얼에서 추출·추천된 수식:", options=options, key="formula_radio",
                 on_change=_sync_formula_from_radio, label_visibility="collapsed")
    else:
        st.caption("추천 수식이 없습니다. 아래에 직접 입력하세요.")
    st.session_state.setdefault("formula_input", options[0])
    raw_formula = st.text_input("이론 수식 (수정 가능) — 표 컬럼명(단위 제외)과 사이드바 상수 기호를 그대로 사용",
                                key="formula_input", placeholder="예: E0 / (1 + (E0/mc2) * (1 - cos(산란각)))")

    var_names = [_base_name(c) for c in st.session_state.input_df.columns if MEAS_KEY not in str(c)]
    symbols = formula_symbols(raw_formula, var_names, list(st.session_state.custom_constants))
    overlap = sorted(set(symbols["vars"]) & set(st.session_state.custom_constants))
    v_txt = ", ".join(f"`{v}`" for v in symbols["vars"]) or "없음"
    c_txt = ", ".join(f"`{c}`" for c in symbols["consts"]) or "없음"
    st.caption(f"✅ 표 변수: {v_txt}  |  ⚙️ 상수: {c_txt}  |  📐 각도 단위: {angle_unit}")
    if overlap:
        st.warning(f"표 컬럼과 상수 이름이 겹칩니다: {', '.join(overlap)} — 표의 값이 우선 적용됩니다.")
    if symbols["unknown"]:
        st.error(f"정의되지 않은 기호: {', '.join(symbols['unknown'])} — 표 컬럼 또는 상수로 추가해야 계산됩니다.")
        cc1, cc2 = st.columns(2)
        if cc1.button("➕ 누락 기호를 표 컬럼으로 자동 추가", use_container_width=True):
            df = st.session_state.input_df.copy()
            meas_col = _find_meas_col(df)
            norm = _normalize_text(st.session_state.manual_text)
            for name in symbols["unknown"]:
                unit = _guess_unit(name, norm)
                col = f"{name} [{unit}]" if unit else name
                if col not in df.columns:
                    df[col] = np.nan
            if meas_col:
                df = df[[c for c in df.columns if c != meas_col] + [meas_col]]
            st.session_state.input_df = df
            st.session_state.table_version += 1
            st.rerun()
        if cc2.button("⚙️ 누락 기호를 상수로 추가 (값 0)", use_container_width=True):
            for name in symbols["unknown"]:
                st.session_state.custom_constants.setdefault(name, 0.0)
            st.rerun()

    st.markdown("**📊 실험 데이터 입력 표** (행 추가/삭제 가능, 컬럼명의 `[단위]` 앞부분이 수식 변수명입니다)")
    edited_df = st.data_editor(st.session_state.input_df, num_rows="dynamic", use_container_width=True,
                               key=f"data_editor_{st.session_state.table_version}")

    with st.popover("⚙️ 표 컬럼 관리 (변수 추가/삭제 · 예시 데이터)"):
        st.write("**새 변수 컬럼 추가**")
        new_col_name = st.text_input("변수명 (예: 전압, 반지름, theta)", key="new_col_name")
        unit_choice = st.selectbox("단위", ["없음", "deg", "rad", "keV", "eV", "V", "A", "m", "cm", "mm", "nm", "μm", "s", "ms", "kg", "g", "C", "N", "T", "Hz", "K", "J"], key="new_col_unit")
        if st.button("컬럼 추가", key="add_col_btn"):
            name = new_col_name.strip()
            if not name:
                st.warning("변수명을 입력하세요.")
            else:
                df = edited_df.copy()
                col = f"{name} [{unit_choice}]" if unit_choice != "없음" else name
                if col in df.columns or name in [_base_name(c) for c in df.columns]:
                    st.warning("같은 이름의 컬럼이 이미 있습니다.")
                else:
                    meas_col = _find_meas_col(df)
                    df[col] = np.nan
                    if meas_col:
                        df = df[[c for c in df.columns if c != meas_col] + [meas_col]]
                    st.session_state.input_df = df
                    st.session_state.table_version += 1
                    st.rerun()

        st.divider()
        st.write("**컬럼 삭제**")
        if len(edited_df.columns) > 1:
            del_col = st.selectbox("삭제할 컬럼", list(edited_df.columns), key="del_col_sel")
            if st.button("컬럼 삭제", key="del_col_btn"):
                st.session_state.input_df = edited_df.drop(columns=[del_col])
                st.session_state.table_version += 1
                st.rerun()

        st.divider()
        b1, b2 = st.columns(2)
        if b1.button("🧹 표 데이터 비우기", key="clear_rows_btn", use_container_width=True):
            st.session_state.input_df = pd.DataFrame(_nan_rows(len(edited_df.columns)), columns=edited_df.columns).astype(float)
            st.session_state.table_version += 1
            st.rerun()
        if sugg.get("example") and b2.button("📝 예시 데이터 불러오기", key="load_example_btn", use_container_width=True,
                                             help="템플릿의 샘플 값으로 표를 채웁니다 (컬럼도 템플릿 형식으로 재설정)."):
            st.session_state.input_df = pd.DataFrame(sugg["example"], columns=sugg["columns"]).astype(float)
            st.session_state.table_version += 1
            st.rerun()

    # ---- 3. 실시간 교차 검증 ----------------------------------------------
    st.divider()
    st.header("🔍 3. 실시간 교차 검증 (이론 예측 vs 실험 측정)")
    st.markdown("<div class='small-note'>현재 수식·상수·표를 바탕으로 이론 예측값을 즉시 계산하여 측정값과 비교합니다. 표나 수식을 수정하면 자동으로 갱신됩니다.</div>", unsafe_allow_html=True)

    meas_col, eval_rows = evaluate_table(edited_df, raw_formula, st.session_state.custom_constants, angle_unit)
    if not meas_col:
        st.error(f"표에 '{MEAS_KEY}' 컬럼이 필요합니다. 컬럼 관리에서 '{MEAS_KEY}' 이름을 포함한 컬럼을 추가하세요.")
    elif not raw_formula.strip():
        st.info("이론 수식을 입력하면 교차 검증 표가 표시됩니다.")
    elif not eval_rows:
        st.info("표에 데이터를 입력하면 교차 검증 표가 표시됩니다.")
    else:
        preview = []
        for r in eval_rows:
            ok = r["theo"] is not None and not math.isnan(r["meas"])
            preview.append({
                "행": r["idx"],
                "이론 예측값": round(r["theo"], 4) if r["theo"] is not None else None,
                "실험 측정값": None if math.isnan(r["meas"]) else r["meas"],
                "차이 (이론−측정)": round(r["theo"] - r["meas"], 4) if ok else None,
                "오차율 (%)": round(abs(r["theo"] - r["meas"]) / abs(r["theo"]) * 100, 3) if ok and r["theo"] != 0 else None,
                "상태": "✅ 정상" if ok else ("⚠️ 측정값 없음" if r["theo"] is not None else f"❌ {r['err']}"),
            })
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    # ---- 4. 분석 실행 -------------------------------------------------------
    if st.button("🚀 정밀 분석 실행 & 리포트 보드에 추가", type="primary", use_container_width=True):
        if not meas_col:
            st.error(f"표에 '{MEAS_KEY}' 컬럼이 필요합니다.")
            return
        if not raw_formula.strip():
            st.error("이론 수식을 입력해 주세요.")
            return
        if not eval_rows:
            st.error("분석할 데이터 행이 없습니다.")
            return

        results, theo_list, meas_list, valid_vars, row_ids = [], [], [], [], []
        for r in eval_rows:
            record: dict[str, Any] = {"행": r["idx"]}
            for c in edited_df.columns:
                if c != meas_col:
                    record[c] = edited_df.iloc[r["idx"] - 1][c]
            record[MEAS_KEY] = None if math.isnan(r["meas"]) else r["meas"]
            if r["theo"] is None:
                record.update({"이론값 (계산)": None, "절대 오차": None, "오차율 (%)": None, "잔차 (이론−측정)": None, "상태": f"❌ {r['err']}"})
            elif math.isnan(r["meas"]):
                record.update({"이론값 (계산)": round(r["theo"], 6), "절대 오차": None, "오차율 (%)": None, "잔차 (이론−측정)": None, "상태": "⚠️ 측정값 없음"})
            else:
                resid = r["theo"] - r["meas"]
                record.update({"이론값 (계산)": round(r["theo"], 6), "절대 오차": round(abs(resid), 6),
                               "오차율 (%)": round(abs(resid) / abs(r["theo"]) * 100, 3) if r["theo"] != 0 else None,
                               "잔차 (이론−측정)": round(resid, 6), "상태": "✅"})
                theo_list.append(r["theo"])
                meas_list.append(r["meas"])
                valid_vars.append(r["vars"])
                row_ids.append(r["idx"])
            results.append(record)

        if not theo_list:
            st.error("이론값과 측정값이 모두 유효한 행이 하나도 없습니다. 위 교차 검증 표의 '상태'를 확인하세요.")
            return

        theo_arr, meas_arr = np.array(theo_list, dtype=float), np.array(meas_list, dtype=float)
        stats = compute_stats(theo_arr, meas_arr)
        used_consts = {k: v for k, v in st.session_state.custom_constants.items() if k in symbols["consts"]}
        const_str = ", ".join(f"{k}={v:.6g}" for k, v in used_consts.items())
        rep_vars: dict[str, float] = {}
        for name in valid_vars[0]:
            vals = [vv[name] for vv in valid_vars if not math.isnan(vv.get(name, float("nan")))]
            rep_vars[name] = float(np.mean(vals)) if vals else float("nan")
        sens = sensitivity_analysis(raw_formula, rep_vars, st.session_state.custom_constants, angle_unit)
        report = build_report(exp_name, raw_formula, const_str, stats, sens)

        st.session_state.history.append({
            "id": len(st.session_state.history) + 1, "title": exp_name, "formula": raw_formula, "constants": const_str,
            "constants_dict": dict(st.session_state.custom_constants), "angle_unit": angle_unit,
            "df": pd.DataFrame(results), "raw_df": edited_df.copy(), "report": report, "stats": stats, "sens": sens,
            "symbols": symbols, "theo_arr": theo_arr, "meas_arr": meas_arr, "row_ids": row_ids,
        })
        st.rerun()

    # ---- 5. 누적 분석 보드 --------------------------------------------------
    st.divider()
    st.header("📚 4. 누적 분석 보드 & 학술 진단")
    if not st.session_state.history:
        st.info("아직 분석 기록이 없습니다. 위의 분석 실행 버튼을 눌러 주세요.")
    else:
        for rec in reversed(st.session_state.history):
            with st.container():
                h1, h2 = st.columns([6, 1])
                h1.markdown(f"### 📊 분석 #{rec['id']} : {rec['title']}")
                if h2.button("🗑️ 삭제", key=f"del_rec_{rec['id']}"):
                    st.session_state.history = [r for r in st.session_state.history if r["id"] != rec["id"]]
                    st.rerun()
                st.markdown(
                    f"""<div class="metric-card"><b>사용 수식:</b> <code>{rec['formula']}</code><br>
                    <b>적용 상수:</b> {rec['constants'] or '없음'} &nbsp;|&nbsp; <b>각도 단위:</b> {rec['angle_unit']}
                    &nbsp;|&nbsp; <b>평균 오차율:</b> {rec['stats']['mean_err']:.3f}% &nbsp;|&nbsp; <b>판정:</b> {_grade(rec['stats']['mean_err'])}</div>""",
                    unsafe_allow_html=True)
                st.dataframe(rec["df"], use_container_width=True, hide_index=True)

                col_graph, col_report = st.columns([1, 1.25])
                with col_graph:
                    st.markdown("**📈 측정값 vs 이론값 상관 & 잔차**")
                    ko = KOREAN_FONT_OK
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5.6), gridspec_kw={"height_ratios": [3, 1.6]})
                    theo, meas = rec["theo_arr"], rec["meas_arr"]
                    ax1.scatter(theo, meas, color="#3182ce", s=40, zorder=3, label="측정 데이터" if ko else "Measured")
                    lo, hi = min(theo.min(), meas.min()), max(theo.max(), meas.max())
                    pad = (hi - lo) * 0.1 if hi > lo else abs(hi) * 0.1 + 1e-9
                    xs = np.array([lo - pad, hi + pad])
                    ax1.plot(xs, xs, "k--", alpha=0.5, label="이상적 일치 (y=x)" if ko else "Ideal (y=x)")
                    if not math.isnan(rec["stats"]["slope"]):
                        ax1.plot(xs, rec["stats"]["slope"] * xs + rec["stats"]["intercept"], color="#e53e3e", alpha=0.7,
                                 label=(f"회귀 (기울기 {rec['stats']['slope']:.3f})" if ko else f"Fit (slope {rec['stats']['slope']:.3f})"))
                    ax1.set_xlabel("이론값 (계산)" if ko else "Theoretical value")
                    ax1.set_ylabel("실험 측정값" if ko else "Measured value")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(fontsize=7)
                    ax2.bar([str(i) for i in rec["row_ids"]], theo - meas, color=np.where(theo - meas >= 0, "#3182ce", "#e53e3e"))
                    ax2.axhline(0, color="k", lw=0.8)
                    ax2.set_xlabel("행 번호" if ko else "Row")
                    ax2.set_ylabel("잔차 (이론−측정)" if ko else "Residual")
                    ax2.grid(True, axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    if not ko:
                        st.caption("한글 폰트가 없어 그래프 라벨은 영문으로 표시됩니다.")
                with col_report:
                    st.markdown("**🧠 학술 진단 리포트**")
                    tab1, tab2, tab3 = st.tabs(["📈 핵심 요약·진단", "🎯 감도·불확도 분석", "🔬 오차 원인·개선 방안"])
                    with tab1:
                        st.markdown(rec["report"]["summary"])
                    with tab2:
                        st.markdown(rec["report"]["sensitivity"])
                    with tab3:
                        st.markdown(rec["report"]["improve"])
            st.write("---")

    # ---- 6. AI 멘토 Q&A ---------------------------------------------------------
    st.header("💬 5. AI 실험 멘토 (원리 설명 · 데이터 연결 · 참고 자료)")
    engine_label = (f"{llm_cfg['provider']} · {llm_cfg['model']}" if llm_cfg and llm_cfg.get("api_key") else "내장 지식 엔진 (오프라인)")
    st.markdown(
        f"<div class='small-note'>현재 엔진: <b>{engine_label}</b>. 원리·유도·역사적 의의·오차 물리·통계 방법론을 실험 데이터와 연결해 상세히 답하고, "
        "읽을 자료와 영상 검색 링크를 함께 제공합니다. 외부 LLM 을 연결하면 어떤 질문이든 실시간으로 생성된 답을 받을 수 있습니다.</div>",
        unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("engine"):
                st.caption(f"엔진: {msg['engine']}")

    if prompt := st.chat_input("자유롭게 질문하세요! (예: 이 실험이 왜 빛의 입자성을 증명하나요? 고각도에서 오차가 큰 이유는?)"):
        last = st.session_state.history[-1] if st.session_state.history else None
        title_ctx = last["title"] if last else exp_name
        ctx = {
            "title": title_ctx,
            "formula": last["formula"] if last else raw_formula,
            "stats": last["stats"] if last else None,
            "sens": last["sens"] if last else None,
            "symbols": last["symbols"] if last else symbols,
            "const_str": last["constants"] if last else ", ".join(f"{k}={v:.6g}" for k, v in st.session_state.custom_constants.items()),
            "constants_dict": last["constants_dict"] if last else dict(st.session_state.custom_constants),
            "df": last["raw_df"] if last else edited_df,
            "manual_text": st.session_state.manual_text,
            "topic": detect_topic(prompt, title_ctx, st.session_state.manual_text),
        }
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("원리·데이터·참고 자료를 종합해 답변을 작성하는 중..."):
                answer, engine = mentor_answer(prompt, ctx, llm_cfg, st.session_state.chat_messages[:-1])
            st.markdown(answer, unsafe_allow_html=True)
            st.caption(f"엔진: {engine}")
        st.session_state.chat_messages.append({"role": "assistant", "content": answer, "engine": engine})

    # ---- 저작권 표기 ---------------------------------------------------------------
    st.markdown(
        """
        <div class="footer-note">
            © 2026 물리실험 결과 분석 시스템. All rights reserved.<br>
            <b>원작자 및 저작권자:</b> 박민후 (kj0419mh@gmail.com)<br>
            본 프로그램의 소스 코드와 UI 구조는 저작권법에 의해 보호됩니다. 무단 복제 및 상업적 이용을 금합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
