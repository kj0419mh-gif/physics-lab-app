# -*- coding: utf-8 -*-
"""
물리실험 결과 분석 시스템
- 원작자 및 저작권자: 박민후 (kj0419mh@gmail.com)

구성
  1. 페이지 설정 / 스타일
  2. 세션 상태 초기화
  3. 수식 계산 엔진  (AST 기반 안전 계산, 공백·한글 변수명 지원)
  4. 매뉴얼 분석 엔진 (템플릿 매칭 + 범용 수식·상수·변수 추출)
  5. 통계·감도 분석 및 리포트 생성 엔진
  6. UI (사이드바 / 메인 / 누적 보드 / AI 멘토)
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import re
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
    "통계·감도 분석이 포함된 심층 학술 리포트와 실시간 교차 검증, 시각화를 제공합니다. "
    "**모든 종류의 물리 실험**에 범용적으로 사용할 수 있습니다."
)

MEAS_KEY = "실험 측정값"


def _setup_korean_font() -> bool:
    """matplotlib 한글 폰트 설정. 사용 가능한 폰트가 없으면 False (그래프 라벨은 영문으로 대체)."""
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
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    ss = st.session_state
    ss.setdefault("history", [])
    ss.setdefault("chat_messages", [])
    ss.setdefault("custom_constants", {})
    ss.setdefault("input_df", pd.DataFrame({"변수 1": [np.nan], MEAS_KEY: [np.nan]}))
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
    """수식에서 사용할 수 있는 함수/상수 네임스페이스. 삼각함수는 선택한 각도 단위를 따른다."""
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
    """공백·한글·특수문자가 포함된 변수/상수명을 안전한 토큰(__vN__)으로 치환한다. (긴 이름 우선)"""
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
    formula: str,
    variables: dict[str, Any],
    constants: dict[str, float],
    angle_unit: str = "deg",
) -> tuple[float | None, str]:
    """수식을 안전하게 계산한다. 반환: (값 또는 None, 오류 메시지 또는 '')"""
    if not formula or not formula.strip():
        return None, "수식이 비어 있습니다."

    ns = _make_namespace(angle_unit)
    all_names = {**constants, **variables}  # 표 변수가 상수보다 우선
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
    """수식에서 사용된 표 변수 / 상수 / 정의되지 않은 기호 목록을 반환한다 (UI 검증용)."""
    result = {"vars": [], "consts": [], "unknown": []}
    if not formula.strip():
        return result
    names = list(dict.fromkeys(list(variables) + list(constants)))
    substituted, mapping = _substitute_names(formula, names)
    for token, name in mapping.items():
        if name in variables:
            result["vars"].append(name)
        else:
            result["consts"].append(name)
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
    """표의 각 행에 대해 이론값을 계산한다. 완전히 빈 행은 제외."""
    meas_col = _find_meas_col(df)
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        variables = {
            _base_name(c): _to_float(row[c]) for c in df.columns if c != meas_col and _base_name(c)
        }
        meas = _to_float(row[meas_col]) if meas_col else float("nan")
        if all(math.isnan(v) for v in variables.values()) and math.isnan(meas):
            continue  # 빈 행
        theo, err = safe_eval_formula(formula, variables, constants, angle_unit)
        rows.append({"idx": i, "vars": variables, "meas": meas, "theo": theo, "err": err})
    return meas_col, rows


# ---------------------------------------------------------------------------
# 4. 매뉴얼 분석 엔진 (템플릿 + 범용 추출)
# ---------------------------------------------------------------------------

def _nan_rows(n_cols: int, n_rows: int = 3) -> list[list[float]]:
    return [[np.nan] * n_cols for _ in range(n_rows)]


# 자주 쓰이는 실험 템플릿. 매뉴얼 키워드가 일치하면 우선 적용되고, 본문 추출 결과가 병합된다.
KNOWN_TEMPLATES: list[dict[str, Any]] = [
    {
        "keys": ["컴프턴", "compton"],
        "title": "컴프턴 산란 실험",
        "constants": {"E0": 661.7, "mc2": 511.0},
        "formulas": [
            "E0 / (1 + (E0/mc2) * (1 - cos(산란각)))",
            "1 / (1/E0 + (1 - cos(산란각))/mc2)",
        ],
        "columns": ["산란각 [deg]", "산란체 유무 [1=O, 0=X]", f"{MEAS_KEY} [keV]"],
        "data": [[30.0, 1.0, 516.0], [60.0, 1.0, 379.0], [90.0, 1.0, 269.0]],
    },
    {
        "keys": ["비전하", "e/m", "헬름홀츠", "helmholtz", "전자의 전하"],
        "title": "전자의 비전하(e/m) 측정",
        "constants": {"N": 130.0, "R": 0.15, "mu0": 1.2566e-6},
        "formulas": [
            "2 * 가속전압 / ((0.7155 * mu0 * N * 코일전류 / R)**2 * 궤도반지름**2)",
            "2 * 가속전압 / (자기장**2 * 궤도반지름**2)",
        ],
        "columns": ["가속전압 [V]", "코일전류 [A]", "궤도반지름 [m]", f"{MEAS_KEY} [C/kg]"],
        "data": [[150.0, 1.35, 0.04, 1.72e11], [200.0, 1.55, 0.04, 1.70e11]],
    },
    {
        "keys": ["프랑크", "franck", "헤르츠", "hertz"],
        "title": "프랑크-헤르츠 실험",
        "constants": {"E_Hg": 4.9},
        "formulas": ["피크 번호 * E_Hg"],
        "columns": ["피크 번호", f"{MEAS_KEY} [V]"],
        "data": [[1.0, 5.1], [2.0, 10.0], [3.0, 14.8]],
    },
    {
        "keys": ["광전", "photoelectric", "플랑크 상수", "planck", "정지 전압", "저지 전압"],
        "title": "광전효과와 플랑크 상수 측정",
        "constants": {"h": 6.626e-34, "c": 2.998e8, "e": 1.602e-19, "W": 2.3},
        "formulas": ["h*c / (파장 * 1e-9 * e) - W"],
        "columns": ["파장 [nm]", f"{MEAS_KEY} [V]"],
        "data": [[365.0, 1.08], [405.0, 0.74], [436.0, 0.52], [546.0, 0.02]],
    },
    {
        "keys": ["밀리컨", "millikan", "기름방울", "유적"],
        "title": "밀리컨 기름방울 실험 (기본 전하량)",
        "constants": {"e_ref": 1.602e-19},
        "formulas": ["전자 개수 * e_ref"],
        "columns": ["전자 개수", f"{MEAS_KEY} [C]"],
        "data": [[1.0, 1.65e-19], [2.0, 3.15e-19], [3.0, 4.90e-19]],
    },
    {
        "keys": ["단진자", "진자", "pendulum", "중력가속도"],
        "title": "단진자를 이용한 중력가속도 측정",
        "constants": {"g": 9.80665},
        "formulas": ["2 * pi * sqrt(길이 / g)", "4 * pi**2 * 길이 / 주기**2"],
        "columns": ["길이 [m]", f"{MEAS_KEY} [s]"],
        "data": [[0.5, 1.43], [0.8, 1.80], [1.0, 2.01]],
    },
    {
        "keys": ["자유낙하", "free fall", "낙하"],
        "title": "자유낙하 실험",
        "constants": {"g": 9.80665},
        "formulas": ["0.5 * g * 시간**2", "sqrt(2 * 높이 / g)"],
        "columns": ["시간 [s]", f"{MEAS_KEY} [m]"],
        "data": [[0.3, 0.45], [0.4, 0.80], [0.5, 1.24]],
    },
    {
        "keys": ["이중슬릿", "단일슬릿", "회절", "간섭", "double slit", "diffraction", "interference"],
        "title": "빛의 간섭·회절 (슬릿 실험)",
        "constants": {"lam": 632.8},
        "formulas": ["차수 * (lam * 1e-9) * 스크린 거리 / (슬릿 간격 * 1e-3) * 1e3"],
        "columns": ["차수", "스크린 거리 [m]", "슬릿 간격 [mm]", f"{MEAS_KEY} [mm]"],
        "data": [[1.0, 1.0, 0.25, 2.5], [2.0, 1.0, 0.25, 5.1], [3.0, 1.0, 0.25, 7.6]],
    },
    {
        "keys": ["브래그", "bragg", "x선", "x-ray", "엑스선"],
        "title": "브래그 회절 (X선 파장 측정)",
        "constants": {"d": 0.2820},
        "formulas": ["2 * d * sin(입사각) / 차수"],
        "columns": ["차수", "입사각 [deg]", f"{MEAS_KEY} [nm]"],
        "data": [[1.0, 7.2, 0.0705], [2.0, 14.5, 0.0708]],
    },
    {
        "keys": ["마이컬슨", "michelson"],
        "title": "마이컬슨 간섭계 실험",
        "constants": {"lam": 632.8},
        "formulas": ["무늬 이동 수 * lam * 1e-3 / 2"],
        "columns": ["무늬 이동 수", f"{MEAS_KEY} [μm]"],
        "data": [[20.0, 6.4], [40.0, 12.5], [60.0, 19.1]],
    },
    {
        "keys": ["옴의 법칙", "ohm", "저항 측정"],
        "title": "옴의 법칙 검증",
        "constants": {"R": 100.0},
        "formulas": ["전압 / R"],
        "columns": ["전압 [V]", f"{MEAS_KEY} [A]"],
        "data": [[1.0, 0.0102], [2.0, 0.0199], [3.0, 0.0305]],
    },
    {
        "keys": ["rc 회로", "rc회로", "축전기", "capacitor", "시간 상수", "시간상수"],
        "title": "RC 회로 충·방전 실험",
        "constants": {"V0": 5.0, "R": 1.0e4, "C": 1.0e-4},
        "formulas": ["V0 * exp(-시간 / (R * C))", "V0 * (1 - exp(-시간 / (R * C)))"],
        "columns": ["시간 [s]", f"{MEAS_KEY} [V]"],
        "data": [[0.5, 3.05], [1.0, 1.86], [2.0, 0.68]],
    },
    {
        "keys": ["훅", "hooke", "용수철", "스프링"],
        "title": "훅의 법칙 (용수철 상수 측정)",
        "constants": {"k": 25.0},
        "formulas": ["k * 늘어난 길이"],
        "columns": ["늘어난 길이 [m]", f"{MEAS_KEY} [N]"],
        "data": [[0.02, 0.49], [0.04, 1.02], [0.06, 1.47]],
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
            t = re.sub("Δ(?=\\w)", "Delta", t)               # ΔT → DeltaT (하나의 변수로 취급)
        t = re.sub(re.escape(g) + r"(?=[\d_])", name, t)   # θ1 → theta1
        t = t.replace(g, f" {name} ")                       # cosθ → cos theta
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
    rhs = re.split(r"[가-힣]", rhs, maxsplit=1)[0]                       # 한글 설명 이전까지
    rhs = re.split(r"\b(?:where|with|for|and|is|the)\b|[,;:]", rhs, maxsplit=1)[0]
    rhs = re.sub(r"\(\s*\d+\s*\)\s*$", "", rhs)                          # 식 번호 (1)
    return re.sub(r"[.\s]+$", "", rhs).strip()


def _formula_to_python(rhs: str) -> str:
    """매뉴얼 표기(생략된 곱셈, cosθ, sin²θ 등)를 파이썬 수식으로 변환한다."""
    f = _to_python_expr(rhs)
    f = re.sub(r"\b(sin|cos|tan)\s*\*\*\s*2\s*\(?\s*([A-Za-z_]\w*)\s*\)?", r"\1(\2)**2", f)
    f = re.sub(r"\b(sin|cos|tan|sqrt|exp|ln|log)(?=[A-Za-z_])", r"\1 ", f)
    f = re.sub(r"\b(sin|cos|tan|sqrt|exp|ln|log)\s+([A-Za-z_]\w*)", r"\1(\2)", f)
    f = re.sub(r"(\d)\s*(?![eE][-+]?\d)([A-Za-z(])", r"\1*\2", f)        # 2pi → 2*pi
    f = re.sub(r"\)\s*([A-Za-z0-9(])", r")*\1", f)                       # )( → )*(
    f = re.sub(
        r"([A-Za-z_]\w*)\s*\(",
        lambda m: m.group(0) if m.group(1) in _FUNC_NAMES else m.group(1) + "*(",
        f,
    )
    f = re.sub(r"([\w)])\s+(?=[\w(])", r"\1*", f)                         # 공백 = 곱셈
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
    """매뉴얼 텍스트 → {title, constants, formulas, columns, data, desc, extracted}"""
    if not text.strip():
        s = copy.deepcopy(KNOWN_TEMPLATES[0])
        s.pop("keys", None)
        s["desc"] = "매뉴얼이 업로드되지 않아 기본 예시(컴프턴 산란) 템플릿을 사용 중입니다. PDF를 올리면 즉시 재분석됩니다."
        s["extracted"] = []
        return s

    norm = _normalize_text(text)
    tmpl, hits = _match_template(text)
    consts = _extract_constants(norm)
    formulas = _extract_formulas(norm, consts)
    title = _extract_title(text)

    if tmpl:
        s = copy.deepcopy(tmpl)
        s.pop("keys", None)
        for k, v in consts.items():
            s["constants"].setdefault(k, v)
        for f in formulas:
            if f["rhs"] not in s["formulas"]:
                s["formulas"].append(f["rhs"])
        s["desc"] = (
            f"매뉴얼을 「{s['title']}」 실험으로 식별했습니다 (감지 키워드: {', '.join(hits)}). "
            f"본문에서 상수 {len(consts)}개, 수식 {len(formulas)}개를 추가로 추출해 추천 목록에 병합했습니다. "
            "본문 추출 수식을 선택하면 필요한 변수 컬럼을 아래에서 한 번에 추가할 수 있습니다."
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
            "title": title or "사용자 정의 실험",
            "constants": consts,
            "formulas": [f["rhs"] for f in formulas],
            "columns": cols,
            "data": _nan_rows(len(cols)),
            "desc": (
                f"알려진 템플릿과 일치하지 않아 매뉴얼 본문에서 직접 추출했습니다 — "
                f"수식 {len(formulas)}개, 상수 {len(consts)}개. 대표 수식 `{main['lhs']} = {main['rhs']}` 의 "
                f"변수({', '.join(main['vars'])})로 표를 구성했습니다. 단위와 값은 확인 후 수정하세요."
            ),
        }
    else:
        s = {
            "title": title or "사용자 정의 실험",
            "constants": consts,
            "formulas": [""],
            "columns": ["변수 1", MEAS_KEY],
            "data": _nan_rows(2),
            "desc": (
                f"매뉴얼에서 수식을 자동 추출하지 못했습니다 (상수 {len(consts)}개 추출). "
                "PDF가 이미지 스캔본이면 텍스트가 없을 수 있습니다. 수식과 표 컬럼을 직접 설정해 주세요."
            ),
        }
    s["extracted"] = formulas
    s["formulas"] = [f for f in s["formulas"] if f is not None]
    return s


@st.cache_data(show_spinner=False)
def _extract_pdf_text(data: bytes) -> str:
    import io
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
    """수치 편미분으로 각 변수/상수의 탄성도(x 1% 변화 시 결과의 % 변화)를 구한다."""
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
            v_plus = dict(variables)
            c_plus = dict(constants)
            v_minus = dict(variables)
            c_minus = dict(constants)
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
    stats.update(
        theo_mean=float(np.mean(theo)), meas_mean=float(np.mean(meas)),
        err_pct=err_pct, mean_err=float(np.nanmean(err_pct)) if np.any(~np.isnan(err_pct)) else float("nan"),
        max_err=float(np.nanmax(err_pct)) if np.any(~np.isnan(err_pct)) else float("nan"),
        max_err_idx=int(np.nanargmax(err_pct)) if np.any(~np.isnan(err_pct)) else 0,
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


def build_report(
    title: str, formula: str, const_str: str, stats: dict[str, Any], sens: list[dict[str, Any]]
) -> dict[str, str]:
    """데이터에 근거한 학술 리포트(마크다운) 3종을 생성한다."""
    n = stats["n"]
    diag: list[str] = []
    if n >= 2 and not math.isnan(stats["slope"]):
        if abs(stats["slope"] - 1) > 0.05:
            diag.append(
                f"- 회귀 기울기 {stats['slope']:.3f}가 1에서 벗어나 **비례(이득) 오차**가 의심됩니다. "
                "장비 교정 계수, 단위 환산(예: deg/rad, cm/m), 상수 값을 점검하세요."
            )
        if abs(stats["intercept"]) > 0.05 * abs(stats["theo_mean"]) and stats["theo_mean"] != 0:
            diag.append(
                f"- 회귀 절편 {stats['intercept']:.4g}가 0에서 유의하게 벗어나 **영점(offset) 오차**가 의심됩니다. "
                "측정기 영점 조정, 배경(background) 보정 여부를 확인하세요."
            )
    if stats["same_sign"]:
        direction = "이론값이 측정값보다 항상 큼" if stats["bias"] > 0 else "측정값이 이론값보다 항상 큼"
        diag.append(
            f"- 모든 잔차의 부호가 동일합니다 ({direction}). 무작위 오차만으로는 설명되지 않는 **체계적 편향**이 존재합니다."
        )
    if not diag:
        diag.append("- 뚜렷한 비례·영점·부호 편향이 관찰되지 않아 **무작위(통계적) 오차**가 지배적인 것으로 판단됩니다.")

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
            f"| {d['kind']} | `{d['name']}` | {d['value']:.4g} | {d['deriv']:.4g} | {d['elasticity']:+.3f} |"
            for d in sens
        )
        top = sens[0]
        sens_md = f"""
**감도 분석 (Sensitivity / Uncertainty Propagation)**

대표 행(유효 데이터의 평균값)에서 수치 편미분 ∂f/∂x 를 계산했습니다.
**탄성도**는 해당 변수가 1% 변할 때 결과값이 몇 % 변하는지를 의미하며, 1차 불확도 전파식
u(f)² = Σ (∂f/∂xᵢ)² u(xᵢ)² 의 가중치에 해당합니다.

| 종류 | 기호 | 대입값 | ∂f/∂x | 탄성도 |
|---|---|---|---|---|
{rows}

- 결과에 가장 민감한 양은 **`{top['name']}`** (탄성도 {top['elasticity']:+.3f}) 입니다. 이 양의 측정 정밀도가 최종 불확도를 지배하므로 가장 정밀하게 측정·교정해야 합니다.
- 탄성도 절댓값이 1보다 크면 오차가 **증폭**되고, 1보다 작으면 **완화**됩니다.
"""
    else:
        sens_md = "감도 분석을 수행할 유효한 행이 없습니다 (수식 계산이 가능한 행이 필요합니다)."

    improve = f"""
**4. 오차 원인 분류 (Error Taxonomy)**
- **기기·교정 오차 (Instrumental)**: 측정기 분해능, 영점 이탈, 이득(gain) 교정 곡선의 비선형성. → 회귀 기울기·절편 진단 참고.
- **기하·정렬 오차 (Geometric)**: 검출기/광원/시료의 정렬 불량, 유효 입체각·유효 길이의 유한 크기 효과.
- **환경 오차 (Environmental)**: 온도·습도·배경 잡음(자연 방사선, 외부 자기장, 진동, 주변광).
- **통계 오차 (Statistical)**: 반복 횟수·계수 시간 부족에 따른 무작위 변동 (표본 수 n = {n}).
- **모델 가정 오차 (Model)**: 이론식이 가정한 이상화(질점, 무마찰, 단일 산란, 균일장 등)와 실제 조건의 차이.

**5. 개선 방안 (Recommendations)**
- {'체계적 편향이 관찰되므로, 표준 시료·기준 선원 등 **알려진 참값으로 다점 교정**을 먼저 수행하세요.' if stats.get('same_sign') else '무작위 오차가 주도하므로 **반복 측정 횟수를 늘리고 평균·표준오차**를 함께 보고하세요.'}
- 최대 오차가 발생한 행 #{stats.get('max_err_idx', 0) + 1} 의 측정 조건(범위 끝단, 낮은 신호 등)을 재점검하고 필요 시 재측정하세요.
- 감도 분석에서 상위에 오른 변수를 우선적으로 정밀 측정하고, 불확도 예산(uncertainty budget)에 반영하세요.
- 배경(background) 측정을 동일 조건에서 별도로 수행하여 차감하고, 단위 환산(각도 단위 포함)을 재확인하세요.
"""
    return {"summary": summary, "sensitivity": sens_md, "improve": improve}


def mentor_answer(prompt: str, ctx: dict[str, Any]) -> str:
    """현재 실험 맥락과 최근 분석 통계를 활용해 동적으로 답변을 생성한다."""
    p = prompt.lower()
    title, formula, stats, sens = ctx["title"], ctx["formula"], ctx.get("stats"), ctx.get("sens") or []

    def stat_line() -> str:
        if not stats or stats.get("n", 0) == 0:
            return "아직 분석 결과가 없어서 일반적인 관점에서 말씀드릴게요. 위의 **분석 실행** 버튼을 누르면 실제 수치를 바탕으로 더 구체적으로 도와드릴 수 있어요."
        return (
            f"최근 분석 결과를 보면 데이터 {stats['n']}개의 평균 오차율은 **{stats['mean_err']:.2f}%**, "
            f"최대 오차는 행 #{stats['max_err_idx'] + 1}에서 **{stats['max_err']:.2f}%** 였어요. "
            f"평균 잔차(이론−측정)는 {stats['bias']:+.4g}"
            + (" 이고, 모든 잔차의 부호가 같아 체계적 편향이 의심돼요." if stats.get("same_sign") else " 이에요.")
        )

    if any(k in p for k in ["오차", "error", "줄이", "틀리", "차이", "왜"]):
        tips = ""
        if stats and not math.isnan(stats.get("slope", float("nan"))):
            if abs(stats["slope"] - 1) > 0.05:
                tips += f"\n- 회귀 기울기가 {stats['slope']:.3f}로 1에서 벗어나 있어요 → **비례 오차**: 상수값·단위 환산(deg/rad, cm/m)·교정 계수를 먼저 확인해 보세요."
            if stats["theo_mean"] and abs(stats["intercept"]) > 0.05 * abs(stats["theo_mean"]):
                tips += f"\n- 절편이 {stats['intercept']:.4g}로 0에서 벗어나 있어요 → **영점 오차**: 측정기 영점·배경 보정을 점검해 보세요."
        if sens:
            tips += f"\n- 감도 분석상 결과에 가장 민감한 양은 `{sens[0]['name']}` (탄성도 {sens[0]['elasticity']:+.2f}) 이에요. 이 양의 측정 정밀도를 높이는 것이 가장 효과적이에요."
        return (
            f"**{title}** 실험의 수식 `{formula}` 을 기준으로 함께 살펴봤어요. 🎯\n\n{stat_line()}\n\n"
            "오차는 보통 이론 모델의 이상적인 가정과 실제 장비 사이의 간극에서 생겨요.\n"
            "1. **기기·교정**: 영점, 이득(gain), 분해능 한계\n"
            "2. **기하·정렬**: 정렬 불량, 유한 크기 효과\n"
            "3. **통계**: 반복 횟수·계수 시간 부족\n"
            "4. **모델 가정**: 무마찰·질점·단일 산란 등 이상화" + (f"\n\n**데이터 기반 힌트**{tips}" if tips else "")
            + "\n\n특정 행의 오차가 유독 크다면 그 조건(범위 끝단, 낮은 신호 등)을 함께 점검해 볼게요!"
        )

    if any(k in p for k in ["불확도", "uncertainty", "표준편차", "신뢰"]):
        sens_txt = ""
        if sens:
            sens_txt = "\n\n현재 수식의 감도(탄성도) 순위는 다음과 같아요:\n" + "\n".join(
                f"- `{d['name']}`: {d['elasticity']:+.3f} (1% 변화 → 결과 {abs(d['elasticity']):.2f}% 변화)" for d in sens[:5]
            )
        return (
            f"불확도가 궁금하시군요! 🌿 **{title}** 에서 불확도는 참값이 존재할 것으로 기대되는 구간의 폭이에요.\n\n"
            f"수식 `{formula}` 의 각 입력량 xᵢ 의 불확도 u(xᵢ)는 1차 전파식\n\n"
            "$$u(f)^2 = \\sum_i \\left(\\frac{\\partial f}{\\partial x_i}\\right)^2 u(x_i)^2$$\n\n"
            "으로 결과의 불확도에 기여해요. 시스템은 이 편미분을 수치적으로 계산해 감도 분석 탭에 보여주고 있어요."
            f"{sens_txt}\n\n확장 불확도(k≈2, 약 95% 신뢰수준)를 함께 보고하면 데이터의 신뢰도를 객관적으로 보여줄 수 있어요."
        )

    if any(k in p for k in ["수식", "공식", "유도", "formula", "식이", "의미"]):
        used = ctx.get("symbols", {})
        return (
            f"현재 **{title}** 에 적용 중인 수식은 `{formula}` 이에요. 💡\n\n"
            f"- 표에서 가져오는 변수: {', '.join(f'`{v}`' for v in used.get('vars', [])) or '없음'}\n"
            f"- 사이드바 상수: {', '.join(f'`{c}`' for c in used.get('consts', [])) or '없음'}\n"
            f"- 정의되지 않은 기호: {', '.join(f'`{u}`' for u in used.get('unknown', [])) or '없음 ✅'}\n\n"
            "수식은 매뉴얼에서 추출된 후보 중 선택하거나 직접 편집할 수 있어요. 삼각함수 각도 단위는 사이드바에서 deg/rad 로 바꿀 수 있고, "
            "`sqrt`, `exp`, `log`, `pi` 등을 지원해요. 특정 항의 물리적 의미가 궁금하면 그 항을 콕 집어 물어보세요!"
        )

    if any(k in p for k in ["그래프", "기울기", "graph", "plot", "회귀", "상관"]):
        if stats and stats.get("n", 0) >= 2 and not math.isnan(stats.get("r", float("nan"))):
            return (
                f"그래프 해석을 도와드릴게요! 📈 **{title}** 의 측정값 vs 이론값 산점도에서\n\n"
                f"- 상관계수 r = **{stats['r']:.4f}** (1에 가까울수록 이론 경향을 잘 따름)\n"
                f"- 회귀식: 측정 = **{stats['slope']:.4f}**·이론 + **{stats['intercept']:.4g}**\n\n"
                "이상적인 경우 기울기 1, 절편 0인 점선(y = x)에 점들이 놓여요. 기울기가 1에서 벗어나면 비례(교정) 오차, "
                "절편이 0에서 벗어나면 영점 오차, 점들이 y = x 주변에 무작위로 흩어져 있으면 통계 오차가 주된 원인이에요. "
                "아래쪽 잔차 막대그래프에서 특정 행만 튀어나오면 그 행의 측정 조건을 다시 확인해 보세요."
            )
        return "그래프 해석은 분석을 2개 이상의 데이터 행으로 실행하면 상관계수·회귀식과 함께 구체적으로 설명해 드릴 수 있어요. 📈 먼저 **분석 실행** 버튼을 눌러 보세요!"

    if any(k in p for k in ["개선", "방법", "어떻게", "improve", "추천", "팁"]):
        return (
            f"**{title}** 실험을 더 정밀하게 하는 방법을 정리해 드릴게요. 🛠️\n\n{stat_line()}\n\n"
            "1. **교정 우선**: 알려진 참값(표준 시료·기준 선원)으로 다점 교정을 먼저 수행하세요.\n"
            "2. **반복 측정**: 같은 조건을 3회 이상 반복하고 평균과 표준오차를 함께 기록하세요.\n"
            "3. **배경 차감**: 시료/신호 없는 상태의 배경을 동일 조건에서 측정해 빼 주세요.\n"
            "4. **민감 변수 집중**: 감도 분석 상위 변수의 측정 정밀도를 높이는 것이 가장 효율적이에요.\n"
            "5. **범위 끝단 주의**: 오차가 큰 극단 조건은 측정 시간을 늘리거나 조건을 재설정하세요."
        )

    if any(k in p for k in ["상수", "constant", "값", "단위"]):
        return (
            f"상수와 단위는 결과에 직접 곱해지기 때문에 아주 중요해요. ⚖️ 현재 적용 중인 상수는 **{ctx.get('const_str') or '없음'}** 이에요.\n\n"
            "- 사이드바에서 값을 바로 수정하거나 추가/삭제할 수 있고, 매뉴얼을 바꾸면 추천 상수가 자동으로 갱신돼요.\n"
            "- 표 컬럼의 단위(예: cm)와 수식이 기대하는 단위(예: m)가 다르면 수식에 환산 계수(×1e-2)를 넣거나 컬럼 단위를 맞춰 주세요.\n"
            "- 각도 컬럼이 deg 인지 rad 인지에 따라 사이드바의 삼각함수 각도 단위도 맞춰야 해요."
        )

    return (
        f"'{prompt}' 에 대해 **{title}** 실험 관점에서 살펴봤어요! 💡\n\n{stat_line()}\n\n"
        f"현재 수식 `{formula}` 을 기준으로, 측정값과 이론값의 차이는 단순 실수보다 장비의 기하학적 특성·교정 상태·환경 요인에서 비롯되는 경우가 많아요. "
        "보드의 잔차 분포와 감도 분석 탭을 함께 보면 어떤 체계적 편향이 작용하는지 파악하기 쉬워요.\n\n"
        "더 구체적으로 **오차 원인**, **불확도**, **수식 의미**, **그래프 해석**, **개선 방법** 중 궁금한 것을 물어보시면 데이터 기반으로 자세히 설명해 드릴게요!"
    )


# ---------------------------------------------------------------------------
# 6. UI
# ---------------------------------------------------------------------------

def _apply_suggestion() -> None:
    """AI 추천(제목·상수·수식·표)을 세션에 적용한다. 위젯 생성 전 또는 콜백에서만 호출."""
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


def _render_sidebar() -> str:
    st.sidebar.header("⚙️ 장비 상수 및 계산 설정")

    angle_unit = st.sidebar.selectbox(
        "삼각함수 각도 단위 (sin/cos/tan 입력값)", ["deg", "rad"], key="angle_unit",
        help="표의 각도 컬럼 단위와 일치시켜 주세요.",
    )

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
        st.session_state.custom_constants[k] = float(
            st.sidebar.number_input(k, key=wkey, format="%.6g", step=None)
        )

    st.sidebar.divider()
    if st.sidebar.button("🧹 분석 보드·대화 초기화", use_container_width=True):
        st.session_state.history = []
        st.session_state.chat_messages = []
        st.rerun()
    return angle_unit


def main() -> None:
    # ---- 1. 매뉴얼 업로드 및 자동 분석 ---------------------------------
    st.header("📄 1. 매뉴얼 업로드 & AI 자동 분석")
    uploaded_files = st.file_uploader(
        "실험 매뉴얼 PDF를 업로드하세요. 파일이 바뀌면 실험 제목·상수·수식·표 형식이 즉시 다시 추천됩니다.",
        type=["pdf"],
        accept_multiple_files=True,
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
        # 매뉴얼이 바뀔 때마다(추가·교체·삭제) 재분석하고, 자동 적용이 켜져 있으면 즉시 반영
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

    angle_unit = _render_sidebar()

    # ---- 2. 실험 설정 및 수식 선택 -------------------------------------
    st.divider()
    st.header("🛠️ 2. 실험 설정 및 수식 선택")

    st.markdown(
        f"<div class='ai-box'><b>🤖 AI 매뉴얼 분석 및 추천 설정</b><br>{sugg['desc']}</div>",
        unsafe_allow_html=True,
    )
    if sugg.get("extracted"):
        with st.expander(f"🔎 매뉴얼 본문에서 추출한 수식 {len(sugg['extracted'])}개 보기"):
            st.dataframe(
                pd.DataFrame(
                    [{"좌변": f["lhs"], "우변(파이썬 표기)": f["rhs"], "변수": ", ".join(f["vars"])} for f in sugg["extracted"]]
                ),
                use_container_width=True, hide_index=True,
            )

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
    st.radio(
        "매뉴얼에서 추출·추천된 수식:", options=options, key="formula_radio",
        on_change=_sync_formula_from_radio, label_visibility="collapsed",
    )
    st.session_state.setdefault("formula_input", options[0])
    raw_formula = st.text_input(
        "이론 수식 (수정 가능) — 표 컬럼명(단위 제외)과 사이드바 상수 기호를 그대로 사용",
        key="formula_input",
        placeholder="예: E0 / (1 + (E0/mc2) * (1 - cos(산란각)))",
    )

    # 수식 기호 검증
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
                cols = [c for c in df.columns if c != meas_col] + [meas_col]
                df = df[cols]
            st.session_state.input_df = df
            st.session_state.table_version += 1
            st.rerun()
        if cc2.button("⚙️ 누락 기호를 상수로 추가 (값 0)", use_container_width=True):
            for name in symbols["unknown"]:
                st.session_state.custom_constants.setdefault(name, 0.0)
            st.rerun()

    # ---- 데이터 표 ------------------------------------------------------
    st.markdown("**📊 실험 데이터 입력 표** (행 추가/삭제 가능, 컬럼명의 `[단위]` 앞부분이 수식 변수명입니다)")
    edited_df = st.data_editor(
        st.session_state.input_df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"data_editor_{st.session_state.table_version}",
    )

    with st.popover("⚙️ 표 컬럼 관리 (변수 추가/삭제)"):
        st.write("**새 변수 컬럼 추가**")
        new_col_name = st.text_input("변수명 (예: 전압, 반지름, theta)", key="new_col_name")
        unit_choice = st.selectbox(
            "단위",
            ["없음", "deg", "rad", "keV", "eV", "V", "A", "m", "cm", "mm", "nm", "μm", "s", "ms", "kg", "g", "C", "N", "T", "Hz", "K", "J"],
            key="new_col_unit",
        )
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
        if st.button("🧹 표 데이터 비우기 (컬럼 유지)", key="clear_rows_btn"):
            st.session_state.input_df = pd.DataFrame(
                _nan_rows(len(edited_df.columns)), columns=edited_df.columns
            ).astype(float)
            st.session_state.table_version += 1
            st.rerun()

    # ---- 3. 실시간 교차 검증 ----------------------------------------------
    st.divider()
    st.header("🔍 3. 실시간 교차 검증 (이론 예측 vs 실험 측정)")
    st.markdown(
        "<div class='small-note'>현재 수식·상수·표를 바탕으로 이론 예측값을 즉시 계산하여 측정값과 비교합니다. "
        "표나 수식을 수정하면 자동으로 갱신됩니다.</div>",
        unsafe_allow_html=True,
    )

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

        results, theo_list, meas_list, valid_vars = [], [], [], []
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
                record.update({
                    "이론값 (계산)": round(r["theo"], 6),
                    "절대 오차": round(abs(resid), 6),
                    "오차율 (%)": round(abs(resid) / abs(r["theo"]) * 100, 3) if r["theo"] != 0 else None,
                    "잔차 (이론−측정)": round(resid, 6),
                    "상태": "✅",
                })
                theo_list.append(r["theo"])
                meas_list.append(r["meas"])
                valid_vars.append(r["vars"])
            results.append(record)

        if not theo_list:
            st.error("이론값과 측정값이 모두 유효한 행이 하나도 없습니다. 위 교차 검증 표의 '상태'를 확인하세요.")
            return

        theo_arr, meas_arr = np.array(theo_list, dtype=float), np.array(meas_list, dtype=float)
        stats = compute_stats(theo_arr, meas_arr)
        used_consts = {k: v for k, v in st.session_state.custom_constants.items() if k in symbols["consts"]}
        const_str = ", ".join(f"{k}={v:.6g}" for k, v in used_consts.items())

        # 감도 분석: 유효 행의 변수 평균값을 대표점으로 사용
        rep_vars: dict[str, float] = {}
        for name in valid_vars[0]:
            vals = [vv[name] for vv in valid_vars if not math.isnan(vv.get(name, float("nan")))]
            rep_vars[name] = float(np.mean(vals)) if vals else float("nan")
        sens = sensitivity_analysis(raw_formula, rep_vars, st.session_state.custom_constants, angle_unit)
        report = build_report(exp_name, raw_formula, const_str, stats, sens)

        st.session_state.history.append({
            "id": len(st.session_state.history) + 1,
            "title": exp_name,
            "formula": raw_formula,
            "constants": const_str,
            "angle_unit": angle_unit,
            "df": pd.DataFrame(results),
            "report": report,
            "stats": stats,
            "sens": sens,
            "symbols": symbols,
            "theo_arr": theo_arr,
            "meas_arr": meas_arr,
            "row_ids": [r["idx"] for r in eval_rows if r["theo"] is not None and not math.isnan(r["meas"])],
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
                    f"""<div class="metric-card">
                    <b>사용 수식:</b> <code>{rec['formula']}</code><br>
                    <b>적용 상수:</b> {rec['constants'] or '없음'} &nbsp;|&nbsp; <b>각도 단위:</b> {rec['angle_unit']}
                    &nbsp;|&nbsp; <b>평균 오차율:</b> {rec['stats']['mean_err']:.3f}% &nbsp;|&nbsp; <b>판정:</b> {_grade(rec['stats']['mean_err'])}
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.dataframe(rec["df"], use_container_width=True, hide_index=True)

                col_graph, col_report = st.columns([1, 1.25])
                with col_graph:
                    st.markdown("**📈 측정값 vs 이론값 상관 & 잔차**")
                    ko = KOREAN_FONT_OK
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5.6), gridspec_kw={"height_ratios": [3, 1.6]})
                    theo, meas = rec["theo_arr"], rec["meas_arr"]
                    ax1.scatter(theo, meas, color="#3182ce", s=40, zorder=3, label="측정 데이터" if ko else "Measured")
                    lo = min(theo.min(), meas.min())
                    hi = max(theo.max(), meas.max())
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
    st.header("💬 5. AI 실험 멘토 (데이터 기반 Q&A)")
    st.markdown(
        "<div class='small-note'>실험에 대해 무엇이든 물어보세요. 최근 분석 결과의 실제 수치를 바탕으로 답변합니다. ☕</div>",
        unsafe_allow_html=True,
    )
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("자유롭게 질문하세요! (예: 고각도에서 오차가 큰 이유는?)"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        last = st.session_state.history[-1] if st.session_state.history else None
        ctx = {
            "title": last["title"] if last else exp_name,
            "formula": last["formula"] if last else raw_formula,
            "stats": last["stats"] if last else None,
            "sens": last["sens"] if last else None,
            "symbols": last["symbols"] if last else symbols,
            "const_str": last["constants"] if last else ", ".join(f"{k}={v:.6g}" for k, v in st.session_state.custom_constants.items()),
        }
        st.session_state.chat_messages.append({"role": "assistant", "content": mentor_answer(prompt, ctx)})
        st.rerun()

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
