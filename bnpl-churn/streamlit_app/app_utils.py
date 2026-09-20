# -*- coding: utf-8 -*-
"""Tiện ích dùng chung cho web app BNPL Churn — load dữ liệu/model (có cache) và style biểu đồ."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


# Bảng màu tiết chế: 1 màu chủ đạo + 1 màu nhấn + xám trung tính
ACCENT = "#2F5D8A"      # xanh đậm — phần lớn biểu đồ
ACCENT_2 = "#C0504D"    # đỏ gạch — chỉ dùng cho churn / cảnh báo
MUTED = "#9AA5B1"       # xám — đường tham chiếu, nhãn phụ
LIGHT = "#E6ECF2"       # xám nhạt — lưới, khung

CHURN_COLORS = {0: ACCENT, 1: ACCENT_2}


def apply_chart_style():
    """Style matplotlib tối giản: bỏ viền trên/phải, lưới mờ, chữ nhỏ vừa phải."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.grid": True,
        "grid.color": LIGHT,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    })


def section(title: str, note: str | None = None):
    """Tiêu đề mục gọn: gạch ngang + chữ đậm, chú thích nhỏ (nếu có)."""
    st.markdown("---")
    st.markdown(f"**{title}**")
    if note:
        st.caption(note)


@st.cache_data(show_spinner=False)
def load_transactions():
    p = DATA_DIR / "bnpl_transactions_clean.csv"
    return pd.read_csv(p, parse_dates=["transaction_date"]) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_customer_features():
    """Bảng đặc trưng TOÀN KỲ (T11) — 43 cột, có lifecycle_segment/churn_label.
    Chỉ dùng cho EDA ở Trang 1-2 (mục 3.3 báo cáo). KHÔNG dùng để huấn luyện/dự báo."""
    p = DATA_DIR / "bnpl_customer_features.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_model_features():
    """Bảng đặc trưng tính tại mốc T_ref (T14) — 27 đặc trưng + churn.
    Dùng cho Trang 3 (file mẫu chấm điểm hàng loạt) và huấn luyện (mục 3.4 báo cáo)."""
    p = DATA_DIR / "bnpl_model_features.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_report(name: str):
    p = REPORTS_DIR / name
    return pd.read_csv(p, index_col=0) if p.exists() else None


GDRIVE_MODEL_URL = ""

MODEL_CANDIDATES = [
    MODELS_DIR / "churn_model_final.pkl",
    MODELS_DIR / "churn_model_final.joblib",
    Path(__file__).resolve().parent / "models" / "churn_model_final.pkl",
]


def _wrap_bare_pipeline(obj):
    """Nếu artifact chỉ là Pipeline (không có metadata), dựng metadata tối thiểu từ bảng đặc trưng T_ref."""
    feat_path = DATA_DIR / "bnpl_model_features.csv"
    if not feat_path.exists():
        return None
    df = pd.read_csv(feat_path, index_col="customer_id")
    y = df["churn"] if "churn" in df.columns else None
    X = df.drop(columns=[c for c in ["churn"] if c in df.columns])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return {
        "pipeline": obj, "best_name": "Mô hình dự báo churn",
        "num_cols": num_cols, "cat_cols": cat_cols,
        "num_stats": {c: {"min": float(X[c].min()), "med": float(X[c].median()), "max": float(X[c].max())}
                      for c in num_cols},
        "cat_values": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "metrics_test": {}, "churn_rate": float(y.mean()) if y is not None else 0.0,
        "top_shap_features": {}, "T_ref": "—",
    }


def _download_from_drive(url: str, dest: Path) -> bool:
    try:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_artifact():
    """Trả về dict artifact, hoặc None. Lý do lỗi (nếu có) được ghi vào st.session_state['model_load_error']."""
    tried = []
    for p in MODEL_CANDIDATES:
        tried.append(str(p))
        if p.exists():
            obj = joblib.load(p)
            return obj if isinstance(obj, dict) else _wrap_bare_pipeline(obj)

    url = GDRIVE_MODEL_URL
    if not url:
        try:
            url = st.secrets.get("GDRIVE_MODEL_URL", "")
        except Exception:      
            url = ""
    if url:
        dest = MODELS_DIR / "churn_model_final.pkl"
        if _download_from_drive(url, dest):
            obj = joblib.load(dest)
            return obj if isinstance(obj, dict) else _wrap_bare_pipeline(obj)
        tried.append(f"Google Drive: {url}")

    st.session_state["model_load_error"] = tried
    return None


def require(obj, what: str, hint: str):
    if obj is None:
        st.error(f"Không tìm thấy {what}. {hint}")
        st.stop()
