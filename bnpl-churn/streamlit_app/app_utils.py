from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

ACCENT = "#2F5D8A"
ACCENT_2 = "#C0504D"
MUTED = "#9AA5B1"
LIGHT = "#E6ECF2"

CHURN_COLORS = {0: ACCENT, 1: ACCENT_2}

def apply_chart_style():

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

    p = DATA_DIR / "bnpl_customer_features.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_model_features():

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
        "top_shap_features": {}, "T_ref": "-",
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
