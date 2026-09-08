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
    p = DATA_DIR / "bnpl_customer_features.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_report(name: str):
    p = REPORTS_DIR / name
    return pd.read_csv(p, index_col=0) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_artifact():
    p = MODELS_DIR / "churn_model_final.pkl"
    return joblib.load(p) if p.exists() else None


def require(obj, what: str, hint: str):
    if obj is None:
        st.error(f"Không tìm thấy {what}. {hint}")
        st.stop()
