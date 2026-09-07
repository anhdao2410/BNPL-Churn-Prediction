# -*- coding: utf-8 -*-
"""Tiện ích dùng chung cho web app BNPL Churn — load dữ liệu & model (có cache)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent          # thư mục gốc repo (bnpl-churn/)
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

PALETTE = {"green": "#2a9d8f", "orange": "#f2b134", "red": "#e76f51", "blue": "#457b9d",
          "purple": "#8e7cc3", "gray": "#9e9e9e"}


@st.cache_data(show_spinner="Đang tải dữ liệu giao dịch...")
def load_transactions():
    p = DATA_DIR / "bnpl_transactions_clean.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["transaction_date"])
    return df


@st.cache_data(show_spinner="Đang tải hồ sơ khách hàng...")
def load_customer_features():
    p = DATA_DIR / "bnpl_customer_features.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


@st.cache_data(show_spinner="Đang tải báo cáo cohort/category...")
def load_report(name: str):
    p = REPORTS_DIR / name
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_artifact():
    p = MODELS_DIR / "churn_model_final.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


def require(obj, what: str, hint: str):
    """Dừng trang với hướng dẫn nếu thiếu dữ liệu/model."""
    if obj is None:
        st.error(f"⚠️ Không tìm thấy **{what}**. {hint}")
        st.stop()


def churn_pct(df: pd.DataFrame, by: str, churn_col: str = "churn_label") -> pd.Series:
    """Tỷ lệ churn (%) theo một biến nhóm, sắp xếp tăng dần."""
    return df.groupby(by)[churn_col].mean().mul(100).sort_values()
