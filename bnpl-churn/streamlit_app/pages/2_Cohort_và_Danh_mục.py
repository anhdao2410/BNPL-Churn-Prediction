# -*- coding: utf-8 -*-
"""Trang 2 — Cohort MoM Retention và Retention theo danh mục thanh toán."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_utils import ACCENT, ACCENT_2, MUTED, apply_chart_style, section, load_report, require  # noqa: E402

st.set_page_config(page_title="BNPL Churn — Cohort & Danh mục", layout="wide")
apply_chart_style()

st.title("Cohort Retention và Retention theo danh mục")
st.caption("Cohort tính theo tháng có giao dịch đầu tiên, chỉ gồm khách kích hoạt từ 01/2024 "
           "để tránh cắt cụt dữ liệu bên trái.")

cohort_matrix = load_report("cohort_retention_matrix.csv")
churn_by_cat = load_report("churn_by_category.csv")
retention_curve_cat = load_report("retention_curve_by_category.csv")
lifecycle_summary = load_report("lifecycle_segment_summary.csv")

require(cohort_matrix, "reports/cohort_retention_matrix.csv",
        "Chạy notebook 03_cohort_analysis.ipynb để sinh các báo cáo này.")

# ---------- 1. Cohort ----------
mat = cohort_matrix.apply(pd.to_numeric, errors="coerce")
mean_curve = mat.mean(axis=0, skipna=True)

def _get(k):
    return mean_curve.get(str(k), mean_curve.get(k, float("nan")))

c1, c2, c3 = st.columns(3)
c1.metric("Retention trung bình — tháng 1", f"{_get(1):.1f}%")
c2.metric("Tháng 3", f"{_get(3):.1f}%")
c3.metric("Tháng 6", f"{_get(6):.1f}%")
st.caption("Sụt giảm lớn nhất xảy ra ngay tháng đầu tiên; sau đó đường cong đi ngang — "
           "giữ chân hiệu quả nhất trong 30 ngày đầu sau giao dịch đầu tiên.")

section("Ma trận Cohort MoM Retention (%)")
fig, ax = plt.subplots(figsize=(11, 5.2))
sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", vmin=40, vmax=85,
            cbar_kws={"label": "Retention (%)", "shrink": 0.8},
            linewidths=0.6, linecolor="white", ax=ax, annot_kws={"size": 7.5})
ax.set_xlabel("Tháng thứ n kể từ giao dịch đầu tiên")
ax.set_ylabel("Cohort")
ax.grid(False)
st.pyplot(fig); plt.close(fig)

# ---------- 2. Danh mục ----------
section("Retention và churn theo danh mục thanh toán",
        "Danh mục thiết yếu, định kỳ giữ chân tốt nhất; danh mục mua theo dịp rời bỏ cao hơn nhiều lần.")

col1, col2 = st.columns([3, 2])
if churn_by_cat is not None:
    cbc = churn_by_cat.copy()
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [ACCENT_2 if v > cbc["churn_rate"].mean() else ACCENT for v in cbc["churn_rate"]]
        ax.barh(cbc.index, cbc["churn_rate"], color=colors, height=0.6)
        for i, (v, n) in enumerate(zip(cbc["churn_rate"], cbc["n"])):
            ax.text(v + 0.5, i, f"{v:.1f}%  ·  n={int(n):,}", va="center", fontsize=8.5)
        ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
        ax.set_xlim(0, cbc["churn_rate"].max() * 1.35)
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.dataframe(cbc.rename(columns={"churn_rate": "Churn (%)", "n": "Số khách"}),
                     width='stretch')

if retention_curve_cat is not None:
    rc = retention_curve_cat.apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for cat in rc.index:
        ax.plot(rc.columns.astype(str), rc.loc[cat], marker="o", ms=3, lw=1.5, label=cat)
    ax.set_title("Đường cong retention trung bình theo danh mục")
    ax.set_xlabel("Tháng thứ n"); ax.set_ylabel("Retention (%)")
    ax.legend(ncol=2, loc="lower left")
    st.pyplot(fig); plt.close(fig)

# ---------- 3. Vòng đời ----------
section("Phân khúc vòng đời khách hàng (FPU / RPU)",
        "No_Transaction là khách đã được cấp hạn mức nhưng chưa từng giao dịch — thất bại ở khâu kích hoạt, "
        "khác bản chất với churn hành vi.")
if lifecycle_summary is not None:
    ls = lifecycle_summary.copy()
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(ls.rename(columns={"so_khach": "Số khách", "ty_trong_%": "Tỷ trọng (%)",
                                        "churn_rate_%": "Churn (%)"}), width='stretch')
    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        colors = [MUTED] + [ACCENT_2, ACCENT_2, ACCENT, ACCENT][:len(ls) - 1]
        ax.bar(ls.index, ls["churn_rate_%"], color=colors[:len(ls)], width=0.6)
        for i, v in enumerate(ls["churn_rate_%"]):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5)
        ax.set_ylabel("Churn (%)"); ax.set_xlabel("")
        plt.xticks(rotation=15)
        st.pyplot(fig); plt.close(fig)
