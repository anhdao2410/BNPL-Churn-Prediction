# -*- coding: utf-8 -*-
"""Trang 2 — 📈 Cohort MoM Retention & Category Retention (insight signature của đề tài)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import PALETTE, load_report, load_customer_features, require  # noqa: E402

st.set_page_config(page_title="BNPL Churn — Cohort & Category", page_icon="📈", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📈 Cohort MoM Retention & Retention theo Danh mục thanh toán")

cohort_matrix = load_report("cohort_retention_matrix.csv")
churn_by_cat = load_report("churn_by_category.csv")
retention_curve_cat = load_report("retention_curve_by_category.csv")
lifecycle_summary = load_report("lifecycle_segment_summary.csv")
feat = load_customer_features()

require(cohort_matrix, "reports/cohort_retention_matrix.csv",
        "Hãy chạy notebook 03_cohort_analysis.ipynb (T13) trước để sinh các báo cáo này.")

# ---------------- 1. Cohort MoM Retention Heatmap ----------------
st.subheader("1️⃣ Ma trận Cohort MoM Retention")
st.caption("Mỗi hàng là một cohort (tháng giao dịch đầu tiên); mỗi cột là tháng thứ n kể từ giao dịch đầu tiên. "
           "Chỉ gồm khách kích hoạt từ 01/2024 để tránh hiện tượng cắt cụt dữ liệu (left-truncation) — xem mục 3.3.3.")

mat = cohort_matrix.apply(pd.to_numeric, errors="coerce")
fig, ax = plt.subplots(figsize=(11, 5.6))
sns.heatmap(mat, annot=True, fmt=".0f", cmap="YlGnBu", vmin=40, vmax=85,
            cbar_kws={"label": "Retention (%)"}, linewidths=0.4, ax=ax, annot_kws={"size": 7.5})
ax.set_xlabel("Tháng thứ n kể từ giao dịch đầu tiên")
ax.set_ylabel("Cohort (tháng giao dịch đầu tiên)")
st.pyplot(fig); plt.close(fig)

mean_curve = mat.mean(axis=0, skipna=True)
m1 = mean_curve.get("1", mean_curve.get(1, float("nan")))
m3 = mean_curve.get("3", mean_curve.get(3, float("nan")))
m6 = mean_curve.get("6", mean_curve.get(6, float("nan")))
c1, c2, c3 = st.columns(3)
c1.metric("Retention TB — tháng 1", f"{m1:.1f}%" if pd.notna(m1) else "—")
c2.metric("Retention TB — tháng 3", f"{m3:.1f}%" if pd.notna(m3) else "—")
c3.metric("Retention TB — tháng 6", f"{m6:.1f}%" if pd.notna(m6) else "—")
st.info("🎯 **Cửa sổ vàng:** sụt giảm lớn nhất diễn ra ngay ở tháng đầu tiên, sau đó đường cong đi ngang ổn định — "
        "ngân sách giữ chân nên tập trung vào 30 ngày đầu sau giao dịch đầu tiên của khách hàng.")

st.divider()

# ---------------- 2. Retention theo danh mục thanh toán ----------------
st.subheader("2️⃣ Retention & Churn theo Danh mục thanh toán ⭐ (insight signature)")

if churn_by_cat is not None:
    cbc = churn_by_cat.copy()
    cbc.index.name = "Danh mục"
    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        colors = plt.cm.RdYlGn_r(cbc["churn_rate"] / cbc["churn_rate"].max())
        ax.barh(cbc.index, cbc["churn_rate"], color=colors)
        for i, (v, n) in enumerate(zip(cbc["churn_rate"], cbc["n"])):
            ax.text(v + 0.5, i, f"{v:.1f}%  (n={int(n):,})", va="center", fontsize=8.5)
        ax.set_xlabel("Churn rate (%)")
        ax.set_title("Tỷ lệ rời bỏ theo danh mục thanh toán")
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.dataframe(cbc.rename(columns={"churn_rate": "Churn (%)", "n": "Số KH"}))
        st.caption("Danh mục thiết yếu, định kỳ (Viễn thông, Điện & Nước) giữ chân tốt nhất; "
                   "danh mục mua theo dịp (Điện tử, Du lịch, Giải trí) rời bỏ cao gấp nhiều lần.")

if retention_curve_cat is not None:
    st.markdown("**Đường cong Retention trung bình theo danh mục** (theo tháng kể từ giao dịch đầu tiên)")
    rc = retention_curve_cat.apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for cat in rc.index:
        ax.plot(rc.columns.astype(str), rc.loc[cat], marker="o", ms=3, lw=1.6, label=cat)
    ax.set_xlabel("Tháng thứ n"); ax.set_ylabel("Retention (%)")
    ax.legend(fontsize=8, ncol=2)
    st.pyplot(fig); plt.close(fig)

st.divider()

# ---------------- 3. FPU / RPU lifecycle summary ----------------
st.subheader("3️⃣ Phân khúc vòng đời khách hàng (FPU/RPU)")
if lifecycle_summary is not None:
    ls = lifecycle_summary.copy()
    ls.index.name = "Phân khúc"
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(ls)
    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        colors = [PALETTE["gray"], PALETTE["red"], PALETTE["orange"], PALETTE["blue"], PALETTE["green"]]
        ax.bar(ls.index, ls["churn_rate_%"], color=colors[:len(ls)])
        for i, v in enumerate(ls["churn_rate_%"]):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Churn rate (%)")
        plt.xticks(rotation=20)
        st.pyplot(fig); plt.close(fig)
    st.caption("No_Transaction: khách đã được cấp hạn mức nhưng chưa từng giao dịch — thất bại ở khâu kích hoạt, "
               "khác bản chất với churn hành vi (xem mục 3.3.2).")
else:
    st.warning("Chưa có `reports/lifecycle_segment_summary.csv` — chạy notebook 03 (T13) để sinh file này.")
