# -*- coding: utf-8 -*-
"""Trang 1 — Tổng quan khách hàng BNPL. Chạy: streamlit run streamlit_app/app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import (ACCENT, ACCENT_2, MUTED, CHURN_COLORS, apply_chart_style, section,
                   load_artifact, load_customer_features, require)

st.set_page_config(page_title="BNPL Churn — Tổng quan", layout="wide")
apply_chart_style()

st.title("Tổng quan khách hàng BNPL")
st.caption("Dữ liệu giao dịch BNPL tổng hợp · Nhãn churn: không giao dịch trong 60 ngày trở lên.")

feat = load_customer_features()
require(feat, "dữ liệu khách hàng (data/bnpl_customer_features.csv)",
        "Đặt file này vào thư mục data/ trước khi chạy.")

active = feat[feat["lifecycle_segment"] != "No_Transaction"].copy()
no_trx = feat[feat["lifecycle_segment"] == "No_Transaction"]

# ---------- Sidebar: bộ lọc ----------
st.sidebar.markdown("**Bộ lọc**")
st.sidebar.caption("Áp dụng cho khách đã có giao dịch.")
cat_opts = sorted(active["top_category"].dropna().unique())
seg_opts = ["FPU", "RPU_Early", "RPU_Mid", "RPU_Loyal"]
tier_opts = sorted(active["city_tier"].dropna().unique())

f_cat = st.sidebar.multiselect("Danh mục thanh toán chính", cat_opts, default=cat_opts)
f_seg = st.sidebar.multiselect("Phân khúc vòng đời", seg_opts, default=seg_opts)
f_tier = st.sidebar.multiselect("Hạng thành phố", tier_opts, default=tier_opts)

d = active[active["top_category"].isin(f_cat)
           & active["lifecycle_segment"].isin(f_seg)
           & active["city_tier"].isin(f_tier)]
if d.empty:
    st.warning("Bộ lọc hiện tại không còn khách hàng nào.")
    st.stop()

# ---------- KPI ----------
overall = active["churn_label"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng khách hàng", f"{len(feat):,}")
c2.metric("Đã có giao dịch", f"{len(active):,}", f"{len(active)/len(feat):.1%} tổng số", delta_color="off")
c3.metric("Churn rate (bộ lọc)", f"{d['churn_label'].mean():.1%}",
          f"{(d['churn_label'].mean()-overall)*100:+.1f} điểm so với toàn bộ", delta_color="inverse")
c4.metric("Chưa từng giao dịch", f"{len(no_trx):,}", f"{len(no_trx)/len(feat):.1%} tổng số", delta_color="off")

art = load_artifact()
if art:
    m = art.get("metrics_test", {})
    st.caption(f"Mô hình đang dùng: {art['best_name']} · AUC {m.get('AUC', 0):.3f} · "
               f"Recall {m.get('Recall', 0):.3f} · đặc trưng tính tại mốc T_ref = {art.get('T_ref', '—')}.")

# ---------- Churn theo vòng đời & danh mục ----------
section("Churn theo phân khúc vòng đời và danh mục thanh toán")
col1, col2 = st.columns(2)

with col1:
    order = [s for s in seg_opts if s in d["lifecycle_segment"].unique()]
    rate = d.groupby("lifecycle_segment")["churn_label"].mean().reindex(order).mul(100)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(rate.index, rate.values, color=ACCENT, width=0.6)
    ax.axhline(overall * 100, color=MUTED, ls="--", lw=1)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_title("Theo phân khúc vòng đời")
    ax.set_ylabel("Churn (%)"); ax.set_xlabel("")
    ax.set_ylim(0, max(rate.max() * 1.15, 10))
    st.pyplot(fig); plt.close(fig)

with col2:
    rate = d.groupby("top_category")["churn_label"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(rate.index, rate.values, color=ACCENT, height=0.6)
    ax.axvline(overall * 100, color=MUTED, ls="--", lw=1)
    for i, v in enumerate(rate.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_title("Theo danh mục thanh toán")
    ax.set_xlabel("Churn (%)"); ax.set_ylabel("")
    ax.set_xlim(0, rate.max() * 1.2)
    st.pyplot(fig); plt.close(fig)

# ---------- Recency & khuyến mãi ----------
section("Recency và mức độ dùng khuyến mãi")
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.histplot(data=d, x="recency_days", hue="churn_label", bins=30, multiple="stack",
                 palette=CHURN_COLORS, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("Số ngày từ giao dịch gần nhất")
    ax.set_xlabel("Recency (ngày)"); ax.set_ylabel("Số khách")
    leg = ax.get_legend()
    if leg:
        leg.set_title(""); 
        for t, lab in zip(leg.texts, ["Ở lại", "Churn"]):
            t.set_text(lab)
    st.pyplot(fig); plt.close(fig)

with col4:
    d2 = d.assign(promo_bucket=pd.cut(d["promo_rate"], [-0.01, 0.01, 0.3, 0.6, 1.01],
                                      labels=["Không dùng", "Thấp", "Trung bình", "Cao"]))
    rate = d2.groupby("promo_bucket", observed=True)["churn_label"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(rate.index.astype(str), rate.values, color=ACCENT, width=0.6)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_title("Churn theo tỷ lệ giao dịch có khuyến mãi")
    ax.set_ylabel("Churn (%)"); ax.set_xlabel("")
    st.pyplot(fig); plt.close(fig)

# ---------- Dữ liệu ----------
with st.expander("Xem dữ liệu chi tiết"):
    st.dataframe(d.head(200), width='stretch')
    st.download_button("Tải dữ liệu đã lọc (CSV)", d.to_csv(index=False).encode("utf-8-sig"),
                       file_name="bnpl_filtered.csv", mime="text/csv")
