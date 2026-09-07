# -*- coding: utf-8 -*-
"""
Trang 1 — 📊 Dashboard Tổng quan
Web app dự báo churn khách hàng BNPL — Chuyên đề tốt nghiệp IE400 (Lan & Đào)
Chạy:  streamlit run streamlit_app/app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import PALETTE, load_artifact, load_customer_features, require

st.set_page_config(page_title="BNPL Churn — Dashboard", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 Dashboard Tổng quan — Khách hàng BNPL")
st.caption("Dữ liệu: log giao dịch BNPL tổng hợp (nhóm tự xây dựng, xem mục 3.1.1 báo cáo) · "
           "Nhãn churn: không giao dịch ≥ 60 ngày (mục 3.1.2).")

feat = load_customer_features()
require(feat, "dữ liệu khách hàng (data/bnpl_customer_features.csv)",
        "Hãy đặt file này vào thư mục data/ trước khi chạy app.")

art = load_artifact()
if art:
    m = art["metrics_test"]
    st.success(f"🤖 Model đang dùng: **{art['best_name']}** — AUC test **{art.get('metrics_test',{}).get('AUC', 0):.3f}**, "
               f"Recall (churn) **{m.get('Recall', 0):.3f}** — huấn luyện trên đặc trưng tính tại mốc "
               f"`T_ref = {art.get('T_ref','?')}` (không dùng bảng snapshot rò rỉ nhãn, xem mục 3.2.2)")

active = feat[feat["lifecycle_segment"] != "No_Transaction"].copy()
no_trx = feat[feat["lifecycle_segment"] == "No_Transaction"]

# ---------------- Sidebar: bộ lọc ----------------
st.sidebar.header("🔎 Bộ lọc (áp dụng cho khách đã có giao dịch)")
cat_opts = sorted(active["top_category"].dropna().unique().tolist())
seg_opts = sorted(active["lifecycle_segment"].dropna().unique().tolist())
gender_opts = sorted(active["gender"].dropna().unique().tolist())
tier_opts = sorted(active["city_tier"].dropna().unique().tolist())

f_cat = st.sidebar.multiselect("Danh mục thanh toán chính", cat_opts, default=cat_opts)
f_seg = st.sidebar.multiselect("Phân khúc vòng đời (Lifecycle)", seg_opts, default=seg_opts)
f_gender = st.sidebar.multiselect("Giới tính", gender_opts, default=gender_opts)
f_tier = st.sidebar.multiselect("Hạng thành phố (City Tier)", tier_opts, default=tier_opts)

mask = (active["top_category"].isin(f_cat) & active["lifecycle_segment"].isin(f_seg)
        & active["gender"].isin(f_gender) & active["city_tier"].isin(f_tier))
d = active[mask]
if d.empty:
    st.warning("Bộ lọc hiện tại không còn khách hàng nào — hãy nới lỏng bộ lọc.")
    st.stop()

# ---------------- KPI ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng khách hàng (toàn bộ)", f"{len(feat):,}")
c2.metric("Khách đã có giao dịch", f"{len(active):,}", f"{len(active)/len(feat):.1%} tổng số")
c3.metric("Churn rate (trong bộ lọc)", f"{d['churn_label'].mean():.1%}",
          delta=f"{(d['churn_label'].mean() - active['churn_label'].mean())*100:+.1f} điểm % so với toàn bộ KH active",
          delta_color="inverse")
c4.metric("Khách chưa từng giao dịch", f"{len(no_trx):,}", f"{len(no_trx)/len(feat):.1%} tổng số",
          delta_color="off")

st.divider()

# ---------------- Hàng 1: churn theo lifecycle & category ----------------
col1, col2 = st.columns(2)
overall = active["churn_label"].mean() * 100

with col1:
    st.subheader("Churn theo phân khúc vòng đời (FPU/RPU)")
    order = ["FPU", "RPU_Early", "RPU_Mid", "RPU_Loyal"]
    rate = d.groupby("lifecycle_segment")["churn_label"].mean().reindex(
        [s for s in order if s in d["lifecycle_segment"].unique()]).mul(100)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = [PALETTE["red"], PALETTE["orange"], PALETTE["blue"], PALETTE["green"]][:len(rate)]
    rate.plot(kind="bar", ax=ax, color=colors)
    ax.axhline(overall, color="crimson", ls="--", lw=1.2, label=f"TB active ({overall:.1f}%)")
    for i, v in enumerate(rate):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Churn (%)"); ax.set_xlabel(""); ax.legend(fontsize=8)
    plt.xticks(rotation=0)
    st.pyplot(fig); plt.close(fig)

with col2:
    st.subheader("Churn theo danh mục thanh toán")
    rate = d.groupby("top_category")["churn_label"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    rate.plot(kind="barh", ax=ax, color=PALETTE["blue"])
    ax.axvline(overall, color="crimson", ls="--", lw=1.2)
    for i, v in enumerate(rate):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Churn (%)"); ax.set_ylabel("")
    st.pyplot(fig); plt.close(fig)

# ---------------- Hàng 2: recency & tenure ----------------
col3, col4 = st.columns(2)
with col3:
    st.subheader("Phân phối Recency (ngày từ giao dịch gần nhất)")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.histplot(data=d, x="recency_days", hue="churn_label", bins=30, multiple="stack",
                 palette={0: PALETTE["green"], 1: PALETTE["red"]}, ax=ax)
    ax.set_xlabel("Recency (ngày)")
    st.pyplot(fig); plt.close(fig)

with col4:
    st.subheader("Churn theo mức độ sử dụng khuyến mãi")
    d2 = d.copy()
    d2["promo_bucket"] = pd.cut(d2["promo_rate"], [-0.01, 0.01, 0.3, 0.6, 1.01],
                                labels=["Không dùng", "Thấp", "Trung bình", "Cao"])
    rate = d2.groupby("promo_bucket", observed=True)["churn_label"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rate.plot(kind="bar", ax=ax, color=PALETTE["purple"])
    ax.set_ylabel("Churn (%)"); ax.set_xlabel("Tỷ lệ giao dịch có khuyến mãi")
    plt.xticks(rotation=0)
    st.pyplot(fig); plt.close(fig)

st.info("💡 Ba tín hiệu mạnh nhất trên dữ liệu: **phân khúc vòng đời** (FPU churn cao gấp ~100 lần RPU_Loyal), "
        "**danh mục thanh toán** (Điện tử/Du lịch/Giải trí cao hơn hẳn Viễn thông/Điện & Nước), và **recency** — "
        "khách càng lâu không giao dịch càng dễ churn. Chi tiết ở Trang 2.")

with st.expander("🔍 Xem dữ liệu chi tiết"):
    st.dataframe(d.head(200))
    st.download_button("⬇️ Tải dữ liệu đã lọc (CSV)",
                       d.to_csv(index=False).encode("utf-8-sig"),
                       file_name="bnpl_filtered.csv", mime="text/csv")
