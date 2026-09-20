import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from app_utils import (
    ACCENT,
    CHURN_COLORS,
    MUTED,
    apply_chart_style,
    load_artifact,
    load_customer_features,
    require,
    section,
)

st.set_page_config(page_title="BNPL Churn — Tổng quan", layout="wide")
apply_chart_style()

st.title("Tổng quan khách hàng BNPL")
st.caption(
    "Dữ liệu giao dịch BNPL tổng hợp · Nhãn churn: không giao dịch trong 60 ngày trở lên."
)


feat = load_customer_features()
require(
    feat,
    "dữ liệu khách hàng (data/bnpl_customer_features.csv)",
    "Đặt file này vào thư mục data/ trước khi chạy.",
)


def _first_present(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


LABEL_ALIASES = [
    "churn_label",
    "churn_60d",
    "churn",
    "is_churn",
    "churned",
    "label",
    "target",
]
label_col = _first_present(feat, LABEL_ALIASES)
if label_col is None:
    st.error(
        "File đặc trưng không có cột nhãn churn (tìm theo: "
        + ", ".join(LABEL_ALIASES)
        + ")."
    )
    st.caption("Các cột hiện có trong file: " + ", ".join(map(str, feat.columns)))
    st.stop()
if label_col != "churn_label":
    feat = feat.rename(columns={label_col: "churn_label"})

feat["churn_label"] = pd.to_numeric(feat["churn_label"], errors="coerce")
if feat["churn_label"].isna().all():
    st.error(f"Cột nhãn `{label_col}` không ở dạng số 0/1.")
    st.stop()

REQUIRED_COLS = ["top_category", "city_tier", "recency_days", "promo_rate"]
missing = [c for c in REQUIRED_COLS if c not in feat.columns]
if missing:
    st.error("File đặc trưng thiếu các cột bắt buộc: " + ", ".join(missing))
    st.caption("Các cột hiện có trong file: " + ", ".join(map(str, feat.columns)))
    st.stop()


SEG_ORDER = ["FPU", "RPU_Early", "RPU_Mid", "RPU_Loyal"]
NO_TRX = "No_Transaction"

SEG_ALIASES = [
    "lifecycle_segment",
    "lifecycle_seg",
    "lifecycle",
    "customer_segment",
    "segment",
]
TXN_COUNT_ALIASES = [
    "freq_total",
    "total_txn",
    "txn_count",
    "n_txn",
    "num_txn",
    "total_transactions",
    "n_transactions",
    "transaction_count",
    "total_orders",
    "order_count",
    "frequency",
]

seg_col = _first_present(feat, SEG_ALIASES)
cnt_col = _first_present(feat, TXN_COUNT_ALIASES)
seg_note = ""

if seg_col is not None:
    if seg_col != "lifecycle_segment":
        feat = feat.rename(columns={seg_col: "lifecycle_segment"})
    HAS_SEG = True
elif cnt_col is not None:
    n_txn = pd.to_numeric(feat[cnt_col], errors="coerce").fillna(0)
    feat["lifecycle_segment"] = pd.cut(
        n_txn,
        bins=[-0.5, 0.5, 1.5, 4.5, 9.5, float("inf")],
        labels=[NO_TRX, "FPU", "RPU_Early", "RPU_Mid", "RPU_Loyal"],
    ).astype(str)
    HAS_SEG = True
else:
    HAS_SEG = False


if HAS_SEG:
    active = feat[feat["lifecycle_segment"] != NO_TRX].copy()
    no_trx = feat[feat["lifecycle_segment"] == NO_TRX]
elif cnt_col is not None:
    n_txn = pd.to_numeric(feat[cnt_col], errors="coerce").fillna(0)
    active = feat[n_txn > 0].copy()
    no_trx = feat[n_txn <= 0]
else:
    active = feat.copy()
    no_trx = feat.iloc[0:0]

active = active[active["churn_label"].notna()]
if active.empty:
    st.error("Không có khách hàng nào có giao dịch trong bộ dữ liệu.")
    st.stop()

# ---------- Sidebar: bộ lọc ----------
st.sidebar.markdown("**Bộ lọc**")
st.sidebar.caption("Áp dụng cho khách đã có giao dịch.")
cat_opts = sorted(active["top_category"].dropna().unique())
tier_opts = sorted(active["city_tier"].dropna().unique())

f_cat = st.sidebar.multiselect("Danh mục thanh toán chính", cat_opts, default=cat_opts)
f_tier = st.sidebar.multiselect("Hạng thành phố", tier_opts, default=tier_opts)

mask = active["top_category"].isin(f_cat) & active["city_tier"].isin(f_tier)

if HAS_SEG:
    present_segs = set(active["lifecycle_segment"].dropna().unique())
    seg_opts = [s for s in SEG_ORDER if s in present_segs]
    seg_opts += sorted(present_segs - set(SEG_ORDER))
    f_seg = st.sidebar.multiselect("Phân khúc vòng đời", seg_opts, default=seg_opts)
    mask &= active["lifecycle_segment"].isin(f_seg)

d = active[mask]
if d.empty:
    st.warning("Bộ lọc hiện tại không còn khách hàng nào.")
    st.stop()

# ---------- KPI ----------
overall = active["churn_label"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng khách hàng", f"{len(feat):,}")
c2.metric(
    "Đã có giao dịch",
    f"{len(active):,}",
    f"{len(active) / len(feat):.1%} tổng số",
    delta_color="off",
)
c3.metric(
    "Churn rate (bộ lọc)",
    f"{d['churn_label'].mean():.1%}",
    f"{(d['churn_label'].mean() - overall) * 100:+.1f} điểm so với toàn bộ",
    delta_color="inverse",
)
c4.metric(
    "Chưa từng giao dịch",
    f"{len(no_trx):,}",
    f"{len(no_trx) / len(feat):.1%} tổng số",
    delta_color="off",
)

art = load_artifact()
if art:
    m = art.get("metrics_test", {})
    st.caption(
        f"Mô hình đang dùng: {art['best_name']} · AUC {m.get('AUC', 0):.3f} · "
        f"Recall {m.get('Recall', 0):.3f} · đặc trưng tính tại mốc T_ref = {art.get('T_ref', '—')}."
    )

if seg_note:
    st.caption(seg_note)

# ---------- Churn theo vòng đời & danh mục ----------
section("Churn theo phân khúc vòng đời và danh mục thanh toán")

if HAS_SEG:
    col1, col2 = st.columns(2)
else:
    col1, col2 = None, st.container()

if col1 is not None:
    with col1:
        present = d["lifecycle_segment"].dropna().unique()
        order = [s for s in SEG_ORDER if s in present]
        order += sorted(set(present) - set(SEG_ORDER))
        rate = (
            d.groupby("lifecycle_segment")["churn_label"].mean().reindex(order).mul(100)
        )
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(rate.index, rate.values, color=ACCENT, width=0.6)
        ax.axhline(overall * 100, color=MUTED, ls="--", lw=1)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5)
        ax.set_title("Theo phân khúc vòng đời")
        ax.set_ylabel("Churn (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, max(rate.max() * 1.15, 10))
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info(
        "Bộ đặc trưng không có cột phân khúc vòng đời — biểu đồ theo vòng đời đã được bỏ qua."
    )

with col2:
    rate = d.groupby("top_category")["churn_label"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(rate.index, rate.values, color=ACCENT, height=0.6)
    ax.axvline(overall * 100, color=MUTED, ls="--", lw=1)
    for i, v in enumerate(rate.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_title("Theo danh mục thanh toán")
    ax.set_xlabel("Churn (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, rate.max() * 1.2)
    st.pyplot(fig)
    plt.close(fig)


section("Recency và mức độ dùng khuyến mãi")
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.histplot(
        data=d,
        x="recency_days",
        hue="churn_label",
        bins=30,
        multiple="stack",
        palette=CHURN_COLORS,
        ax=ax,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("Số ngày từ giao dịch gần nhất")
    ax.set_xlabel("Recency (ngày)")
    ax.set_ylabel("Số khách")
    leg = ax.get_legend()
    if leg:
        leg.set_title("")
        for t, lab in zip(leg.texts, ["Ở lại", "Churn"]):
            t.set_text(lab)
    st.pyplot(fig)
    plt.close(fig)

with col4:
    d2 = d.assign(
        promo_bucket=pd.cut(
            d["promo_rate"],
            [-0.01, 0.01, 0.3, 0.6, 1.01],
            labels=["Không dùng", "Thấp", "Trung bình", "Cao"],
        )
    )
    rate = d2.groupby("promo_bucket", observed=True)["churn_label"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(rate.index.astype(str), rate.values, color=ACCENT, width=0.6)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_title("Churn theo tỷ lệ giao dịch có khuyến mãi")
    ax.set_ylabel("Churn (%)")
    ax.set_xlabel("")
    st.pyplot(fig)
    plt.close(fig)

# ---------- Dữ liệu ----------
with st.expander("Xem dữ liệu chi tiết"):
    st.dataframe(d.head(200), width="stretch")
    st.download_button(
        "Tải dữ liệu đã lọc (CSV)",
        d.to_csv(index=False).encode("utf-8-sig"),
        file_name="bnpl_filtered.csv",
        mime="text/csv",
    )
