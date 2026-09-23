from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Cấu hình hiển thị và đường dẫn thư mục
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="DejaVu Sans")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Đọc dữ liệu
tx = pd.read_csv(
    DATA_DIR / "bnpl_transactions_clean.csv", parse_dates=["transaction_date"]
)
cust = pd.read_csv(
    DATA_DIR / "bnpl_customers_clean.csv", parse_dates=["activation_date"]
)
feat = pd.read_csv(
    DATA_DIR / "bnpl_customer_features.csv",
    parse_dates=["activation_date", "first_trx_date", "last_trx_date"],
)

OBS_DATE = pd.Timestamp("2024-12-30")
CHURN_COL = "churn_60d"

print(
    f"Giao dịch : {len(tx):,} | {tx.transaction_date.min():%d/%m/%Y} -> {tx.transaction_date.max():%d/%m/%Y}"
)
print(
    f"Khách hàng: {len(cust):,} (đăng ký) | {tx.customer_id.nunique():,} có ít nhất 1 giao dịch"
)

# 2. Xử lý cắt cụt bên trái (Left Truncation) cho Cohort
tx["txn_month"] = tx.transaction_date.dt.to_period("M")
first_txn = tx.groupby("customer_id")["transaction_date"].min()

has_txn = feat[feat.lifecycle_segment != "No_Transaction"].copy()
truncated = has_txn[has_txn.activation_date < "2024-01-01"]
cohort_pool = has_txn[has_txn.activation_date >= "2024-01-01"]

print(
    f"Khách kích hoạt trước 2024 (loại khỏi cohort): {len(truncated):,} ({len(truncated) / len(has_txn):.1%})"
)
print(
    f"Khách kích hoạt trong 2024 (dùng cho cohort): {len(cohort_pool):,} ({len(cohort_pool) / len(has_txn):.1%})"
)

tx_c = tx[tx.customer_id.isin(cohort_pool.customer_id)].copy()
tx_c["cohort_month"] = tx_c.customer_id.map(first_txn.dt.to_period("M"))
tx_c["period_index"] = (tx_c.txn_month - tx_c.cohort_month).apply(lambda p: p.n)

cohort_sizes = (
    tx_c.groupby("cohort_month")["customer_id"].nunique().rename("Quy mô cohort")
)

# 3. Ma trận Cohort MoM Retention (Heatmap)
active = (
    tx_c.groupby(["cohort_month", "period_index"])["customer_id"]
    .nunique()
    .reset_index(name="n_active")
)
retention = active.pivot(
    index="cohort_month", columns="period_index", values="n_active"
)
retention_pct = retention.divide(cohort_sizes, axis=0) * 100

fig, ax = plt.subplots(figsize=(12.5, 6.2))
sns.heatmap(
    retention_pct,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    vmin=40,
    vmax=85,
    cbar_kws={"label": "Retention (%)"},
    annot_kws={"size": 8},
    linewidths=0.4,
    ax=ax,
)
ax.set_title(
    "Ma trận Cohort MoM Retention — % khách còn giao dịch ở tháng thứ n",
    fontsize=13,
)
ax.set_xlabel("Tháng thứ n kể từ giao dịch đầu tiên")
ax.set_ylabel("Cohort (tháng giao dịch đầu tiên)")
plt.tight_layout()
plt.savefig(FIG_DIR / "cohort_retention_heatmap.png", bbox_inches="tight")
plt.close(fig)

# 4. Đường cong Retention trung bình (Mature Cohorts >= 6 tháng)
MATURE_MIN_PERIODS = 6
mature = retention_pct[retention_pct.notna().sum(axis=1) >= MATURE_MIN_PERIODS]
mean_curve = mature.mean(axis=0)

fig, ax = plt.subplots(figsize=(10.5, 5.5))
for c in mature.index:
    ax.plot(
        mature.columns,
        mature.loc[c],
        alpha=0.45,
        lw=1.3,
        marker="o",
        ms=3,
        label=str(c),
    )
ax.plot(
    mean_curve.index,
    mean_curve.values,
    color="black",
    lw=3,
    label="Trung bình các cohort",
)
ax.set_ylim(0, 105)
ax.set_xlabel("Tháng thứ n kể từ giao dịch đầu tiên")
ax.set_ylabel("Retention (%)")
ax.set_title(
    f"Đường cong Retention — {len(mature)} cohort có >= {MATURE_MIN_PERIODS} tháng quan sát"
)
ax.legend(ncol=2, fontsize=8.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "cohort_retention_curves.png", bbox_inches="tight")
plt.close(fig)

# 5. Phân tích FPU vs RPU & Thời gian đến giao dịch thứ 2
seg_order = ["No_Transaction", "FPU", "RPU_Early", "RPU_Mid", "RPU_Loyal"]
seg = (
    feat.groupby("lifecycle_segment")
    .agg(so_khach=("customer_id", "size"), churn_rate=(CHURN_COL, "mean"))
    .reindex(seg_order)
)
seg["ty_trong_%"] = (seg.so_khach / seg.so_khach.sum() * 100).round(1)
seg["churn_rate_%"] = (seg.churn_rate * 100).round(1)
seg = seg[["so_khach", "ty_trong_%", "churn_rate_%"]]

fpu_n = int(seg.loc["FPU", "so_khach"])
rpu_n = int(seg.loc[["RPU_Early", "RPU_Mid", "RPU_Loyal"], "so_khach"].sum())

first_two = (
    tx.sort_values(["customer_id", "transaction_date"])
    .groupby("customer_id")
    .head(2)
    .groupby("customer_id")["transaction_date"]
    .agg(["first", "last"])
)
first_two = first_two[first_two["first"] != first_two["last"]]
gap = (first_two["last"] - first_two["first"]).dt.days

# 6. Retention theo danh mục thanh toán
cat_of = feat.set_index("customer_id")["top_category"]
tx_c["cat"] = tx_c.customer_id.map(cat_of)

rows = []
for cat, grp in tx_c.groupby("cat"):
    size = grp.groupby("cohort_month")["customer_id"].nunique()
    act_ = (
        grp.groupby(["cohort_month", "period_index"])["customer_id"].nunique().unstack()
    )
    pct = act_.divide(size, axis=0) * 100
    pct = pct[pct.notna().sum(axis=1) >= MATURE_MIN_PERIODS]
    if len(pct):
        rows.append(pct.mean(axis=0).rename(cat))
cat_curves = pd.DataFrame(rows)

churn_by_cat = (
    feat.dropna(subset=["top_category"])
    .groupby("top_category")[CHURN_COL]
    .agg(churn_rate=lambda s: s.mean() * 100, n="size")
    .sort_values("churn_rate")
)

# 7. Xuất file báo cáo phục vụ Streamlit
retention_pct.round(1).to_csv(
    REPORTS_DIR / "cohort_retention_matrix.csv", encoding="utf-8-sig"
)
seg.to_csv(REPORTS_DIR / "lifecycle_segment_summary.csv", encoding="utf-8-sig")
churn_by_cat.round(2).to_csv(
    REPORTS_DIR / "churn_by_category.csv", encoding="utf-8-sig"
)
cat_curves.round(1).to_csv(
    REPORTS_DIR / "retention_curve_by_category.csv", encoding="utf-8-sig"
)

summary = {
    "So khach hang": len(feat),
    "So khach co giao dich": len(has_txn),
    "So khach chua tung giao dich": int(
        (feat.lifecycle_segment == "No_Transaction").sum()
    ),
    "Churn rate toan bo (60d)": round(feat[CHURN_COL].mean() * 100, 2),
    "Churn rate tren KH co giao dich": round(has_txn[CHURN_COL].mean() * 100, 2),
    "Retention TB thang 1 (%)": round(float(mean_curve.get(1, np.nan)), 1),
    "Retention TB thang 3 (%)": round(float(mean_curve.get(3, np.nan)), 1),
    "Retention TB thang 6 (%)": round(float(mean_curve.get(6, np.nan)), 1),
    "Ty le FPU (%)": round(fpu_n / (fpu_n + rpu_n) * 100, 1),
    "Ty le RPU (%)": round(rpu_n / (fpu_n + rpu_n) * 100, 1),
    "Trung vi ngay den GD thu 2": float(gap.median()),
}
pd.Series(summary, name="Gia tri").to_frame().to_csv(
    REPORTS_DIR / "cohort_summary_kpi.csv", encoding="utf-8-sig"
)

print(f"Đã xuất toàn bộ kết quả vào: {REPORTS_DIR}")
