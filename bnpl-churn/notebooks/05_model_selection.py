import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

plt.rcParams["figure.dpi"] = 110
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = f"{BASE_DIR}/data"
REPORTS_DIR = f"{BASE_DIR}/reports"
FIG_DIR = f"{REPORTS_DIR}/figures"
MODELS_DIR = f"{BASE_DIR}/models"

feat = pd.read_csv(f"{DATA_DIR}/bnpl_model_features.csv", index_col="customer_id")
artifact = joblib.load(f"{MODELS_DIR}/churn_model_bnpl.pkl")
num_cols, cat_cols = artifact["num_cols"], artifact["cat_cols"]

X = feat[num_cols + cat_cols]
y = feat["churn"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(
    f"Tái tạo train/test giống hệt T14 - Test AUC kiểm tra: "
    f"{roc_auc_score(y_te, artifact['pipeline'].predict_proba(X_te)[:, 1]):.4f} "
    f"(phải khớp số trong model_comparison_bnpl.csv)"
)

results = pd.read_csv(f"{REPORTS_DIR}/model_comparison_bnpl.csv", index_col="Model")
results

qualitative = pd.DataFrame(
    {
        "Khả năng diễn giải": [
            "Cao (hệ số tuyến tính)",
            "Trung bình (feature_importances_)",
            "Trung bình-Thấp (cần SHAP)",
            "Trung bình (đã bổ trợ bằng SHAP)",
        ],
        "Độ phức tạp huấn luyện": [
            "Thấp",
            "Trung bình",
            "Cao (nhiều siêu tham số)",
            "Cao (GridSearchCV)",
        ],
        "Độ nhạy với outlier": ["Cao (cần chuẩn hóa)", "Thấp", "Thấp", "Thấp"],
        "Phù hợp production": [
            "Dễ giám sát/giải trình",
            "Cân bằng",
            "Cấu hình cơ sở trước khi tinh chỉnh",
            "Mô hình chính thức của ứng dụng",
        ],
    },
    index=["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (tuned)"],
)

summary_table = results.join(qualitative)
summary_table.to_csv(f"{REPORTS_DIR}/model_selection_summary.csv", encoding="utf-8-sig")
summary_table

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
metrics_plot = results[["AUC", "PR-AUC", "F1 (churn)", "Recall"]]
metrics_plot.plot(
    kind="bar", ax=axes[0], color=["#2a9d8f", "#457b9d", "#f2b134", "#e76f51"]
)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Điểm số")
axes[0].set_title("So sánh 4 chỉ số chính giữa các mô hình")
axes[0].legend(fontsize=8, ncol=2)
axes[0].tick_params(axis="x", rotation=15)

axes[1].scatter(
    results["Precision"], results["Recall"], s=180,
    c=["#457b9d", "#2a9d8f", "#f2b134", "#e76f51"][: len(results)],
)
for name, row in results.iterrows():
    axes[1].annotate(
        name,
        (row["Precision"], row["Recall"]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
    )
axes[1].set_xlabel("Precision")
axes[1].set_ylabel("Recall")
axes[1].set_title("Đánh đổi Precision-Recall")
axes[1].set_xlim(0.4, 1)
axes[1].set_ylim(0.6, 1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_comparison_summary.png", bbox_inches="tight")
plt.show()

pipe = artifact["pipeline"]
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]

X_te_pre = pre.transform(X_te)
feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
X_te_df = pd.DataFrame(X_te_pre, columns=feature_names, index=X_te.index)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_te_df)

fig = plt.figure(figsize=(9, 6.5))
shap.summary_plot(shap_values, X_te_df, max_display=15, show=False)
plt.title(f"SHAP Summary - {artifact['best_name']}", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary_beeswarm.png", bbox_inches="tight", dpi=110)
plt.show()

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=feature_names
).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
mean_abs_shap.head(15).sort_values().plot(kind="barh", ax=ax, color="#2a9d8f")
ax.set_xlabel("Mean |SHAP value| - mức ảnh hưởng trung bình đến xác suất churn")
ax.set_title(f"Top 15 đặc trưng quan trọng nhất theo SHAP - {artifact['best_name']}")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_importance_bar.png", bbox_inches="tight")
plt.show()

mean_abs_shap.head(10).round(4)

default_imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(
    ascending=False
)
compare = (
    pd.DataFrame(
        {
            "SHAP rank": mean_abs_shap.rank(ascending=False).astype(int),
            "feature_importances_ rank": default_imp.rank(ascending=False).astype(int),
        }
    )
    .sort_values("SHAP rank")
    .head(10)
)
print(
    "Đối chiếu thứ hạng Top 10 đặc trưng giữa hai phương pháp - càng gần nhau càng đáng tin cậy:"
)
compare

final_name = artifact["best_name"]
final_pipe = artifact["pipeline"]
final_metrics = results.loc[final_name].to_dict()

final_artifact = {
    **artifact,
    "selection_rationale": (
        "Chọn XGBoost (tuned) làm mô hình cuối: hiệu năng AUC/Recall cận trên trong 3 mô hình, "
        "đã qua GridSearchCV, và SHAP xác nhận 4 đặc trưng hàng đầu (recency_days, txn_per_month, "
        "freq_180d, freq_90d) đều thuộc nhóm RFM có nền tảng lý thuyết ở Chương 2. "
        "Phương án thay thế nếu ưu tiên diễn giải: Random Forest (chênh AUC test 0.002)."
    ),
    "top_shap_features": mean_abs_shap.head(10).round(4).to_dict(),
    "model_selection_summary": summary_table.to_dict(orient="index"),
    "finalized_at": str(pd.Timestamp.now()),
}
joblib.dump(final_artifact, f"{MODELS_DIR}/churn_model_final.pkl")

with open(f"{MODELS_DIR}/model_selection_report.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "final_model": final_name,
            "test_metrics": final_metrics,
            "rationale": final_artifact["selection_rationale"],
            "top_shap_features": final_artifact["top_shap_features"],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

chk = joblib.load(f"{MODELS_DIR}/churn_model_final.pkl")
print(f"Đã lưu ../models/churn_model_final.pkl - mô hình: {chk['best_name']}")
print(
    f"   Test AUC: {chk['metrics_test']['AUC']} | Recall: {chk['metrics_test']['Recall']}"
)
print(
    "Load lại OK, demo proba:",
    chk["pipeline"].predict_proba(X_te.iloc[:3])[:, 1].round(3),
)
