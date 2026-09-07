# 📉 BNPL Churn Prediction — Chuyên đề tốt nghiệp IE400

Phân tích hành vi và dự báo khả năng rời bỏ (churn) khách hàng sản phẩm **Mua trước trả sau (BNPL)** trên ví điện tử.

| | |
|---|---|
| **Sinh viên** | Cao Thị Hoàng Lan (Lead) · Mai Thị Anh Đào (Member) |
| **CBHD** | ThS. Mai Xuân Hùng |

## 🗂 Cấu trúc

```
bnpl-churn/
├── data/
│   ├── bnpl_transactions_clean.csv     # log giao dịch sạch (T10)
│   ├── bnpl_customers_clean.csv        # hồ sơ khách hàng sạch (T10)
│   ├── bnpl_customer_features.csv      # đặc trưng toàn kỳ — DÙNG CHO EDA, KHÔNG DÙNG ĐỂ TRAIN (T11)
│   └── bnpl_model_features.csv         # đặc trưng tính theo T_ref — dùng để train (T14)
├── notebooks/
│   ├── 03_cohort_analysis.ipynb        # T13 — cohort MoM, FPU/RPU, category retention
│   ├── 04_modeling.ipynb               # T14 — 3 model, đặc trưng leakage-safe (T_ref)
│   └── 05_model_selection.ipynb        # T15 — SHAP, so sánh model, chọn model final
├── models/
│   └── churn_model_final.pkl           # ⭐ model app đang dùng (XGBoost tuned, AUC 0.96)
├── reports/                            # bảng số liệu + figures/ cho báo cáo & app
└── streamlit_app/                      # T17–T18 ✅ web app 3 trang
    ├── app.py                          # 📊 Trang 1: Dashboard
    ├── pages/2_Cohort_Category.py      # 📈 Trang 2: Cohort & Category retention
    ├── pages/3_Du_Bao_Churn.py         # 🔮 Trang 3: Dự báo churn
    ├── utils.py
    └── WIREFRAME.md                    # T17 — thiết kế UI/use-case cho mục 4.1
```

## 🚀 Chạy nhanh

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```
Repo đã kèm sẵn model + dữ liệu nên chạy được ngay, không cần huấn luyện lại.

## ⚠️ Lưu ý quan trọng — 2 bảng đặc trưng khác mục đích

- `bnpl_customer_features.csv` (T11, toàn kỳ 2024): dùng cho **EDA** (Trang 1, 2 của app; mục 3.3 báo cáo). **Không** dùng trực tiếp để huấn luyện — chứa rò rỉ nhãn vì tính trên cùng cửa sổ dùng để xác định churn (xem mục 3.2.2).
- `bnpl_model_features.csv` (T14, cắt tại mốc T_ref): dùng để **huấn luyện & dự báo** (Trang 3 của app; mục 3.4 báo cáo).

## 🐙 GitHub & ☁️ Deploy (T19)

```bash
cd bnpl-churn
git init && git add . && git commit -m "BNPL churn app — T17-T19"
git remote add origin https://github.com/<username>/bnpl-churn.git
git branch -M main && git push -u origin main
```
Deploy: https://share.streamlit.io → **New app** → chọn repo, main file `streamlit_app/app.py` → Deploy.
Backup nếu không deploy được: quay video demo 2–3 phút chạy app local.

## 🗺 Notebook/App ↔ Mục báo cáo

| Sản phẩm | Task | Mục báo cáo |
|---|---|---|
| notebooks/03 | T13 | 3.3.2 · 3.3.3 |
| notebooks/04 | T14 | 3.2.2 · 3.4 |
| notebooks/05 | T15 | 3.5.2 · 3.5.3 |
| streamlit_app/ + WIREFRAME.md | T17 · T18 | 4.1 · 4.2 |
| Deploy | T19 | 4.2 (link/video demo) |
