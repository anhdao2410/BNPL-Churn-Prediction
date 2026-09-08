# WIREFRAME & THIẾT KẾ UI — WEB APP DỰ BÁO CHURN BNPL
### Task T17 (Đào, Lan input yêu cầu) · Tư liệu cho mục 4.1 (Yêu cầu, Use-case, Kiến trúc) của báo cáo

## 1. Yêu cầu chức năng (4.1.1)
- **F1** Dashboard tổng quan: KPI churn, phân bố vòng đời khách hàng (FPU/RPU), bộ lọc theo danh mục/phương thức thanh toán.
- **F2** Phân tích Cohort MoM Retention + Retention theo Danh mục thanh toán (signature insight của đề tài).
- **F3** Dự báo xác suất churn cho 1 khách hàng nhập tay, kèm biểu đồ feature importance; chấm điểm hàng loạt qua CSV.
- **Phi chức năng:** chạy được local (`streamlit run app.py`) và Streamlit Community Cloud; dùng đúng model đã chọn ở T15 (`churn_model_final.pkl`, XGBoost đã tinh chỉnh, đặc trưng tính theo mốc `T_ref` — không dùng bảng snapshot có rò rỉ nhãn); giao diện tiếng Việt.

## 2. Use-case chính (4.1.2)
Nhân viên CSKH/phân tích mở Dashboard xem tình hình chung → sang trang Cohort/Category xem nhóm khách và danh mục nào giữ chân tốt/kém → sang trang Dự báo: nhập hồ sơ 1 khách hoặc upload danh sách toàn bộ → nhận xác suất churn + phân hạng rủi ro → xuất danh sách nhóm rủi ro cao cho chiến dịch giữ chân.

## 3. Kiến trúc luồng xử lý (4.1.3)
```
bnpl_transactions_clean.csv ─┐
bnpl_customers_clean.csv ────┼─▶ notebooks 03/04/05 (T13/T14/T15)
bnpl_customer_features.csv ──┘         │
                                        ▼
                     reports/*.csv + models/churn_model_final.pkl
                                        │
                                        ▼
                     streamlit_app/ (3 trang, cache bằng st.cache_data/resource)
```
Model dùng trong app **không** được huấn luyện trực tiếp trên `bnpl_customer_features.csv` (bảng này rò rỉ nhãn — xem mục 3.2.2 báo cáo) mà trên đặc trưng tính lại từ log giao dịch, cắt tại mốc `T_ref`. Form dự báo ở Trang 3 vì vậy hỏi các câu hỏi hành vi gần đây (bao nhiêu ngày chưa giao dịch, tần suất 30/90/180 ngày gần nhất...) chứ không hỏi tổng số giao dịch cả năm.

## 4. Wireframe 3 trang (4.2)

**Trang 1 — Tổng quan khách hàng** (`app.py`)
```
│ [KPI: Số KH] [Churn % (active)] [Số KH FPU/RPU] [Churn nhóm No_Transaction] │
│ [Sidebar lọc: Danh mục ưa thích / Phương thức thanh toán / Lifecycle segment]│
│ [Bar: churn theo lifecycle segment] [Bar: churn theo danh mục thanh toán]   │
│ [Hist: phân phối recency_days theo churn]                                  │
│ [Expander: bảng dữ liệu + tải CSV]                                         │
```

**Trang 2 — Cohort & Category Retention** (`pages/2_Cohort_Category.py`)
```
│ [Heatmap: Cohort MoM Retention]                                            │
│ [Line: đường cong retention trung bình theo tháng]                        │
│ [Bar + Line: Retention theo danh mục thanh toán]                          │
│ [Bảng: FPU/RPU — số khách, tỷ trọng, churn rate]                          │
```

**Trang 3 — Dự báo Churn** (`pages/3_Du_Bao_Churn.py`)
```
│ [Slider ngưỡng θ]                                                          │
│ Tab 1: form các trường hành vi gần T_ref (tự sinh từ metadata model)      │
│        → % churn + progress + hạng mức rủi ro Thấp/Trung bình/Cao + biểu đồ SHAP    │
│ Tab 2: upload CSV (đúng schema đặc trưng) → bảng kết quả + nút tải về     │
```

## 5. Trạng thái triển khai
- Cả 3 trang code xong (T18), model dùng đúng `churn_model_final.pkl` (XGBoost tuned, AUC test 0,96).
- T19 (deploy) cần tài khoản Streamlit Community Cloud của nhóm: push repo lên GitHub → share.streamlit.io → New app → chọn repo, main file `streamlit_app/app.py` → Deploy. Backup: quay video demo 2–3 phút.

## 6. Nguyên tắc giao diện (bản chỉnh sửa 08/09/2026)
- Không dùng emoji/icon trong tiêu đề, nút, thông báo.
- Bảng màu tiết chế: một màu chủ đạo (xanh đậm) cho biểu đồ, một màu nhấn (đỏ gạch) chỉ dành cho churn/cảnh báo, xám cho đường tham chiếu.
- Biểu đồ matplotlib bỏ viền trên/phải, lưới mờ, chú thích số trực tiếp trên cột.
- Mỗi mục bắt đầu bằng một dòng tiêu đề đậm + chú thích nhỏ; tránh nhiều hộp thông báo màu.
- Form dự báo gom 23 trường số thành 4 nhóm (Hành vi gần đây / Lịch sử giao dịch / Thanh toán & rủi ro / Hồ sơ), 4 cột mỗi hàng.
