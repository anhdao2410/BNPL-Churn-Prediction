# Phân tích hành vi và dự báo khả năng rời bỏ của khách hàng sản phẩm Mua trước trả sau (BNPL) ứng dụng học máy

Chuyên đề tốt nghiệp IE400 - Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
Khoa Khoa học và Kỹ thuật Thông tin

| | |
|---|---|
| Sinh viên thực hiện | Cao Thị Hoàng Lan (25210129), Mai Thị Anh Đào (25210078) |
| Giảng viên hướng dẫn | ThS. Mai Xuân Hùng |
| Ứng dụng trực tuyến | https://bnpl-churn-prediction-j6jkrbefserjvhsoykqjdc.streamlit.app |

---

## 1. Giới thiệu

Dịch vụ Mua trước trả sau (Buy Now, Pay Later - BNPL) trên ví điện tử là dịch vụ phi hợp đồng: khách hàng không làm thủ tục hủy mà lặng lẽ ngừng giao dịch (silent churn). Đề tài xây dựng quy trình hoàn chỉnh từ dữ liệu đến ứng dụng:

- **Định nghĩa churn**: khách hàng không phát sinh giao dịch trong 60 ngày liên tiếp.
- **Chống rò rỉ nhãn**: đặc trưng chỉ tính từ giao dịch trước mốc tham chiếu `T_ref = 31/10/2024`, nhãn xác định từ hoạt động sau mốc này.
- **Phân tích hành vi**: Cohort MoM Retention, vòng đời FPU/RPU, tỷ lệ rời bỏ theo danh mục thanh toán.
- **Mô hình học máy**: so sánh Logistic Regression, Random Forest, XGBoost; diễn giải bằng SHAP.
- **Ứng dụng web Streamlit**: dashboard phân tích và công cụ dự báo cho từng khách hàng hoặc theo lô.

## 2. Dữ liệu và phát hiện chính

Tập dữ liệu tổng hợp mô phỏng hành vi khách hàng BNPL trên ví điện tử Việt Nam: 8.000 khách hàng, 64.746 giao dịch năm 2024. Tỷ lệ churn trên 7.024 khách hàng đã giao dịch là 17,6%.

1. **Danh mục thanh toán**: churn chênh lệch gần 8 lần giữa các danh mục. Nhóm thiết yếu, định kỳ giữ chân tốt (Viễn thông 5,8%, Điện & Nước 6,2%); nhóm mua theo dịp rời bỏ nhiều (Điện tử 44,5%, Du lịch & Vận chuyển 35,7%, Giải trí 32,4%).
2. **Vòng đời FPU/RPU**: churn giảm mạnh theo số giao dịch tích lũy, từ 67,5% ở nhóm chỉ giao dịch 1 lần (FPU) xuống 38,8% (RPU_Early, 2-5 giao dịch) và 0,7% (RPU_Loyal, trên 15 giao dịch). Trung vị thời gian đến giao dịch thứ hai là 29 ngày.
3. **Cửa sổ vàng 30 ngày**: retention sụt mạnh nhất ngay tháng đầu (còn 68,7%), sau đó đi ngang quanh 71-74%.

## 3. Kết quả mô hình

Huấn luyện trên 27 đặc trưng (23 số, 4 phân loại), đánh giá trên tập kiểm tra độc lập 1.353 khách hàng:

| Mô hình | ROC-AUC | PR-AUC | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0,937 | 0,827 | 0,532 | 0,868 | 0,659 |
| Random Forest | 0,959 | 0,879 | 0,791 | 0,761 | 0,776 |
| XGBoost | 0,957 | 0,875 | 0,765 | 0,761 | 0,763 |
| **XGBoost (đã tinh chỉnh)** | **0,960** | **0,879** | 0,673 | **0,823** | 0,741 |

Mô hình chính thức là XGBoost tinh chỉnh bằng GridSearchCV (`max_depth = 3`, `learning_rate = 0.05`): phát hiện được 82,3% khách hàng sắp rời bỏ. Theo SHAP, bốn đặc trưng ảnh hưởng mạnh nhất đều thuộc nhóm Recency/Frequency: `recency_days`, `txn_per_month`, `freq_180d`, `freq_90d`.

## 4. Cài đặt và chạy

Yêu cầu Python 3.10 trở lên (khuyến nghị 3.11). Mọi lệnh chạy tại thư mục `bnpl-churn`.

```bash
git clone https://github.com/janecao262/UIT_IE400.git
cd UIT_IE400/bnpl-churn

python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate

pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/1_Tổng_quan.py
```

Ứng dụng mở tại http://localhost:8501. Repo kèm sẵn dữ liệu và mô hình đã huấn luyện nên chạy được ngay, không cần huấn luyện lại.

Hướng dẫn chi tiết từng bước, cách dùng từng trang và xử lý lỗi thường gặp: xem tài liệu **Hướng dẫn cài đặt và sử dụng** nộp kèm báo cáo.

### Ứng dụng gồm 3 trang

| Trang | Tệp | Chức năng |
|---|---|---|
| Tổng quan | `streamlit_app/1_Tổng_quan.py` | Chỉ số tổng quan, churn theo vòng đời và danh mục, bộ lọc ở thanh bên |
| Cohort và Danh Mục | `streamlit_app/pages/2_Cohort_và_Danh_Mục.py` | Ma trận Cohort MoM Retention, retention theo danh mục, phân khúc FPU/RPU |
| Dự báo Churn | `streamlit_app/pages/3_Dự_báo_Churn.py` | Dự báo một khách hàng (kèm SHAP) hoặc chấm điểm hàng loạt từ CSV, điều chỉnh ngưỡng phân lớp |

## 5. Huấn luyện lại và kiểm thử (tùy chọn)

```bash
pip install shap pytest

# Tắt cửa sổ biểu đồ khi chạy script
# Windows (cmd): set MPLBACKEND=Agg   |   macOS / Linux: export MPLBACKEND=Agg
python notebooks/03_cohort_analysis.py     # Cohort, FPU/RPU, danh mục -> reports/
python notebooks/04_modeling.py            # Đặc trưng T_ref, huấn luyện 3 mô hình, GridSearchCV
python notebooks/05_model_selection.py     # SHAP, xuất models/churn_model_final.pkl

python -m pytest tests/ -q                 # 10 kịch bản TC01-TC10, kết quả mong đợi: 12 passed
```

Script 04 và 05 ghi đè mô hình và bảng kết quả, nên sao lưu `models/` và `reports/` trước. Mô hình trong repo là mô hình dùng cho số liệu báo cáo; huấn luyện lại bằng phiên bản thư viện mới hơn cho kết quả rất sát nhưng không trùng tuyệt đối (ROC-AUC 0,9598 so với 0,9600).

## 6. Hai bảng đặc trưng có mục đích khác nhau

| Tệp | Dùng cho | Ghi chú |
|---|---|---|
| `data/bnpl_customer_features.csv` | Phân tích mô tả, Trang 1-2 | Tổng hợp toàn kỳ 2024, **không** dùng để huấn luyện vì trùng với giai đoạn gán nhãn (rò rỉ nhãn) |
| `data/bnpl_model_features.csv` | Huấn luyện, Trang 3 | 27 đặc trưng tính tại mốc `T_ref`, cột `churn` là nhãn |

Trong mã nguồn, `app_utils.load_customer_features()` đọc bảng toàn kỳ và `app_utils.load_model_features()` đọc bảng tại `T_ref`.

## 7. Cấu trúc thư mục

```text
bnpl-churn/
├── data/
│   ├── bnpl_transactions_clean.csv      # Log giao dịch đã làm sạch
│   ├── bnpl_customers_clean.csv         # Hồ sơ khách hàng đã làm sạch
│   ├── bnpl_customer_features.csv       # Đặc trưng toàn kỳ (EDA, Trang 1-2)
│   └── bnpl_model_features.csv          # Đặc trưng tại T_ref (huấn luyện, Trang 3)
├── models/
│   ├── churn_model_final.pkl            # Mô hình ứng dụng đang dùng (XGBoost tuned)
│   ├── churn_model_bnpl.pkl             # Mô hình trung gian do 04 tạo
│   ├── metrics_bnpl.json                # Chỉ số đánh giá
│   └── model_selection_report.json      # Kết quả lựa chọn mô hình và SHAP
├── notebooks/
│   ├── 03_cohort_analysis.py            # Cohort MoM, FPU/RPU, retention theo danh mục
│   ├── 04_modeling.py                   # Đặc trưng T_ref, huấn luyện và tinh chỉnh
│   └── 05_model_selection.py            # SHAP, chọn và xuất mô hình cuối
├── reports/                             # Bảng số liệu và figures/ cho báo cáo, ứng dụng
├── streamlit_app/
│   ├── 1_Tổng_quan.py                   # Trang 1 (tệp chính)
│   ├── app_utils.py                     # Nạp dữ liệu, mô hình; style biểu đồ
│   ├── requirements.txt                 # Thư viện cần cài
│   └── pages/
│       ├── 2_Cohort_và_Danh_Mục.py      # Trang 2
│       └── 3_Dự_báo_Churn.py            # Trang 3
├── tests/
│   └── test_app.py                      # Kiểm thử tự động TC01-TC10
├── .gitignore
└── README.md
```

## 8. Triển khai trên Streamlit Community Cloud

Tại https://share.streamlit.io, tạo ứng dụng từ repo này với:

- **Branch**: `main`
- **Main file path**: `bnpl-churn/streamlit_app/1_Tổng_quan.py`

Streamlit Cloud tự cài thư viện theo `streamlit_app/requirements.txt` và triển khai lại mỗi khi có commit mới lên nhánh `main`.
