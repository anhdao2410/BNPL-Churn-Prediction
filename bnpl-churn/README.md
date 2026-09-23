# PHÂN TÍCH HÀNH VI VÀ DỰ BÁO KHẢ NĂNG RỜI BỎ CỦA KHÁCH HÀNG SẢN PHẨM MUA TRƯỚC TRẢ SAU (BNPL) ỨNG DỤNG HỌC MÁY

Chuyên đề tốt nghiệp Cử nhân Công nghệ Thông tin
Trường Đại học Công nghệ Thông tin – ĐHQG-HCM
Khoa Khoa học và Kỹ thuật Thông tin

Sinh viên thực hiện:
- Cao Thị Hoàng Lan (MSSV: 25210129)
- Mai Thị Anh Đào (MSSV: 25210078)

Giảng viên hướng dẫn:
- ThS. Mai Xuân Hùng

---

## 1. Giới thiệu đề tài

Dịch vụ Mua trước trả sau (Buy Now, Pay Later - BNPL) trên ví điện tử là dịch vụ tài chính phi hợp đồng (non-contractual). Khách hàng không thực hiện thủ tục hủy dịch vụ mà thường âm thầm ngừng phát sinh giao dịch, dẫn đến hiện tượng rời bỏ im lặng (silent churn).

Đề tài tập trung giải quyết các bài toán sau:
- Định nghĩa nhãn Churn phi hợp đồng: Khách hàng không phát sinh giao dịch nào trong 60 ngày liên tiếp.
- Xử lý triệt để rò rỉ nhãn (Data Leakage): Tách biệt thời gian tính toán đặc trưng (trước mốc tham chiếu T_ref = 31/10/2024) và thời gian gán nhãn dự báo (sau mốc T_ref).
- Phân tích khám phá dữ liệu (EDA) và Cohort: Xác định cửa sổ vàng giữ chân khách hàng trong 30 ngày đầu tiên và khảo sát sự phân hóa tỷ lệ rời bỏ theo danh mục chi tiêu.
- Xây dựng mô hình phân loại học máy: So sánh Logistic Regression, Random Forest và XGBoost; diễn giải đóng góp của từng đặc trưng bằng giá trị SHAP (SHapley Additive exPlanations).
- Triển khai ứng dụng thực tế: Xây dựng Dashboard phân tích và công cụ dự báo trực quan trên giao diện Streamlit.

---

## 2. Dữ liệu và Các phát hiện cốt lõi

Dự án sử dụng tập dữ liệu mô phỏng hành vi khách hàng BNPL trên ví điện tử Việt Nam gồm 8.000 khách hàng và 64.746 giao dịch năm 2024 (tỷ lệ rời bỏ trên 7.024 khách hàng đã giao dịch là 17,6%).

Ba insight cốt lõi thu được từ phân tích:
1. Phân hóa theo Danh mục thanh toán: Tỷ lệ churn phân hóa gần 8 lần giữa các danh mục. Các nhóm thiết yếu, định kỳ (Viễn thông 5,8%, Điện & Nước 6,2%) giữ chân tốt hơn hẳn các nhóm mua sắm theo dịp, mùa vụ (Điện tử 44,5%, Du lịch & Vận chuyển 35,7%, Giải trí 32,4%).
2. Ngưỡng sống còn theo vòng đời FPU/RPU: Tỷ lệ rời bỏ giảm mạnh theo số lượng giao dịch tích lũy: Nhóm khách hàng chỉ giao dịch 1 lần (FPU) có churn rate lên tới 67,5%, giảm xuống 38,8% ở nhóm RPU_Early (2-5 giao dịch) và chỉ còn 0,7% ở nhóm RPU_Loyal (> 15 giao dịch). Thời gian trung vị để khách hàng thực hiện giao dịch thứ hai là 29 ngày.
3. Cửa sổ vàng can thiệp: Phân tích Cohort MoM Retention chỉ ra mức sụt giảm lớn nhất (31,3 điểm phần trăm) diễn ra ngay trong 30 ngày đầu sau giao dịch đầu tiên (tháng 1 còn 68,7%), sau đó đường cong giữ chân đi ngang ổn định quanh mức 71% - 74%.

---

## 3. Hiệu năng mô hình học máy

Các mô hình được huấn luyện trên 27 đặc trưng (RFM, hành vi thanh toán, nhân khẩu học) và đánh giá trên tập kiểm tra độc lập 1.353 khách hàng:

| Mô hình | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.937 | 0.827 | 0.532 | 0.868 | 0.659 |
| Random Forest | 0.959 | 0.879 | 0.791 | 0.761 | 0.776 |
| XGBoost (Mặc định) | 0.957 | 0.875 | 0.765 | 0.761 | 0.763 |
| XGBoost (Tuned) | 0.960 | 0.879 | 0.673 | 0.823 | 0.741 |

Mô hình được chọn triển khai:
XGBoost đã tinh chỉnh qua GridSearchCV (max_depth = 3, learning_rate = 0.05) được chọn làm mô hình chính thức. Mô hình đạt Recall = 0.823 (phát hiện được hơn 82% khách hàng sắp rời bỏ), phù hợp với mục tiêu giữ chân khách hàng khi chi phí bỏ sót một khách hàng lớn hơn nhiều chi phí gửi nhầm ưu đãi.

Phân tích tầm quan trọng bằng SHAP:
Bốn đặc trưng dẫn đầu đều thuộc nhóm Recency và Frequency: recency_days, txn_per_month, freq_180d và freq_90d. Điều này chứng minh tính liên tục và nhịp độ giao dịch đóng vai trò quyết định hơn giá trị chi tiêu tiền tệ.

---

## 4. Cấu trúc thư mục dự án

```text
bnpl-churn/
├── data/
│   ├── bnpl_customers_clean.csv          # Dữ liệu khách hàng đã làm sạch
│   ├── bnpl_transactions_clean.csv       # Dữ liệu giao dịch đã làm sạch
│   ├── bnpl_customer_features.csv        # Đặc trưng toàn kỳ (cho EDA và Trang 1-2)
│   └── bnpl_model_features.csv           # Đặc trưng tính tại mốc T_ref (cho huấn luyện và Trang 3)
│
├── models/
│   ├── churn_model_bnpl.pkl              # Artifact mô hình cơ sở
│   ├── churn_model_final.pkl             # Artifact mô hình cuối cùng phục vụ ứng dụng
│   ├── metrics_bnpl.json                 # Chỉ số đánh giá mô hình
│   └── model_selection_report.json       # Báo cáo lựa chọn mô hình và SHAP
│
├── notebooks/
│   ├── 04_modeling.py                    # Huấn luyện mô hình chống rò rỉ nhãn tại T_ref
│   └── 05_model_selection.py             # Đánh giá mô hình chi tiết và trích xuất SHAP
│
├── reports/
│   ├── churn_by_category.csv             # Tỷ lệ rời bỏ theo danh mục thanh toán
│   ├── cohort_retention_matrix.csv       # Ma trận giữ chân khách hàng theo cohort
│   ├── cohort_summary_kpi.csv            # Tóm tắt chỉ số KPI cohort
│   ├── lifecycle_segment_summary.csv     # Tổng hợp phân khúc vòng đời FPU/RPU
│   ├── model_comparison_bnpl.csv         # Bảng so sánh định lượng các mô hình
│   ├── model_selection_summary.csv       # Bảng so sánh định lượng và định tính
│   ├── retention_curve_by_category.csv   # Đường cong giữ chân theo danh mục
│   └── figures/                          # Biểu đồ phân tích và kết quả mô hình
│
├── streamlit_app/
│   ├── 1_Tổng_quan.py                    # Màn hình 1: Tổng quan khách hàng và bộ lọc
│   ├── app_utils.py                      # Hàm tiện ích nạp mô hình, dữ liệu và định dạng biểu đồ
│   ├── requirements.txt                  # Danh sách thư viện phụ thuộc cho ứng dụng
│   └── pages/
│       ├── 2_Cohort_và_Danh_Mục.py       # Màn hình 2: Phân tích Cohort MoM và Danh mục
│       └── 3_Dự_báo_Churn.py             # Màn hình 3: Dự báo đơn lẻ (SHAP) và theo lô CSV
│
├── tests/
│   └── test_app.py                       # Kịch bản kiểm thử tự động cho ứng dụng
│
├── .gitignore
└── README.md