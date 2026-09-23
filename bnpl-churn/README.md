# Phân Tích Hành Vi & Dự Báo Nguy Cơ Rời Bỏ (Churn) Khách Hàng BNPL Ứng Dụng Học Máy

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

> **Chuyên đề tốt nghiệp Cử nhân Công nghệ Thông tin**  
> **Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT)**  
> **Sinh viên thực hiện:** Cao Thị Hoàng Lan & Mai Thị Anh Đào  
> **Giảng viên hướng dẫn:** ThS. Mai Xuân Hùng  

---

## Giới thiệu đề tài

Dịch vụ **Mua trước trả sau (Buy Now, Pay Later - BNPL)** trên các nền tảng ví điện tử là dịch vụ tài chính **phi hợp đồng (non-contractual)**. Khách hàng hiếm khi làm thủ tục "hủy dịch vụ" mà thường âm thầm ngừng giao dịch — hiện tượng được gọi là **Rời bỏ im lặng (Silent Churn)**. 

Đề tài giải quyết bài toán giữ chân khách hàng (Retention) và cảnh báo sớm nguy cơ rời bỏ thông qua việc:
1. **Định nghĩa nhãn Churn phi hợp đồng chuẩn xác:** Khách hàng không phát sinh giao dịch nào trong cửa sổ **60 ngày** liên tiếp.
2. **Thiết kế chống rò rỉ dữ liệu (Anti Data Leakage):** Tách biệt thời gian tính toán đặc trưng (trước mốc tham chiếu $T_{ref} = 31/10/2024$) và thời gian gán nhãn dự báo (sau $T_{ref}$).
3. **Phân tích khám phá & Cohort:** Tìm ra *"cửa sổ vàng"* giữ chân khách hàng trong 30 ngày đầu và phát hiện sự phân hóa tỷ lệ rời bỏ theo danh mục chi tiêu.
4. **Xây dựng mô hình máy học & XAI:** So sánh Logistic Regression, Random Forest và XGBoost; diễn giải nguyên nhân dự báo churn bằng **SHAP**.
5. **Đóng gói ứng dụng thực tế:** Phát triển Dashboard và công cụ chấm điểm thời gian thực trên giao diện web **Streamlit**.

---

## Bộ dữ liệu & Ba Insight Cốt Lõi

Hệ thống vận hành trên bộ dữ liệu tổng hợp mô phỏng thị trường ví điện tử Việt Nam gồm **8.000 khách hàng** và **64.746 giao dịch** (tỷ lệ Churn toàn tệp có giao dịch là **17,6%**):

* **Insight 1 (Phân hóa theo Danh mục thanh toán):** Churn rate phân hóa gần **8 lần** giữa các ngành hàng. Các danh mục thiết yếu/định kỳ giữ chân rất cao (*Viễn thông 5,8%*, *Điện & Nước 6,2%*), trong khi các danh mục mua theo dịp/mùa vụ rời bỏ rất lớn (*Điện tử 44,5%*, *Du lịch 35,7%*, *Giải trí 32,4%*).
* **Insight 2 (Ngưỡng sống còn theo vòng đời FPU/RPU):** Tỷ lệ rời bỏ giảm mạnh theo số lần giao dịch tích lũy: Nhóm chỉ giao dịch 1 lần (**FPU**) có churn rate lên đến **67,5%**, giảm xuống **38,8%** (RPU_Early: 2-5 GD) và chỉ còn **0,7%** ở nhóm trung thành (**RPU_Loyal** > 15 GD). Thời gian trung vị để khách quay lại giao dịch lần 2 là **29 ngày**.
* **Insight 3 (Cửa sổ vàng can thiệp):** Phân tích ma trận *Cohort MoM Retention* cho thấy sự sụt giảm lớn nhất (31,3 điểm %) diễn ra ngay trong **30 ngày đầu tiên** sau giao dịch đầu (tháng 1 còn 68,7%), sau đó đường cong đi ngang ổn định quanh mức 71% – 74%.

---

## Kết quả Huấn luyện Mô hình Máy học

Các mô hình được huấn luyện trên 27 đặc trưng (RFM, Payment Quality, Nhân khẩu học) với tập kiểm tra độc lập 1.353 khách hàng:

| Mô hình | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.937 | 0.827 | 0.532 | 0.868 | 0.659 | Baseline tuyến tính, FP cao |
| **Random Forest** | 0.959 | 0.879 | **0.791** | 0.761 | **0.776** | Cân bằng, ưu tiên Precision |
| **XGBoost (Mặc định)** | 0.957 | 0.875 | 0.765 | 0.761 | 0.763 | Nhanh, hiệu năng tốt |
| **XGBoost (Tuned) ** | **0.960** | **0.879** | 0.673 | **0.823** | 0.741 | **Mô hình triển khai chính thức** |

> **Lựa chọn triển khai:** **XGBoost (đã qua tinh chỉnh GridSearchCV)** được chọn làm mô hình cuối cùng vì đạt **Recall = 0.823** (bắt được > 82% khách hàng sắp rời bỏ). Trong bài toán giữ chân khách hàng, chi phí bỏ sót một khách hàng giá trị lớn hơn nhiều so với chi phí gửi nhầm một ưu đãi chăm sóc.
> 
> **Diễn giải bằng SHAP:** 4 đặc trưng cảnh báo churn sớm mạnh nhất thuộc về nhóm **Recency & Frequency** (`recency_days`, `txn_per_month`, `freq_180d`, `freq_90d`). Khách hàng chi tiêu số tiền lớn nhưng thưa thớt vẫn có rủi ro rời bỏ cao hơn khách hàng chi tiêu nhỏ nhưng đều đặn.

---

## Ứng dụng Web Dashboard (Streamlit App)

Ứng dụng gồm 3 trang chức năng chính phục vụ đội ngũ Quản trị trải nghiệm khách hàng (CRM) và Phân tích kinh doanh (BI):

* **Trang 1: Tổng quan khách hàng BNPL (`1_Tổng_quan.py`)**
  * Theo dõi 4 KPI vận hành cốt lõi, tách riêng nhóm "chưa từng giao dịch" (kích hoạt hụt 12,2%).
  * Bộ lọc đa chiều Sidebar tương tác: theo Danh mục, Phân khúc vòng đời (FPU/RPU), Hạng thành phố.
  * Phân phối số ngày Recency theo ngưỡng 60 ngày và tỷ lệ Churn theo mức độ dùng khuyến mãi.
* **Trang 2: Cohort Retention & Danh mục (`2_Cohort_và_Danh_Mục.py`)**
  * Hiển thị Ma trận nhiệt **Cohort MoM Retention** (loại bỏ hiện tượng left-truncation).
  * So sánh đường cong sống sót (retention curve) và tỷ lệ Churn giữa 8 danh mục chi tiêu.
* **Trang 3: Dự báo nguy cơ rời bỏ (`3_Dự_báo_Churn.py`)**
  * **Tab 1 - Dự báo một khách hàng:** Form nhập 27 đặc trưng (tự động lấy trung vị làm mặc định), trả về xác suất churn, phân cấp rủi ro (Thấp/Trung bình/Cao) và biểu đồ SHAP giải thích lý do cụ thể tại sao khách hàng này có nguy cơ rời bỏ.
  * **Tab 2 - Chấm điểm hàng loạt (Batch Scoring):** Tải lên file CSV danh sách khách hàng, hệ thống chấm điểm tức thì (< 1s / 1.000 dòng), tự động sắp xếp theo thứ tự rủi ro giảm dần và xuất file kết quả.
  * **Thanh trượt ngưỡng phân lớp $\theta$:** Tùy biến ngưỡng từ 0.05 đến 0.95 để điều chỉnh đánh đổi Precision – Recall phù hợp với ngân sách từng chiến dịch retention.

---

## Cấu trúc thư mục dự án

```text
├── data/
│   ├── bnpl_customers_clean.csv       # Dữ liệu hồ sơ khách hàng đã làm sạch
│   ├── bnpl_transactions_clean.csv    # Dữ liệu nhật ký giao dịch đã làm sạch
│   ├── bnpl_customer_features.csv     # Đặc trưng toàn kỳ (dùng cho EDA & Trang 1-2)
│   └── bnpl_model_features.csv        # Đặc trưng tính tại mốc T_ref (dùng huấn luyện & Trang 3)
│
├── models/
│   ├── churn_model_final.pkl          # Artifact mô hình cuối cùng (Pipeline + Metadata)
│   └── metrics_bnpl.json              # Kết quả đánh giá hiệu năng mô hình
│
├── notebooks/ (hoặc scripts/)
│   ├── 03_cohort_analysis.py          # Script phân tích Cohort & FPU/RPU
│   ├── 04_modeling.py                 # Huấn luyện mô hình chống rò rỉ nhãn tại T_ref
│   └── 05_model_selection.py          # Đánh giá mô hình & phân tích tầm quan trọng bằng SHAP
│   └── bnpl_churn_pipeline.py         # Pipeline trọn gói End-to-End từ EDA đến Model
│
├── reports/
│   ├── cohort_retention_matrix.csv    # Ma trận giữ chân phục vụ Streamlit Trang 2
│   ├── churn_by_category.csv          # Tỷ lệ churn theo danh mục thanh toán
│   └── figures/                       # Hình ảnh biểu đồ xuất ra cho báo cáo
│
├── streamlit_app/
│   ├── app_utils.py                   # Tiện ích dùng chung, nạp dữ liệu/mô hình & style biểu đồ
│   ├── 1_Tổng_quan.py                 # Màn hình 1: Dashboard tổng quan và bộ lọc
│   └── pages/
│       ├── 2_Cohort_và_Danh_Mục.py    # Màn hình 2: Phân tích Cohort MoM & Danh mục
│       └── 3_Dự_báo_Churn.py          # Màn hình 3: Dự báo đơn lẻ (SHAP) & Batch CSV
│
├── requirements.txt                   # Danh sách các thư viện cần cài đặt
└── README.md                          # Tài liệu giới thiệu & hướng dẫn sử dụng dự án