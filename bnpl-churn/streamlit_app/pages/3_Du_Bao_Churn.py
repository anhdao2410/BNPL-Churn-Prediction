# -*- coding: utf-8 -*-
"""Trang 3 — Dự báo nguy cơ churn: từng khách hàng và chấm điểm hàng loạt."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_utils import (ACCENT, ACCENT_2, MUTED, apply_chart_style, section,  # noqa: E402
                   load_artifact, load_customer_features, require)

st.set_page_config(page_title="BNPL Churn — Dự báo", layout="wide")
apply_chart_style()

st.title("Dự báo nguy cơ rời bỏ")

art = load_artifact()
require(art, "mô hình (models/churn_model_final.pkl)",
        "Chạy notebook 05_model_selection.ipynb để huấn luyện và lưu mô hình.")

pipe = art["pipeline"]
num_cols, cat_cols = art["num_cols"], art["cat_cols"]
num_stats, cat_values = art["num_stats"], art["cat_values"]
m = art.get("metrics_test", {})
st.caption(f"{art['best_name']} · AUC {m.get('AUC', 0):.3f} · Recall {m.get('Recall', 0):.3f} · "
           f"Precision {m.get('Precision', 0):.3f} · đặc trưng tính tại mốc T_ref = {art.get('T_ref', '—')}.")

threshold = st.slider("Ngưỡng phân lớp", 0.05, 0.95, 0.50, 0.05,
                      help="Hạ ngưỡng để bắt nhiều khách rủi ro hơn (tăng Recall), đổi lại nhiều cảnh báo nhầm hơn.")

# Nhãn tiếng Việt + nhóm để bố cục form gọn
GROUPS = {
    "Hành vi gần đây": ["recency_days", "freq_30d", "freq_90d", "freq_180d", "monetary_90d"],
    "Lịch sử giao dịch": ["freq_total", "tenure_days", "txn_per_month", "mean_inter_days",
                          "monetary_total", "aov", "amount_std", "unique_categories"],
    "Thanh toán & rủi ro": ["late_rate", "partial_rate", "promo_rate", "suspicious_count"],
    "Hồ sơ khách hàng": ["age", "city_tier", "kyc_verified", "phone_verified",
                         "credit_limit_vnd", "account_age_days"],
}
VN = {
    "recency_days": "Ngày từ giao dịch gần nhất", "freq_30d": "Số giao dịch 30 ngày",
    "freq_90d": "Số giao dịch 90 ngày", "freq_180d": "Số giao dịch 180 ngày",
    "monetary_90d": "Chi tiêu 90 ngày (VNĐ)", "freq_total": "Tổng số giao dịch",
    "tenure_days": "Ngày từ giao dịch đầu tiên", "txn_per_month": "Giao dịch / tháng",
    "mean_inter_days": "Khoảng cách TB giữa giao dịch (ngày)", "monetary_total": "Tổng chi tiêu (VNĐ)",
    "aov": "Giá trị giao dịch TB (VNĐ)", "amount_std": "Độ lệch chuẩn giá trị",
    "unique_categories": "Số danh mục đã dùng", "late_rate": "Tỷ lệ trả trễ",
    "partial_rate": "Tỷ lệ trả một phần", "promo_rate": "Tỷ lệ có khuyến mãi",
    "suspicious_count": "Số giao dịch nghi ngờ", "age": "Tuổi", "city_tier": "Hạng thành phố",
    "kyc_verified": "Đã eKYC (0/1)", "phone_verified": "Đã xác thực SĐT (0/1)",
    "credit_limit_vnd": "Hạn mức (VNĐ)", "account_age_days": "Tuổi tài khoản (ngày)",
    "top_category": "Danh mục thanh toán chính", "gender": "Giới tính",
    "device_type": "Thiết bị", "referral_source": "Kênh giới thiệu",
}

tab1, tab2 = st.tabs(["Một khách hàng", "Hàng loạt (CSV)"])

# ---------- Tab 1 ----------
with tab1:
    with st.form("predict_form"):
        st.caption("Giá trị mặc định là trung vị toàn bộ dữ liệu — chỉnh để thử nghiệm.")
        values = {}

        cols = st.columns(len(cat_cols))
        for i, c in enumerate(cat_cols):
            values[c] = cols[i].selectbox(VN.get(c, c), cat_values[c])

        for group, fields in GROUPS.items():
            fields = [f for f in fields if f in num_cols]
            if not fields:
                continue
            st.markdown(f"<span style='color:{MUTED};font-size:0.85rem'>{group}</span>",
                        unsafe_allow_html=True)
            cols = st.columns(4)
            for i, c in enumerate(fields):
                s = num_stats[c]
                step = 1.0 if s["max"] - s["min"] > 20 else 0.01
                values[c] = cols[i % 4].number_input(
                    VN.get(c, c), min_value=float(s["min"]), max_value=float(s["max"]),
                    value=float(s["med"]), step=step, key=f"in_{c}")
        submitted = st.form_submit_button("Dự báo")

    if submitted:
        row = pd.DataFrame([values])[num_cols + cat_cols]
        proba = float(pipe.predict_proba(row)[0, 1])
        pred = proba >= threshold
        tier = "Thấp" if proba < 0.30 else ("Trung bình" if proba < 0.60 else "Cao")

        section("Kết quả")
        r1, r2, r3 = st.columns(3)
        r1.metric("Xác suất churn", f"{proba:.1%}")
        r2.metric("Mức rủi ro", tier)
        r3.metric(f"Phân loại (ngưỡng {threshold:.2f})", "Churn" if pred else "Ở lại")
        st.progress(min(proba, 1.0))

        top_shap = art.get("top_shap_features", {})
        top_num = [f for f in top_shap if f in num_cols][:8]
        colA, colB = st.columns(2)
        with colA:
            st.caption("So với trung vị toàn bộ khách hàng (đặc trưng ảnh hưởng mạnh nhất)")
            comp = pd.DataFrame({
                "Đặc trưng": [VN.get(f, f) for f in top_num],
                "Giá trị nhập": [values[f] for f in top_num],
                "Trung vị": [num_stats[f]["med"] for f in top_num],
            })
            st.dataframe(comp, hide_index=True, width='stretch')
        with colB:
            if top_shap:
                imp = pd.Series(top_shap).sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.barh([VN.get(f, f) for f in imp.index], imp.values, color=ACCENT, height=0.6)
                ax.set_title("Mức ảnh hưởng trung bình (SHAP)")
                ax.set_xlabel("Mean |SHAP|")
                st.pyplot(fig); plt.close(fig)

# ---------- Tab 2 ----------
with tab2:
    st.caption("File CSV cần có đúng các cột đặc trưng đầu vào, tính tại cùng mốc T_ref như notebook 04. "
               "Cột khác nếu có sẽ được giữ nguyên trong kết quả.")

    feat = load_customer_features()
    if feat is not None:
        demo_cols = [c for c in (["customer_id"] + num_cols + cat_cols) if c in feat.columns]
        sample = feat[demo_cols].dropna().head(5)
        st.download_button("Tải file mẫu (chỉ để tham khảo định dạng cột)",
                           sample.to_csv(index=False).encode("utf-8-sig"),
                           file_name="mau_cham_diem_churn.csv", mime="text/csv")

    up = st.file_uploader("Chọn file CSV", type=["csv"], label_visibility="collapsed")
    if up is not None:
        batch = pd.read_csv(up)
        missing = [c for c in num_cols + cat_cols if c not in batch.columns]
        if missing:
            st.error("Thiếu các cột bắt buộc: " + ", ".join(missing))
        else:
            probas = pipe.predict_proba(batch[num_cols + cat_cols])[:, 1]
            out = batch.copy()
            out["churn_probability"] = probas.round(4)
            out["risk_tier"] = pd.cut(out["churn_probability"], [-0.01, 0.3, 0.6, 1.01],
                                      labels=["Thấp", "Trung bình", "Cao"])
            out["predicted_churn"] = (out["churn_probability"] >= threshold).astype(int)
            out = out.sort_values("churn_probability", ascending=False)

            k1, k2, k3 = st.columns(3)
            k1.metric("Số khách được chấm", f"{len(out):,}")
            k2.metric("Rủi ro cao", f"{(out['risk_tier'] == 'Cao').sum():,}")
            k3.metric(f"Gắn cờ churn (ngưỡng {threshold:.2f})", f"{int(out['predicted_churn'].sum()):,}")

            st.dataframe(out.head(30), width='stretch')
            st.download_button("Tải toàn bộ kết quả", out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ket_qua_cham_diem_churn.csv", mime="text/csv")
