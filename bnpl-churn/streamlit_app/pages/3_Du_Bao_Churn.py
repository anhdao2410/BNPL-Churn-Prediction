# -*- coding: utf-8 -*-
"""Trang 3 — 🔮 Dự báo nguy cơ churn (từng khách hàng + chấm điểm hàng loạt)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import PALETTE, load_artifact, load_customer_features, require  # noqa: E402

st.set_page_config(page_title="BNPL Churn — Dự báo", page_icon="🔮", layout="wide")

st.title("🔮 Dự báo nguy cơ rời bỏ (Churn)")

art = load_artifact()
require(art, "model (models/churn_model_final.pkl)",
        "Hãy chạy notebook 05_model_selection.ipynb (T15) để huấn luyện và lưu model.")

pipe = art["pipeline"]
num_cols, cat_cols = art["num_cols"], art["cat_cols"]
num_stats, cat_values = art["num_stats"], art["cat_values"]

m = art["metrics_test"]
st.caption(f"Model: **{art['best_name']}** · AUC {m.get('AUC', 0):.3f} · Recall {m.get('Recall', 0):.3f} · "
           f"Precision {m.get('Precision', 0):.3f} · đặc trưng tính tại mốc T_ref = {art.get('T_ref', '?')} "
           f"(không dùng bảng snapshot toàn kỳ — xem mục 3.2.2 báo cáo)")

threshold = st.slider("Ngưỡng phân lớp (θ) — hạ ngưỡng để bắt được nhiều khách rủi ro hơn (tăng Recall, "
                      "đổi lại nhiều cảnh báo nhầm hơn — xem mục 3.5.3)",
                      0.05, 0.95, 0.50, 0.05)

tab1, tab2 = st.tabs(["👤 Dự báo 1 khách hàng", "📄 Chấm điểm hàng loạt (CSV)"])

VN_NUM = {
    "recency_days": "Số ngày từ giao dịch gần nhất (đến mốc T_ref)",
    "tenure_days": "Số ngày từ giao dịch đầu tiên đến T_ref",
    "freq_total": "Tổng số giao dịch trước T_ref",
    "freq_30d": "Số giao dịch trong 30 ngày gần T_ref",
    "freq_90d": "Số giao dịch trong 90 ngày gần T_ref",
    "freq_180d": "Số giao dịch trong 180 ngày gần T_ref",
    "monetary_90d": "Tổng chi tiêu 90 ngày gần T_ref (VNĐ)",
    "monetary_total": "Tổng chi tiêu toàn kỳ trước T_ref (VNĐ)",
    "aov": "Giá trị giao dịch trung bình (VNĐ)",
    "amount_std": "Độ lệch chuẩn giá trị giao dịch",
    "unique_categories": "Số danh mục thanh toán từng dùng",
    "txn_per_month": "Số giao dịch trung bình mỗi tháng",
    "mean_inter_days": "Khoảng cách trung bình giữa 2 giao dịch (ngày)",
    "late_rate": "Tỷ lệ giao dịch trả trễ hạn (0-1)",
    "partial_rate": "Tỷ lệ giao dịch trả một phần (0-1)",
    "promo_rate": "Tỷ lệ giao dịch có khuyến mãi (0-1)",
    "suspicious_count": "Số giao dịch bị gắn cờ nghi ngờ",
    "age": "Tuổi khách hàng",
    "city_tier": "Hạng thành phố (1-3)",
    "kyc_verified": "Đã xác thực eKYC (0/1)",
    "phone_verified": "Đã xác thực SĐT (0/1)",
    "credit_limit_vnd": "Hạn mức tín dụng (VNĐ)",
    "account_age_days": "Số ngày từ kích hoạt tài khoản đến T_ref",
}
VN_CAT = {
    "top_category": "Danh mục thanh toán chính",
    "gender": "Giới tính",
    "device_type": "Thiết bị",
    "referral_source": "Kênh giới thiệu",
}

# ---------------- Tab 1: một khách hàng ----------------
with tab1:
    with st.form("predict_form"):
        st.markdown("**Thông tin khách hàng** *(mặc định = trung vị toàn bộ dữ liệu — chỉnh để thử nghiệm)*")
        ccols = st.columns(3)
        values = {}
        for i, c in enumerate(cat_cols):
            values[c] = ccols[i % 3].selectbox(VN_CAT.get(c, c), cat_values[c])
        ncols = st.columns(3)
        for i, c in enumerate(num_cols):
            s = num_stats[c]
            step = 1.0 if s["max"] - s["min"] > 20 else 0.01
            values[c] = ncols[i % 3].number_input(
                VN_NUM.get(c, c), min_value=float(s["min"]), max_value=float(s["max"]),
                value=float(s["med"]), step=step)
        submitted = st.form_submit_button("🚀 Dự báo", type="primary")

    if submitted:
        row = pd.DataFrame([values])[num_cols + cat_cols]
        proba = float(pipe.predict_proba(row)[0, 1])
        pred = int(proba >= threshold)

        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Xác suất churn", f"{proba:.1%}",
                      delta=f"so với tỷ lệ nền {art.get('churn_rate', 0):.1%}", delta_color="off")
            st.progress(min(proba, 1.0))
            if proba < 0.30:
                st.success("🟢 Rủi ro THẤP")
            elif proba < 0.60:
                st.warning("🟡 Rủi ro TRUNG BÌNH")
            else:
                st.error("🔴 Rủi ro CAO")
            st.write(f"Phân loại với ngưỡng θ = {threshold:.2f}: "
                     + ("**CHURN** ⚠️" if pred else "**Ở lại** ✅"))

        with r2:
            st.markdown("**Khách này so với mặt bằng chung** *(đặc trưng quan trọng nhất theo SHAP)*")
            top_shap = art.get("top_shap_features", {})
            top_num = [f for f in top_shap if f in num_cols][:8]
            if top_num:
                comp = pd.DataFrame({
                    "Đặc trưng": [VN_NUM.get(f, f) for f in top_num],
                    "Giá trị nhập": [values[f] for f in top_num],
                    "Trung vị dữ liệu": [num_stats[f]["med"] for f in top_num],
                })
                comp["Chênh lệch"] = comp["Giá trị nhập"] - comp["Trung vị dữ liệu"]
                st.dataframe(comp, hide_index=True)

        with st.expander("📊 Tầm quan trọng đặc trưng của model (SHAP, toàn cục — xem mục 3.5.2)"):
            top_shap = art.get("top_shap_features", {})
            if top_shap:
                imp = pd.Series(top_shap).sort_values()
                fig, ax = plt.subplots(figsize=(7, 4.2))
                imp.plot(kind="barh", ax=ax, color=PALETTE["green"])
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title(f"Top đặc trưng — {art['best_name']}")
                st.pyplot(fig); plt.close(fig)
            else:
                st.caption("Chưa có dữ liệu SHAP trong model artifact.")

# ---------------- Tab 2: hàng loạt ----------------
with tab2:
    st.markdown("Tải lên file CSV chứa **đúng các cột đặc trưng đầu vào** (tính tại cùng mốc T_ref — xem "
                "notebook 04_modeling.ipynb để tự tính cho khách hàng mới). Kết quả trả về kèm "
                "`churn_probability`, `risk_tier` và nhãn dự báo theo ngưỡng θ.")

    feat = load_customer_features()
    if feat is not None:
        demo_cols = [c for c in (["customer_id"] + num_cols + cat_cols) if c in feat.columns]
        sample = feat[demo_cols].dropna().head(5)
        st.download_button("⬇️ Tải file CSV mẫu (tham khảo định dạng cột — KHÔNG dùng số liệu để nộp báo cáo)",
                           sample.to_csv(index=False).encode("utf-8-sig"),
                           file_name="mau_cham_diem_churn.csv", mime="text/csv")
        st.caption("⚠️ File mẫu lấy từ bảng đặc trưng toàn kỳ (dùng cho EDA) chỉ để minh họa đúng tên cột — "
                   "giá trị `recency_days` trong đó KHÔNG cùng mốc thời gian T_ref của model, nên xác suất dự "
                   "báo trên file mẫu này chỉ mang tính minh họa giao diện.")

    up = st.file_uploader("Chọn file CSV", type=["csv"])
    if up is not None:
        batch = pd.read_csv(up)
        missing_cols = [c for c in num_cols + cat_cols if c not in batch.columns]
        if missing_cols:
            st.error(f"File thiếu các cột bắt buộc: `{', '.join(missing_cols)}`")
        else:
            probas = pipe.predict_proba(batch[num_cols + cat_cols])[:, 1]
            out = batch.copy()
            out["churn_probability"] = probas.round(4)
            out["risk_tier"] = pd.cut(out["churn_probability"], bins=[-0.01, 0.3, 0.6, 1.01],
                                      labels=["Thấp", "Trung bình", "Cao"])
            out[f"pred@theta={threshold:.2f}"] = (out["churn_probability"] >= threshold).astype(int)
            out = out.sort_values("churn_probability", ascending=False)

            k1, k2, k3 = st.columns(3)
            k1.metric("Số khách được chấm", f"{len(out):,}")
            k2.metric("Rủi ro CAO (>60%)", f"{(out['risk_tier'] == 'Cao').sum():,}")
            k3.metric("Bị gắn cờ theo ngưỡng θ", f"{int(out[f'pred@theta={threshold:.2f}'].sum()):,}")

            st.dataframe(out.head(30))
            st.download_button("⬇️ Tải toàn bộ kết quả",
                               out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ket_qua_cham_diem_churn.csv", mime="text/csv")
            st.caption("Gợi ý mục 5.2: xuất danh sách nhóm **Cao** cho đội CSKH ưu tiên chăm sóc/ưu đãi giữ chân.")
