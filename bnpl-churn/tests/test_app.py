import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app"
PAGES = [APP / "1_Tổng_quan.py",
         APP / "pages" / "2_Cohort_và_Danh_Mục.py",
         APP / "pages" / "3_Dự_báo_Churn.py"]
TIMEOUT = 120

def _run(path):
    return AppTest.from_file(str(path), default_timeout=TIMEOUT).run()

def _metric(at, label_part):
    for m in at.metric:
        if label_part in m.label:
            return m.value
    raise AssertionError(f"Không thấy metric chứa '{label_part}'")

@pytest.fixture(scope="module")
def artifact():
    return joblib.load(ROOT / "models" / "churn_model_final.pkl")

@pytest.fixture(scope="module")
def model_features():
    return pd.read_csv(ROOT / "data" / "bnpl_model_features.csv")

@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_tc01_pages_start(page):
    at = _run(page)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error

def test_tc02_fpu_filter():
    at = _run(PAGES[0])
    ms = [x for x in at.sidebar.multiselect if x.label == "Phân khúc vòng đời"][0]
    ms.set_value(["FPU"]); at.run()
    assert _metric(at, "Churn rate") == "67.5%"

def test_tc03_empty_filter():
    at = _run(PAGES[0])
    ms = [x for x in at.sidebar.multiselect if x.label == "Danh mục thanh toán chính"][0]
    ms.set_value([]); at.run()
    assert not at.exception
    assert any("không còn khách hàng" in w.value for w in at.warning)

def test_tc04_retention_milestones():
    at = _run(PAGES[1])
    assert _metric(at, "tháng 1") == "68.7%"
    assert _metric(at, "Tháng 3") == "73.5%"
    assert _metric(at, "Tháng 6") == "71.1%"

def test_tc05_default_prediction():
    at = _run(PAGES[2])
    t = time.time(); at.button[0].click(); at.run(); elapsed = time.time() - t
    val = _metric(at, "Xác suất churn")
    p = float(val.strip("%")) / 100
    assert 0.0 <= p <= 1.0
    assert len(at.metric) >= 3
    assert elapsed < 5.0

def test_tc06_monotonic_recency():
    at = _run(PAGES[2])
    at.number_input(key="in_recency_days").set_value(5.0)
    at.number_input(key="in_freq_30d").set_value(3.0)
    at.button[0].click(); at.run()
    low = float(_metric(at, "Xác suất churn").strip("%"))
    at.number_input(key="in_recency_days").set_value(150.0)
    for k in ("in_freq_30d", "in_freq_90d", "in_freq_180d"):
        at.number_input(key=k).set_value(0.0)
    at.button[0].click(); at.run()
    high = float(_metric(at, "Xác suất churn").strip("%"))
    assert high > low

def test_tc07_batch_1000(artifact, model_features):
    pipe, cols = artifact["pipeline"], artifact["num_cols"] + artifact["cat_cols"]
    batch = model_features.head(1000)
    t = time.time(); probas = pipe.predict_proba(batch[cols])[:, 1]; elapsed = time.time() - t
    assert len(probas) == 1000
    assert probas.min() >= 0 and probas.max() <= 1
    assert elapsed < 3.0

def test_tc08_missing_column(artifact, model_features):
    cols = artifact["num_cols"] + artifact["cat_cols"]
    batch = model_features.head(5).drop(columns=["recency_days"])
    missing = [c for c in cols if c not in batch.columns]
    assert missing == ["recency_days"]

def test_tc09_auc_consistency(artifact, model_features):
    pipe, cols = artifact["pipeline"], artifact["num_cols"] + artifact["cat_cols"]
    s = model_features.head(1000)
    auc = roc_auc_score(s["churn"], pipe.predict_proba(s[cols])[:, 1])
    assert auc > 0.95

def test_tc10_threshold(artifact, model_features):
    pipe, cols = artifact["pipeline"], artifact["num_cols"] + artifact["cat_cols"]
    p = pipe.predict_proba(model_features.head(1000)[cols])[:, 1]
    assert (p >= 0.30).sum() > (p >= 0.50).sum()
