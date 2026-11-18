import numpy as np
import pandas as pd
import pytest

import ml
from conftest import LOW, VALUES


@pytest.fixture(scope="module")
def trained():
    return ml.train_model(ml.load_builtin("cleveland"), "test", fast=True, log=lambda *a: None)


def test_builtin_datasets_have_documented_sizes():
    sizes = {k: len(ml.load_builtin(k)) for k in ("cleveland", "hungarian", "switzerland", "va")}
    assert sizes == {"cleveland": 303, "hungarian": 294, "switzerland": 123, "va": 200}   # Hungarian header line is dropped


def test_targets_are_binary_and_zero_means_missing():
    for k in ("cleveland", "hungarian", "switzerland", "va"):
        d = ml.load_builtin(k)
        assert set(d["target"].unique()) <= {0, 1}
        assert not (d["chol"] == 0).any() and not (d["trestbps"] == 0).any()


def test_validate_inputs_accepts_good_and_rejects_bad():
    ok, errs = ml.validate_inputs(VALUES)
    assert errs == [] and ok["age"] == 67
    _, errs = ml.validate_inputs({**VALUES, "age": 5})
    assert any("Age" in e for e in errs)
    _, errs = ml.validate_inputs({**VALUES, "cp": 9})
    assert any("Chest pain" in e for e in errs)
    bad = dict(VALUES); bad.pop("chol")
    assert any("Cholesterol" in e for e in ml.validate_inputs(bad)[1])
    assert ml.validate_inputs({**VALUES, "age": "abc"})[1]
    assert ml.validate_inputs({**VALUES, "age": float("nan")})[1]


def test_training_produces_tree_model_and_sane_thresholds(trained):
    bundle, m = trained
    assert m["model_name"] in ("Decision Tree", "Random Forest", "XGBoost")
    assert 0.05 < bundle["screening_threshold"] <= 0.5
    assert set(m["datasets"]) == {"heldout", "hungarian", "switzerland", "va"}
    assert all(0 <= d["standard"]["recall"] <= 1 for d in m["datasets"].values())


def test_screening_threshold_does_not_lower_recall(trained):
    _, m = trained
    for d in m["datasets"].values():
        assert d["screening"]["recall"] >= d["standard"]["recall"]


def test_high_risk_scores_above_low_risk(trained):
    bundle, _ = trained
    assert ml.predict_proba(bundle, VALUES) > ml.predict_proba(bundle, LOW) + 0.3


def test_shap_contributions_add_up_to_the_prediction(trained):
    bundle, _ = trained
    for v in (VALUES, LOW):
        e = ml.explain(bundle, v)
        assert abs(e["check_pct"] - e["prob"] * 100) < 0.3
        assert len(e["contributions"]) == 13
        assert [abs(c["points"]) for c in e["contributions"]] == sorted((abs(c["points"]) for c in e["contributions"]), reverse=True)


def test_xgboost_explainer_is_in_probability_units(trained):
    from xgboost import XGBClassifier
    bundle, _ = trained
    d = ml.load_builtin("cleveland")
    X = bundle["imputer"].transform(d[ml.FEATURES])
    xg = XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss").fit(X, d["target"])
    ex = ml.make_explainer(xg, "XGBoost", X)
    sv = ml._class1(ex.shap_values(X[:1]))[0]
    assert abs(float(np.atleast_1d(ex.expected_value)[-1]) + sv.sum() - xg.predict_proba(X[:1])[0][1]) < 1e-3


def test_band_edges_follow_threshold():
    assert ml.band_for(0.2, 0.5) == "low" and ml.band_for(0.4, 0.5) == "moderate" and ml.band_for(0.5, 0.5) == "high"
    assert ml.band_for(0.25, 0.3) == "moderate"


def test_similar_patients_and_range_warnings(trained):
    bundle, _ = trained
    s = ml.similar_patients(bundle, VALUES)
    assert s["k"] == 15 and 0 <= s["with_disease"] <= 15
    assert ml.range_warnings(bundle, {**VALUES, "age": 95})
    assert ml.range_warnings(bundle, VALUES) == []


def test_drift_needs_data_then_detects_a_shifted_population(trained):
    bundle, _ = trained
    same = ml.load_builtin("cleveland")[ml.FEATURES].sample(80, random_state=1)
    assert ml.compute_drift(bundle, same.head(3))["status"] == "insufficient"
    assert ml.compute_drift(bundle, same)["status"] in ("ok", "watch")
    shifted = same.copy()
    shifted["age"] += 18; shifted["trestbps"] += 40; shifted["chol"] += 90
    assert ml.compute_drift(bundle, shifted)["status"] == "drift"
