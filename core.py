"""Service layer: model loading, analysis, saving assessments, audit trail."""
import json
import os

import joblib
import pandas as pd
from flask import current_app, has_request_context, request
from sqlalchemy import func, select

import advice
import ml
from models import Assessment, AuditLog, ModelVersion, db, utcnow

_BUNDLES = {}


class NoModelError(RuntimeError):
    pass


def audit(action, detail=""):
    ip = request.remote_addr if has_request_context() else "cli"
    db.session.add(AuditLog(action=action, detail=str(detail)[:500], ip=ip))
    db.session.commit()


def active_version():
    return db.session.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))


def get_bundle(version):
    if version.id not in _BUNDLES:
        _BUNDLES[version.id] = joblib.load(os.path.join(current_app.config["MODEL_DIR"], version.file_name))
    return _BUNDLES[version.id]


def active():
    v = active_version()
    if v is None:
        raise NoModelError("No active model. Run `python manage.py init` first.")
    return v, get_bundle(v)


def clear_cache():
    _BUNDLES.clear()


def threshold_for(bundle, mode):
    return bundle["screening_threshold"] if mode == "screening" else 0.5


def analyze(values, mode="standard", full=True):
    """Everything the result page needs for one patient (no database writes)."""
    version, bundle = active()
    t = threshold_for(bundle, mode)
    e = ml.explain(bundle, values)
    p = e["prob"]
    out = {
        "prob": p, "pct": round(p * 100, 1), "threshold": t, "mode": mode,
        "band": ml.band_for(p, t), "flagged": p >= t, "edges": list(ml.band_edges(t)),
        "base_pct": e["base_pct"], "check_pct": e["check_pct"], "contributions": e["contributions"],
        "warnings": ml.range_warnings(bundle, values),
        "screening_threshold": bundle["screening_threshold"],
        "model": {"id": version.id, "name": version.name, "algorithm": version.model_name},
    }
    if full:
        out["sentences"] = advice.plain_sentences(e["contributions"])
        out["advice"] = advice.build_advice(e["contributions"])
        out["similar"] = ml.similar_patients(bundle, values)
    return out


def save_assessment(result, values, patient_id=None, source="web", is_demo=False, created_at=None):
    a = Assessment(
        patient_id=patient_id, source=source, mode=result["mode"], threshold=result["threshold"],
        probability=result["prob"], band=result["band"], flagged=bool(result["flagged"]),
        model_version_id=result["model"]["id"], shap_json=json.dumps(result["contributions"]),
        is_demo=is_demo, created_at=created_at or utcnow(), **{f: values[f] for f in ml.FEATURES})
    db.session.add(a)
    db.session.commit()
    return a


def train_and_register(df, source, activate=False, fast=None, log=print):
    """Train a new model version from a labelled frame and store it."""
    fast = current_app.config["FAST_TRAIN"] if fast is None else fast
    bundle, metrics = ml.train_model(df, source, fast=fast, log=log)
    os.makedirs(current_app.config["MODEL_DIR"], exist_ok=True)
    n = (db.session.scalar(select(func.max(ModelVersion.id))) or 0) + 1
    file_name = f"model_v{n}.pkl"
    joblib.dump(bundle, os.path.join(current_app.config["MODEL_DIR"], file_name))
    held = metrics["datasets"]["heldout"]["standard"]
    best = metrics["comparison"][metrics["model_name"]]
    mv = ModelVersion(
        name=f"v{n}", source=source, n_train=metrics["n_train"], n_test=metrics["n_test"],
        model_name=metrics["model_name"], cv_auc=best["cv_auc"], test_acc=held["accuracy"],
        test_recall=held["recall"], test_auc=held["roc_auc"], screening_threshold=bundle["screening_threshold"],
        file_name=file_name, metrics_json=json.dumps(metrics), is_active=False)
    db.session.add(mv)
    db.session.commit()
    if activate:
        activate_version(mv)
    return mv


def activate_version(mv):
    for other in db.session.scalars(select(ModelVersion)).all():
        other.is_active = other.id == mv.id
    db.session.commit()
    clear_cache()


def ensure_model(log=print):
    """Train the first model from the built-in Cleveland data if none exists."""
    if active_version() is not None:
        return active_version()
    log("No model yet - training the first one from the built-in Cleveland data (first run only).")
    return train_and_register(ml.load_builtin("cleveland"), "Cleveland (built-in)", activate=True, log=log)


def recent_inputs(limit=300):
    rows = db.session.scalars(select(Assessment).order_by(Assessment.created_at.desc()).limit(limit)).all()
    return pd.DataFrame([r.inputs for r in rows], columns=ml.FEATURES)


def drift_report():
    _, bundle = active()
    return ml.compute_drift(bundle, recent_inputs())


def result_from_assessment(a, with_similar=True):
    """Rebuild the full result dict for a saved assessment (explains it on demand if needed)."""
    mv = a.model_version
    contribs = a.contributions
    bundle = get_bundle(mv) if mv is not None else None
    if not contribs and bundle is not None:          # batch rows are saved without an explanation
        contribs = ml.explain(bundle, a.inputs)["contributions"]
        a.shap_json = json.dumps(contribs)
        db.session.commit()
    total = sum(c["points"] for c in contribs)
    out = {
        "prob": a.probability, "pct": a.pct, "threshold": a.threshold, "mode": a.mode, "band": a.band,
        "flagged": bool(a.flagged), "edges": list(ml.band_edges(a.threshold)), "base_pct": round(a.pct - total, 1),
        "contributions": contribs, "sentences": advice.plain_sentences(contribs), "advice": advice.build_advice(contribs),
        "warnings": ml.range_warnings(bundle, a.inputs) if bundle else [],
        "model": {"id": a.model_version_id, "name": mv.name if mv else "?", "algorithm": mv.model_name if mv else "?"},
        "screening_threshold": bundle["screening_threshold"] if bundle else a.threshold,
    }
    if with_similar and bundle is not None:
        out["similar"] = ml.similar_patients(bundle, a.inputs)
    return out
