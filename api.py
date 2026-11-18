"""REST API. Endpoints register themselves, and the docs page + OpenAPI file are generated from that registry."""
import hashlib
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import wraps

import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import select

import core
import ml
from models import ApiKey, Assessment, db, utcnow

bp = Blueprint("api", __name__, url_prefix="/api")
REGISTRY = []
_hits = defaultdict(deque)          # api key id -> timestamps of recent requests
DISCLAIMER = "Educational screening aid. Not a medical diagnosis."
EXAMPLE_BODY = {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
                "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3, "mode": "standard"}


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_key(name: str) -> str:
    """Create a key and return it. Only the hash is stored, so this is the one time it can be shown."""
    raw = "hw_" + secrets.token_urlsafe(24)
    db.session.add(ApiKey(name=name[:80] or "key", prefix=raw[:8], key_hash=hash_key(raw)))
    db.session.commit()
    return raw


def error(status, code, message, details=None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status


def _check_key():
    raw = request.headers.get("X-API-Key", "").strip()
    if not raw:
        return error(401, "missing_api_key", "Send your key in the X-API-Key header.")
    key = db.session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(raw), ApiKey.revoked.is_(False)))
    if key is None:
        return error(401, "invalid_api_key", "This API key is not valid or was revoked.")
    now, window = time.time(), _hits[key.id]
    while window and now - window[0] > 60:
        window.popleft()
    limit = current_app.config["API_RATE_LIMIT"]
    if len(window) >= limit:
        resp, _ = error(429, "rate_limited", f"Limit is {limit} requests per minute for each key.")
        resp.headers["Retry-After"] = str(max(1, int(60 - (now - window[0]))))
        return resp, 429
    window.append(now)
    key.request_count = (key.request_count or 0) + 1
    key.last_used_at = utcnow()
    db.session.commit()
    return None


def endpoint(method, path, summary, description="", auth=True, request_example=None, response_example=None):
    def deco(fn):
        REGISTRY.append(dict(method=method, path="/api" + path, summary=summary, description=description, auth=auth,
                             request_example=request_example, response_example=response_example))

        @wraps(fn)
        def wrapper(*a, **k):
            if auth:
                err = _check_key()
                if err:
                    return err
            return fn(*a, **k)

        bp.add_url_rule(path, endpoint=fn.__name__, view_func=wrapper, methods=[method])
        return wrapper
    return deco


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def _prepare(data):
    mode = data.get("mode", "standard")
    if mode not in ("standard", "screening"):
        return None, None, error(400, "validation_error", "mode must be 'standard' or 'screening'.")
    values, errs = ml.validate_inputs(data)
    if errs:
        return None, None, error(400, "validation_error", "Some inputs are missing or out of range.", errs)
    return values, mode, None


def _summary(res, a):
    return {"id": a.id, "probability": round(res["prob"], 4), "percent": res["pct"], "risk_band": res["band"],
            "flagged": res["flagged"], "threshold": res["threshold"], "mode": res["mode"],
            "model_version": res["model"]["name"], "warnings": res["warnings"], "disclaimer": DISCLAIMER}


@endpoint("GET", "/health", "Service health", "Public. Use it for uptime checks.", auth=False,
          response_example={"status": "ok", "version": "1.0.0", "model": "v1"})
def health():
    v = core.active_version()
    return jsonify(status="ok", version=current_app.config["APP_VERSION"], model=v.name if v else None,
                   time=datetime.now(timezone.utc).isoformat(timespec="seconds"))


@endpoint("GET", "/model", "Active model details", "Algorithm, training size, thresholds and test metrics of the active model.",
          response_example={"version": "v1", "algorithm": "Random Forest", "thresholds": {"standard": 0.5, "screening": 0.31}})
def model_info():
    v = core.active_version()
    if v is None:
        return error(503, "no_model", "No active model.")
    m = v.metrics
    return jsonify(version=v.name, algorithm=v.model_name, trained_at=v.created_at.isoformat(timespec="seconds"),
                   source=v.source, n_train=v.n_train, n_test=v.n_test, thresholds=m["thresholds"],
                   test={"accuracy": v.test_acc, "recall": v.test_recall, "roc_auc": v.test_auc},
                   other_hospitals={k: d["standard"] for k, d in m["datasets"].items() if k != "heldout"})


@endpoint("POST", "/predict", "Predict heart-disease likelihood",
          "Send the 13 inputs as JSON. `mode` is optional: 'standard' flags at 50%, 'screening' uses a lower threshold to miss fewer patients.",
          request_example=EXAMPLE_BODY,
          response_example={"id": 12, "probability": 0.916, "percent": 91.6, "risk_band": "high", "flagged": True, "threshold": 0.5,
                            "mode": "standard", "model_version": "v1", "warnings": [], "disclaimer": DISCLAIMER})
def predict():
    data = _body()
    if data is None:
        return error(400, "invalid_json", "Send a JSON object.")
    values, mode, err = _prepare(data)
    if err:
        return err
    res = core.analyze(values, mode, full=False)
    a = core.save_assessment(res, values, source="api")
    return jsonify(_summary(res, a))


@endpoint("POST", "/explain", "Predict and explain with SHAP",
          "Same input as /predict. Adds each input's contribution to the estimate, in percentage points, plus plain-language sentences.",
          request_example=EXAMPLE_BODY,
          response_example={"percent": 91.6, "risk_band": "high", "base_percent": 46.3,
                            "contributions": [{"feature": "ca", "label": "Major vessels", "value": 3.0, "points": 12.8}],
                            "plain_language": ["Major vessels (3 vessels) raised the estimate by 12.8 points."]})
def explain_endpoint():
    data = _body()
    if data is None:
        return error(400, "invalid_json", "Send a JSON object.")
    values, mode, err = _prepare(data)
    if err:
        return err
    res = core.analyze(values, mode, full=True)
    a = core.save_assessment(res, values, source="api")
    out = _summary(res, a)
    out.update(base_percent=res["base_pct"], plain_language=res["sentences"],
               contributions=[{"feature": c["feature"], "label": c["label"], "value": c["value"], "points": c["points"]} for c in res["contributions"]])
    return jsonify(out)


@endpoint("POST", "/batch", "Score up to 200 patients at once",
          "Body: {\"patients\": [ {...13 inputs...}, ... ], \"mode\": \"standard\"}. Invalid rows are reported by index; valid rows are still scored.",
          request_example={"patients": [EXAMPLE_BODY], "mode": "standard"},
          response_example={"count": 1, "scored": 1, "results": [{"index": 0, "ok": True, "percent": 91.6, "risk_band": "high", "flagged": True}]})
def batch():
    data = _body()
    if data is None or not isinstance(data.get("patients"), list):
        return error(400, "invalid_json", "Send {\"patients\": [ ... ]}.")
    rows = data["patients"]
    if not 1 <= len(rows) <= 200:
        return error(400, "validation_error", "Send between 1 and 200 patients.")
    mode = data.get("mode", "standard")
    if mode not in ("standard", "screening"):
        return error(400, "validation_error", "mode must be 'standard' or 'screening'.")
    version, bundle = core.active()
    t = core.threshold_for(bundle, mode)
    ok_idx, ok_vals, results = [], [], [None] * len(rows)
    for i, r in enumerate(rows):
        values, errs = ml.validate_inputs(r if isinstance(r, dict) else {})
        if errs:
            results[i] = {"index": i, "ok": False, "errors": errs}
        else:
            ok_idx.append(i)
            ok_vals.append(values)
    if ok_vals:
        probs = ml.predict_many(bundle, pd.DataFrame(ok_vals))
        for i, values, p in zip(ok_idx, ok_vals, probs):
            band = ml.band_for(float(p), t)
            db.session.add(Assessment(source="batch", mode=mode, threshold=t, probability=float(p), band=band, flagged=bool(p >= t),
                                      model_version_id=version.id, shap_json="[]", **values))
            results[i] = {"index": i, "ok": True, "probability": round(float(p), 4), "percent": round(float(p) * 100, 1),
                          "risk_band": band, "flagged": bool(p >= t)}
        db.session.commit()
    return jsonify(count=len(rows), scored=len(ok_vals), threshold=t, mode=mode, model_version=version.name, results=results)


@endpoint("GET", "/assessments", "Recent assessments", "Query: `limit` (1-100, default 20) and `flagged=true|false`.",
          response_example={"count": 1, "items": [{"id": 12, "created_at": "2026-10-27T10:00:00", "percent": 91.6, "risk_band": "high", "flagged": True, "source": "api"}]})
def assessments():
    limit = min(max(request.args.get("limit", 20, type=int), 1), 100)
    q = select(Assessment).order_by(Assessment.created_at.desc()).limit(limit)
    flagged = request.args.get("flagged")
    if flagged in ("true", "false"):
        q = q.where(Assessment.flagged.is_(flagged == "true"))
    items = [{"id": a.id, "created_at": a.created_at.isoformat(timespec="seconds"), "percent": a.pct, "risk_band": a.band,
              "flagged": a.flagged, "mode": a.mode, "source": a.source, "patient_id": a.patient_id} for a in db.session.scalars(q)]
    return jsonify(count=len(items), items=items)


def openapi_spec():
    paths = {}
    for e in REGISTRY:
        op = {"summary": e["summary"], "description": e["description"], "responses": {"200": {"description": "OK"}}}
        if e["response_example"] is not None:
            op["responses"]["200"]["content"] = {"application/json": {"example": e["response_example"]}}
        if e["auth"]:
            op["security"] = [{"ApiKeyAuth": []}]
        if e["request_example"] is not None:
            op["requestBody"] = {"required": True, "content": {"application/json": {"example": e["request_example"]}}}
        paths.setdefault(e["path"], {})[e["method"].lower()] = op
    return {"openapi": "3.0.3", "info": {"title": "HeartWise API", "version": current_app.config["APP_VERSION"],
            "description": "Explainable heart-disease risk estimates. " + DISCLAIMER}, "paths": paths,
            "components": {"securitySchemes": {"ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}}}


@endpoint("GET", "/openapi.json", "OpenAPI specification", "Machine-readable description of this API, generated from the code.", auth=False)
def openapi():
    return jsonify(openapi_spec())
