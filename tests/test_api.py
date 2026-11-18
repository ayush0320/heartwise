import pytest
from sqlalchemy import select

from conftest import VALUES


def test_health_is_public(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_key_required(client):
    r = client.post("/api/predict", json=VALUES)
    assert r.status_code == 401 and r.json["error"]["code"] == "missing_api_key"
    r = client.post("/api/predict", json=VALUES, headers={"X-API-Key": "nope"})
    assert r.status_code == 401 and r.json["error"]["code"] == "invalid_api_key"


def test_predict_returns_a_saved_assessment(client, hdr, app):
    from models import Assessment, db
    r = client.post("/api/predict", json=VALUES, headers=hdr)
    assert r.status_code == 200
    j = r.json
    assert 0 <= j["probability"] <= 1 and j["risk_band"] in ("low", "moderate", "high") and "disclaimer" in j
    with app.app_context():
        a = db.session.get(Assessment, j["id"])
        assert a.source == "api" and a.age == 67


def test_predict_screening_mode_uses_lower_threshold(client, hdr):
    std = client.post("/api/predict", json=VALUES, headers=hdr).json
    scr = client.post("/api/predict", json={**VALUES, "mode": "screening"}, headers=hdr).json
    assert scr["threshold"] < std["threshold"] == 0.5


def test_validation_errors_are_reported(client, hdr):
    bad = dict(VALUES); bad.pop("age"); bad["chol"] = 5000
    r = client.post("/api/predict", json=bad, headers=hdr)
    assert r.status_code == 400 and r.json["error"]["code"] == "validation_error" and len(r.json["error"]["details"]) == 2
    assert client.post("/api/predict", data="not json", headers=hdr).status_code == 400
    assert client.post("/api/predict", json={**VALUES, "mode": "wild"}, headers=hdr).status_code == 400


def test_explain_contributions_add_up(client, hdr):
    j = client.post("/api/explain", json=VALUES, headers=hdr).json
    assert len(j["contributions"]) == 13 and j["plain_language"]
    assert abs(j["base_percent"] + sum(c["points"] for c in j["contributions"]) - j["percent"]) < 0.3


def test_batch_scores_valid_rows_and_reports_bad_ones(client, hdr):
    rows = [VALUES, {**VALUES, "age": 1}, {**VALUES, "age": 50}]
    j = client.post("/api/batch", json={"patients": rows}, headers=hdr).json
    assert j["count"] == 3 and j["scored"] == 2
    assert j["results"][0]["ok"] and not j["results"][1]["ok"] and j["results"][2]["ok"]
    assert client.post("/api/batch", json={"patients": [VALUES] * 201}, headers=hdr).status_code == 400
    assert client.post("/api/batch", json={"nope": 1}, headers=hdr).status_code == 400


def test_rate_limit_returns_429_with_retry_after(app, client):
    import api
    with app.app_context():
        key = api.create_key("limited")
    old = app.config["API_RATE_LIMIT"]
    app.config["API_RATE_LIMIT"] = 3
    try:
        codes = [client.get("/api/model", headers={"X-API-Key": key}) for _ in range(5)]
    finally:
        app.config["API_RATE_LIMIT"] = old
    assert [c.status_code for c in codes] == [200, 200, 200, 429, 429]
    assert int(codes[3].headers["Retry-After"]) >= 1


def test_revoked_key_stops_working(app, client):
    import api
    from models import ApiKey, db
    with app.app_context():
        key = api.create_key("temp")
        k = db.session.scalar(select(ApiKey).where(ApiKey.key_hash == api.hash_key(key)))
        k.revoked = True
        db.session.commit()
    assert client.get("/api/model", headers={"X-API-Key": key}).status_code == 401


def test_keys_are_stored_hashed(app, api_key):
    from models import ApiKey, db
    with app.app_context():
        assert db.session.scalar(select(ApiKey).where(ApiKey.key_hash == api_key)) is None


def test_model_assessments_and_openapi(client, hdr):
    m = client.get("/api/model", headers=hdr).json
    assert m["thresholds"]["standard"] == 0.5 and "other_hospitals" in m
    client.post("/api/predict", json=VALUES, headers=hdr)
    a = client.get("/api/assessments?limit=5", headers=hdr).json
    assert 1 <= a["count"] <= 5
    spec = client.get("/api/openapi.json").json
    assert "/api/predict" in spec["paths"] and spec["openapi"].startswith("3.")
    assert client.get("/api/does-not-exist").json["error"]["code"] == "not_found"
