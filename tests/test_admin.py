import io

import pandas as pd
import pytest
from sqlalchemy import func, select

import ml


def _csv_bytes(n=240):
    d = pd.concat([ml.load_builtin("cleveland")]).head(n)
    return d.to_csv(index=False).encode()


def test_retrain_then_activate_then_switch_back(app, client, post):
    from models import ModelVersion, db
    with app.app_context():
        first = db.session.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True))).id
    r = post("/admin/models/retrain", {"file": (io.BytesIO(_csv_bytes()), "mine.csv")}, content_type="multipart/form-data")
    assert r.status_code == 303
    with app.app_context():
        new = db.session.scalar(select(ModelVersion).order_by(ModelVersion.id.desc()))
        assert new.id != first and not new.is_active and new.source.startswith("Upload")
        new_id = new.id
    assert b"mine.csv" in client.get("/admin/models").data
    assert post(f"/admin/models/{new_id}/activate").status_code == 303
    with app.app_context():
        assert db.session.get(ModelVersion, new_id).is_active
        assert db.session.scalar(select(func.count(ModelVersion.id)).where(ModelVersion.is_active.is_(True))) == 1
    assert client.get("/model").status_code == 200                       # pages still work with the new version
    post(f"/admin/models/{first}/activate")
    with app.app_context():
        assert db.session.get(ModelVersion, first).is_active


def test_retrain_rejects_bad_files(client, post):
    r = post("/admin/models/retrain", {"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data", follow_redirects=True)
    assert b"Missing columns" in r.data
    tiny = ml.load_builtin("cleveland").head(30).to_csv(index=False).encode()
    r = post("/admin/models/retrain", {"file": (io.BytesIO(tiny), "t.csv")}, content_type="multipart/form-data", follow_redirects=True)
    assert b"at least 100" in r.data


def test_read_only_mode_blocks_admin_writes(app, client, post):
    from models import ModelVersion, db
    with app.app_context():
        before = db.session.scalar(select(func.count(ModelVersion.id)))
    app.config["READ_ONLY_ADMIN"] = True
    try:
        r = post("/admin/models/retrain", {"file": (io.BytesIO(_csv_bytes()), "m.csv")}, content_type="multipart/form-data", follow_redirects=True)
        assert b"read-only" in r.data
        assert post("/admin/keys", {"name": "x"}).status_code == 303
        assert post("/admin/demo", {"action": "load"}).status_code == 303
    finally:
        app.config["READ_ONLY_ADMIN"] = False
    with app.app_context():
        assert db.session.scalar(select(func.count(ModelVersion.id))) == before


def test_admin_can_create_a_key_that_works_once_shown(client, post):
    r = post("/admin/keys", {"name": "from admin"}, follow_redirects=True)
    assert b"shown only once" in r.data
    key = r.data.decode().split('class="keybox')[1].split(">")[1].split("<")[0].strip()
    assert key.startswith("hw_")
    assert client.get("/api/model", headers={"X-API-Key": key}).status_code == 200
    assert b"shown only once" not in client.get("/admin/system").data      # not shown a second time


def test_demo_data_loads_and_clears(app, client, post):
    from models import Patient, db
    post("/admin/demo", {"action": "load"})
    with app.app_context():
        assert db.session.scalar(select(func.count(Patient.id)).where(Patient.is_demo.is_(True))) == 8
    assert b"Asha Verma" in client.get("/patients").data and b"Demo" in client.get("/patients").data
    assert client.get("/dashboard").status_code == 200 and client.get("/admin").status_code == 200
    post("/admin/demo", {"action": "clear"})
    with app.app_context():
        assert db.session.scalar(select(func.count(Patient.id)).where(Patient.is_demo.is_(True))) == 0


def test_drift_view_reacts_to_shifted_inputs(client, hdr):
    base = dict(age=80, sex=1, cp=4, trestbps=190, chol=450, fbs=1, restecg=2, thalach=80, exang=1, oldpeak=4, slope=3, ca=3, thal=7)
    for _ in range(12):
        client.post("/api/predict", json=base, headers=hdr)
    page = client.get("/admin").data
    assert b"Drift detected" in page or b"Watch" in page
