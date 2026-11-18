import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKEN = "test-token"
VALUES = dict(age=67, sex=1, cp=4, trestbps=160, chol=286, fbs=0, restecg=2, thalach=108, exang=1, oldpeak=1.5, slope=2, ca=3, thal=3)
LOW = dict(age=45, sex=0, cp=3, trestbps=118, chol=190, fbs=0, restecg=0, thalach=172, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3)


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    import core
    from app import create_app
    d = tmp_path_factory.mktemp("heartwise")
    application = create_app({
        "TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{d / 'test.db'}", "MODEL_DIR": str(d / "models"),
        "UPLOAD_DIR": str(d / "uploads"), "FAST_TRAIN": True, "API_RATE_LIMIT": 1000, "READ_ONLY_ADMIN": False})
    with application.app_context():
        core.ensure_model(log=lambda *a: None)
    return application


@pytest.fixture
def client(app):
    c = app.test_client()
    with c.session_transaction() as s:
        s["csrf"] = TOKEN
    return c


@pytest.fixture
def post(client):
    def _post(url, data=None, **kw):
        body = dict(data or {})
        body["_csrf"] = TOKEN
        return client.post(url, data=body, **kw)
    return _post


@pytest.fixture
def api_key(app):
    import api
    with app.app_context():
        return api.create_key("pytest")


@pytest.fixture
def hdr(api_key):
    return {"X-API-Key": api_key}
