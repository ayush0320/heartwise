import io
import re

import pytest

from conftest import LOW, VALUES

PAGES = ["/", "/predict", "/whatif", "/patients", "/patients/new", "/assessments", "/batch", "/dashboard", "/model", "/learn",
         "/about", "/disclaimer", "/docs", "/admin", "/admin/models", "/admin/system", "/health"]


@pytest.mark.parametrize("path", PAGES)
def test_pages_load(client, path):
    assert client.get(path).status_code == 200


def test_unknown_page_gives_a_friendly_404(client):
    r = client.get("/no-such-page")
    assert r.status_code == 404 and b"404" in r.data and b"Go home" in r.data


def test_csrf_is_enforced(app):
    c = app.test_client()
    assert c.post("/predict", data=VALUES).status_code == 400
    assert c.post("/patients/new", data={"name": "X"}).status_code == 400


def test_predict_flow_saves_and_shows_explanation(client, post):
    r = post("/predict", {**VALUES, "mode": "standard"})
    assert r.status_code == 303
    page = client.get(r.headers["Location"])
    assert page.status_code == 200 and b"Why this estimate" in page.data and b"PDF report" in page.data


def test_predict_rejects_bad_input_with_messages(post):
    r = post("/predict", {**VALUES, "age": "7"})
    assert r.status_code == 400 and b"Age must be between" in r.data


def _make_patient(post, name="Test Person"):
    r = post("/patients/new", {"name": name, "sex": "1", "birth_year": "1970", "notes": "n"})
    assert r.status_code == 303
    return int(re.search(r"/patients/(\d+)", r.headers["Location"]).group(1))


def test_patient_lifecycle(client, post):
    pid = _make_patient(post, "Lifecycle Pat")
    assert b"Lifecycle Pat" in client.get(f"/patients/{pid}").data
    assert post(f"/patients/{pid}/edit", {"name": "Lifecycle Renamed", "sex": "", "birth_year": "", "notes": ""}).status_code == 303
    assert b"Lifecycle Renamed" in client.get("/patients?q=Renamed").data
    assert post(f"/patients/{pid}/delete").status_code == 303
    assert client.get(f"/patients/{pid}").status_code == 404
    assert b"Lifecycle Renamed" not in client.get("/patients?q=Renamed").data


def test_patient_form_validation(post):
    r = post("/patients/new", {"name": "", "birth_year": "3000", "next_checkup": "not-a-date"})
    assert r.status_code == 400 and r.data.count(b"<li>") >= 3


def test_health_log_and_prefill(client, post):
    pid = _make_patient(post, "Log Pat")
    assert post(f"/patients/{pid}/logs", {"systolic_bp": "171", "cholesterol": "260"}).status_code == 303
    assert b'value="171"' in client.get(f"/predict?patient_id={pid}").data              # latest reading prefills the form
    bad = post(f"/patients/{pid}/logs", {"systolic_bp": "9999"}, follow_redirects=True)
    assert b"must be between" in bad.data
    assert b"at least one reading" in post(f"/patients/{pid}/logs", {}, follow_redirects=True).data
    assert client.get(f"/patients/{pid}/export.csv?what=logs").data.startswith(b"date,systolic_bp")


def test_assessment_for_patient_appears_on_their_page_and_in_exports(client, post):
    pid = _make_patient(post, "Assess Pat")
    r = post("/predict", {**VALUES, "mode": "screening", "patient_id": pid})
    aid = int(r.headers["Location"].rsplit("/", 1)[1])
    assert f"/assessments/{aid}".encode() in client.get(f"/patients/{pid}").data
    csv_text = client.get(f"/patients/{pid}/export.csv").data.decode()
    assert csv_text.splitlines()[0].startswith("id,created_at_utc,patient") and "Assess Pat" in csv_text
    assert client.get(f"/assessments?patient_id={pid}&export=csv").data.decode().count("\n") >= 2


def test_followup_sets_a_date(client, post):
    pid = _make_patient(post, "Follow Pat")
    assert post(f"/patients/{pid}/followup", {"days": "90"}).status_code == 303
    assert b"In 89 days" in client.get(f"/patients/{pid}").data or b"In 90 days" in client.get(f"/patients/{pid}").data
    assert post(f"/patients/{pid}/followup", {"days": "7"}).status_code == 400


def test_pdf_report(client, post):
    loc = post("/predict", {**LOW, "mode": "standard"}).headers["Location"]
    r = client.get(loc + "/report.pdf")
    assert r.status_code == 200 and r.mimetype == "application/pdf" and r.data.startswith(b"%PDF") and len(r.data) > 3000


def test_compare_two_assessments(client, post):
    a = post("/predict", {**LOW, "mode": "standard"}).headers["Location"].rsplit("/", 1)[1]
    b = post("/predict", {**VALUES, "mode": "standard"}).headers["Location"].rsplit("/", 1)[1]
    r = client.get(f"/compare?a={a}&b={b}")
    assert r.status_code == 200 and b"CHANGE IN ESTIMATE" in r.data
    assert client.get(f"/compare?a={a}&b={a}").status_code == 302


def test_whatif_live_endpoint(client):
    r = client.post("/live", json=VALUES, headers={"X-CSRF-Token": "test-token"})
    assert r.status_code == 200 and len(r.json["contributions"]) == 13 and 0 <= r.json["pct"] <= 100
    assert client.post("/live", json={**VALUES, "age": 2}, headers={"X-CSRF-Token": "test-token"}).status_code == 400
    assert client.post("/live", json=VALUES).status_code == 400          # no CSRF header


def test_batch_upload(client, post):
    sample = client.get("/batch/sample.csv").data
    assert sample.startswith(b"name,age")
    r = post("/batch", {"mode": "screening", "file": (io.BytesIO(sample), "s.csv")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Sample 1" in r.data and b"Scored" in r.data
    r = post("/batch", {"file": (io.BytesIO(b"age,sex\n1,1\n"), "bad.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"Missing columns" in r.data


def test_history_filters(client, post):
    post("/predict", {**VALUES, "mode": "standard"})
    assert client.get("/assessments?band=high&flagged=yes").status_code == 200
    assert client.get("/assessments?source=api").status_code == 200
    assert client.get("/assessments?page=999").status_code == 200


def test_security_headers_and_escaping(client, post):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    pid = _make_patient(post, "<script>alert(1)</script>")
    body = client.get(f"/patients/{pid}").data
    assert b"<script>alert(1)</script>" not in body and b"&lt;script&gt;" in body
    assert client.get("/patients?q=' OR 1=1 --").status_code == 200


def test_audit_trail_records_actions(client, post):
    _make_patient(post, "Audit Pat")
    assert b"patient.create" in client.get("/admin/system").data
