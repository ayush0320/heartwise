"""Pages: everything the browser sees."""
import csv
import io
import math
from datetime import date, timedelta
from functools import wraps

import pandas as pd
from flask import (Blueprint, Response, abort, current_app, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from sqlalchemy import func, select

import advice
import api
import core
import ml
import reports
import seed
from models import ApiKey, Assessment, AuditLog, HealthLog, ModelVersion, Patient, db, utcnow

bp = Blueprint("web", __name__)
FOLLOW_UP_DAYS = {"high": 30, "moderate": 90, "low": 180}


# ------------------------------------------------------------------------ helpers
class Page:
    """Tiny pagination helper for lists that are filtered in Python."""

    def __init__(self, items, page, per_page):
        self.total = len(items)
        self.per_page = per_page
        self.pages = max(1, math.ceil(self.total / per_page))
        self.page = min(max(page, 1), self.pages)
        self.items = items[(self.page - 1) * per_page: self.page * per_page]
        self.has_prev, self.has_next = self.page > 1, self.page < self.pages
        self.prev_num, self.next_num = self.page - 1, self.page + 1


def admin_write(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if current_app.config["READ_ONLY_ADMIN"]:
            flash("This deployment is read-only for admin actions.", "warn")
            return redirect(request.referrer or url_for("web.admin_overview"), code=303)
        return fn(*a, **k)
    return wrapper


def default_values():
    return {f: s["default"] for f, s in ml.FIELD_SPECS.items()}


def get_patient(pid):
    p = db.session.get(Patient, pid)
    if p is None or p.deleted_at is not None:
        abort(404)
    return p


def active_patients():
    return db.session.scalars(select(Patient).where(Patient.deleted_at.is_(None)).order_by(Patient.name)).all()


def prefill_for(patient):
    """Starting values for a patient: last assessment, overridden by newer health-log readings."""
    vals = default_values()
    if patient is None:
        return vals
    if patient.sex is not None:
        vals["sex"] = patient.sex
    if patient.age:
        vals["age"] = patient.age
    last = patient.latest
    if last:
        vals.update({f: v for f, v in last.inputs.items() if v is not None})
    since = last.created_at.date() if last else date.min
    for attr, field in (("systolic_bp", "trestbps"), ("cholesterol", "chol")):
        log = next((l for l in patient.logs if getattr(l, attr) is not None), None)   # logs are newest first
        if log and log.logged_on >= since:
            vals[field] = getattr(log, attr)
    return vals


def csv_response(rows, header, filename):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    core.audit("export.csv", filename)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


def parse_date(text):
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return "bad"


def assessment_rows(items):
    header = ["id", "created_at_utc", "patient", "source", "mode", "threshold", "probability", "band", "flagged", *ml.FEATURES]
    rows = [[a.id, a.created_at.isoformat(timespec="seconds"), a.patient.name if a.patient else "", a.source, a.mode, a.threshold,
             round(a.probability, 4), a.band, int(bool(a.flagged)), *[a.inputs[f] for f in ml.FEATURES]] for a in items]
    return header, rows


def result_json(a, res):
    return {"pct": res["pct"], "edges": res["edges"], "threshold": res["threshold"], "band": res["band"], "flagged": res["flagged"],
            "base_pct": res["base_pct"], "contributions": res["contributions"]}


# --------------------------------------------------------------------------- core pages
@bp.route("/")
def home():
    v = core.active_version()
    m = v.metrics if v else {}
    return render_template("home.html", v=v, m=m, n_patients=db.session.scalar(select(func.count(Patient.id)).where(Patient.deleted_at.is_(None))),
                           n_assess=db.session.scalar(select(func.count(Assessment.id))))


@bp.route("/predict", methods=["GET", "POST"])
def predict():
    patients = active_patients()
    mode = request.values.get("mode", "standard")
    if request.method == "POST":
        pid_raw = request.form.get("patient_id", "")
        patient = get_patient(int(pid_raw)) if pid_raw.isdigit() else None
        values, errors = ml.validate_inputs(request.form)
        if mode not in ("standard", "screening"):
            errors.append("Mode must be standard or screening.")
        if errors:
            return render_template("predict.html", patients=patients, selected=patient, values=dict(request.form), errors=errors,
                                   mode=mode, specs=ml.FIELD_SPECS, choices=ml.CHOICES), 400
        res = core.analyze(values, mode, full=False)
        a = core.save_assessment(res, values, patient_id=patient.id if patient else None)
        core.audit("assessment.create", f"#{a.id} {res['pct']}% {res['band']}" + (f" patient {patient.id}" if patient else ""))
        return redirect(url_for("web.assessment_view", aid=a.id), code=303)

    patient, values = None, default_values()
    src = request.args.get("from_assessment", type=int)
    if src and (old := db.session.get(Assessment, src)):
        values = {f: v for f, v in old.inputs.items()}
        mode = old.mode
        patient = old.patient if old.patient and old.patient.deleted_at is None else None
    elif request.args.get("patient_id", type=int):
        patient = get_patient(request.args.get("patient_id", type=int))
        values = prefill_for(patient)
    return render_template("predict.html", patients=patients, selected=patient, values=values, errors=[], mode=mode,
                           specs=ml.FIELD_SPECS, choices=ml.CHOICES)


@bp.route("/assessments/<int:aid>")
def assessment_view(aid):
    a = db.session.get(Assessment, aid) or abort(404)
    res = core.result_from_assessment(a)
    follow = FOLLOW_UP_DAYS.get(a.band) if a.patient and a.patient.deleted_at is None else None
    return render_template("result.html", a=a, res=res, data=result_json(a, res), follow=follow, links=advice.LINKS,
                           inputs=[(ml.LABELS[f], ml.value_text(f, a.inputs[f])) for f in ml.FEATURES])


@bp.route("/assessments/<int:aid>/report.pdf")
def assessment_pdf(aid):
    a = db.session.get(Assessment, aid) or abort(404)
    pdf = reports.build_report(a, a.patient, core.result_from_assessment(a, with_similar=False))
    core.audit("report.pdf", f"#{a.id}")
    return Response(pdf, mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename=heartwise-report-{a.id}.pdf"})


@bp.route("/assessments")
def history():
    q = select(Assessment).order_by(Assessment.created_at.desc())
    f = {k: request.args.get(k, "") for k in ("patient_id", "flagged", "band", "source")}
    if f["patient_id"].isdigit():
        q = q.where(Assessment.patient_id == int(f["patient_id"]))
    if f["flagged"] in ("yes", "no"):
        q = q.where(Assessment.flagged.is_(f["flagged"] == "yes"))
    if f["band"] in ("low", "moderate", "high"):
        q = q.where(Assessment.band == f["band"])
    if f["source"] in ("web", "api", "batch"):
        q = q.where(Assessment.source == f["source"])
    if request.args.get("export") == "csv":
        return csv_response(*reversed(assessment_rows(db.session.scalars(q.limit(5000)).all())), "heartwise-assessments.csv")
    page = db.paginate(q, page=request.args.get("page", 1, type=int), per_page=current_app.config["PER_PAGE"], error_out=False)
    return render_template("history.html", page=page, f=f, patients=active_patients())


@bp.route("/compare")
def compare():
    ids = [request.args.get("a", type=int), request.args.get("b", type=int)]
    if None in ids or ids[0] == ids[1]:
        flash("Pick two different assessments to compare.", "warn")
        return redirect(url_for("web.history"))
    A, B = (db.session.get(Assessment, i) or abort(404) for i in ids)
    ra, rb = core.result_from_assessment(A, False), core.result_from_assessment(B, False)
    pa, pb = ({c["feature"]: c["points"] for c in r["contributions"]} for r in (ra, rb))
    deltas = sorted(({"feature": f, "label": ml.LABELS[f], "points": round(pb.get(f, 0) - pa.get(f, 0), 2),
                      "value_text": (f"{ml.value_text(f, A.inputs[f])} to {ml.value_text(f, B.inputs[f])}" if A.inputs[f] != B.inputs[f]
                                     else f"{ml.value_text(f, A.inputs[f])} (unchanged)")} for f in ml.FEATURES),
                    key=lambda d: -abs(d["points"]))
    rows = [(ml.LABELS[f], ml.value_text(f, A.inputs[f]), ml.value_text(f, B.inputs[f]), A.inputs[f] != B.inputs[f]) for f in ml.FEATURES]
    return render_template("compare.html", A=A, B=B, ra=ra, rb=rb, rows=rows, delta=round(B.pct - A.pct, 1),
                           data={"a": result_json(A, ra), "b": result_json(B, rb), "deltas": deltas},
                           same_model=A.model_version_id == B.model_version_id)


# ------------------------------------------------------------------------- what-if
@bp.route("/whatif")
def whatif():
    values, mode, label = default_values(), "standard", None
    src = request.args.get("from", type=int)
    if src and (old := db.session.get(Assessment, src)):
        values, mode, label = dict(old.inputs), old.mode, f"assessment #{old.id}"
    return render_template("whatif.html", values=values, mode=mode, label=label, specs=ml.FIELD_SPECS, choices=ml.CHOICES,
                           limits=ml.LIMITS)


@bp.route("/live", methods=["POST"])
def live():
    data = request.get_json(silent=True) or {}
    values, errs = ml.validate_inputs(data)
    mode = data.get("mode", "standard")
    if errs or mode not in ("standard", "screening"):
        return jsonify(error=errs or ["Invalid mode."]), 400
    res = core.analyze(values, mode, full=False)
    return jsonify(pct=res["pct"], band=res["band"], flagged=res["flagged"], threshold=res["threshold"], edges=res["edges"],
                   base_pct=res["base_pct"], contributions=res["contributions"], warnings=res["warnings"])


# ---------------------------------------------------------------------------- patients
@bp.route("/patients")
def patients():
    q = select(Patient).where(Patient.deleted_at.is_(None)).order_by(Patient.name)
    term = request.args.get("q", "").strip()
    if term:
        q = q.where(Patient.name.ilike(f"%{term}%"))
    items = db.session.scalars(q).all()
    band, due = request.args.get("band", ""), request.args.get("due", "")
    if band in ("low", "moderate", "high"):
        items = [p for p in items if p.latest and p.latest.band == band]
    if band == "none":
        items = [p for p in items if not p.latest]
    today = date.today()
    if due == "1":
        items = [p for p in items if p.next_checkup and p.next_checkup <= today + timedelta(days=7)]
    page = Page(items, request.args.get("page", 1, type=int), current_app.config["PER_PAGE"])
    return render_template("patients.html", page=page, q=term, band=band, due=due, today=today)


def _patient_form(patient=None):
    f = request.form
    errors, name = [], f.get("name", "").strip()
    if not name or len(name) > 120:
        errors.append("Name is required (up to 120 characters).")
    sex = f.get("sex", "")
    if sex not in ("", "0", "1"):
        errors.append("Sex must be male, female or left blank.")
    by = f.get("birth_year", "").strip()
    if by and not (by.isdigit() and 1900 <= int(by) <= date.today().year):
        errors.append("Birth year must be a valid year.")
    nxt = parse_date(f.get("next_checkup", ""))
    if nxt == "bad":
        errors.append("Next check-up must be a valid date.")
    if errors:
        return render_template("patient_form.html", patient=patient, form=f, errors=errors), 400
    p = patient or Patient()
    p.name, p.sex = name, (int(sex) if sex != "" else None)
    p.birth_year = int(by) if by else None
    p.notes = f.get("notes", "").strip()[:1000]
    p.next_checkup = nxt
    if patient is None:
        db.session.add(p)
    db.session.commit()
    core.audit("patient.update" if patient else "patient.create", f"#{p.id} {p.name}")
    flash("Patient saved.", "ok")
    return redirect(url_for("web.patient_detail", pid=p.id), code=303)


@bp.route("/patients/new", methods=["GET", "POST"])
def patient_new():
    if request.method == "POST":
        return _patient_form()
    return render_template("patient_form.html", patient=None, form={}, errors=[])


@bp.route("/patients/<int:pid>/edit", methods=["GET", "POST"])
def patient_edit(pid):
    p = get_patient(pid)
    if request.method == "POST":
        return _patient_form(p)
    return render_template("patient_form.html", patient=p, errors=[], form={
        "name": p.name, "sex": "" if p.sex is None else str(p.sex), "birth_year": p.birth_year or "", "notes": p.notes,
        "next_checkup": p.next_checkup.isoformat() if p.next_checkup else ""})


@bp.route("/patients/<int:pid>")
def patient_detail(pid):
    p = get_patient(pid)
    asc = sorted(p.assessments, key=lambda a: a.created_at)
    logs = sorted(p.logs, key=lambda l: l.logged_on)

    def series(attr):
        return [{"x": l.logged_on.strftime("%d %b"), "y": getattr(l, attr)} for l in logs if getattr(l, attr) is not None]

    data = {"risk": [{"x": a.created_at.strftime("%d %b"), "y": a.pct, "t": round(a.threshold * 100)} for a in asc],
            "bp": series("systolic_bp"), "chol": series("cholesterol"), "weight": series("weight_kg"), "hr": series("resting_hr")}
    return render_template("patient_detail.html", p=p, data=data, today=date.today(), prefill_logged=date.today().isoformat())


@bp.route("/patients/<int:pid>/delete", methods=["POST"])
def patient_delete(pid):
    p = get_patient(pid)
    p.deleted_at = utcnow()
    db.session.commit()
    core.audit("patient.delete", f"#{p.id} {p.name}")
    flash(f"{p.name} was removed. Their past assessments are kept in the history.", "ok")
    return redirect(url_for("web.patients"), code=303)


@bp.route("/patients/<int:pid>/logs", methods=["POST"])
def log_add(pid):
    p = get_patient(pid)
    f, errors = request.form, []
    on = parse_date(f.get("logged_on", "")) or date.today()
    if on == "bad":
        on = date.today()

    def num(name, lo, hi, cast):
        raw = f.get(name, "").strip()
        if not raw:
            return None
        try:
            x = cast(raw)
        except ValueError:
            errors.append(f"{name.replace('_', ' ').title()} must be a number.")
            return None
        if not lo <= x <= hi:
            errors.append(f"{name.replace('_', ' ').title()} must be between {lo} and {hi}.")
            return None
        return x

    vals = dict(systolic_bp=num("systolic_bp", 60, 260, int), cholesterol=num("cholesterol", 80, 600, int),
                weight_kg=num("weight_kg", 20, 300, float), resting_hr=num("resting_hr", 30, 220, int))
    if not errors and all(v is None for v in vals.values()):
        errors.append("Enter at least one reading.")
    if errors:
        for e in errors:
            flash(e, "error")
    else:
        db.session.add(HealthLog(patient_id=p.id, logged_on=on, note=f.get("note", "")[:200], **vals))
        db.session.commit()
        core.audit("log.add", f"patient #{p.id}")
        flash("Reading saved.", "ok")
    return redirect(url_for("web.patient_detail", pid=p.id) + "#readings", code=303)


@bp.route("/patients/<int:pid>/logs/<int:lid>/delete", methods=["POST"])
def log_delete(pid, lid):
    p = get_patient(pid)
    log = db.session.get(HealthLog, lid)
    if log is None or log.patient_id != p.id:
        abort(404)
    db.session.delete(log)
    db.session.commit()
    core.audit("log.delete", f"patient #{p.id}")
    flash("Reading deleted.", "ok")
    return redirect(url_for("web.patient_detail", pid=p.id) + "#readings", code=303)


@bp.route("/patients/<int:pid>/followup", methods=["POST"])
def followup(pid):
    p = get_patient(pid)
    days = request.form.get("days", type=int)
    if days not in (30, 90, 180):
        abort(400)
    p.next_checkup = date.today() + timedelta(days=days)
    db.session.commit()
    core.audit("followup.set", f"patient #{p.id} in {days} days")
    flash(f"Follow-up set for {p.next_checkup:%d %b %Y}.", "ok")
    return redirect(url_for("web.patient_detail", pid=p.id), code=303)


@bp.route("/patients/<int:pid>/export.csv")
def patient_export(pid):
    p = get_patient(pid)
    if request.args.get("what") == "logs":
        rows = [[l.logged_on.isoformat(), l.systolic_bp, l.cholesterol, l.weight_kg, l.resting_hr, l.note] for l in sorted(p.logs, key=lambda l: l.logged_on)]
        return csv_response(rows, ["date", "systolic_bp", "cholesterol", "weight_kg", "resting_hr", "note"], f"patient-{p.id}-readings.csv")
    header, rows = assessment_rows(sorted(p.assessments, key=lambda a: a.created_at))
    return csv_response(rows, header, f"patient-{p.id}-assessments.csv")


# ------------------------------------------------------------------------------ batch
@bp.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", results=None, error=None)
    file = request.files.get("file")
    mode = request.form.get("mode", "standard")
    if not file or not file.filename:
        return render_template("batch.html", results=None, error="Choose a CSV file first."), 400
    if mode not in ("standard", "screening"):
        mode = "standard"
    try:
        df = pd.read_csv(file, na_values=["?", ""])
    except Exception:
        return render_template("batch.html", results=None, error="That file could not be read as CSV."), 400
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [f for f in ml.FEATURES if f not in df.columns]
    if missing:
        return render_template("batch.html", results=None, error="Missing columns: " + ", ".join(missing)), 400
    if not 1 <= len(df) <= 500:
        return render_template("batch.html", results=None, error="The file must have between 1 and 500 rows."), 400
    version, bundle = core.active()
    t = core.threshold_for(bundle, mode)
    label_col = next((c for c in ("name", "id", "label") if c in df.columns), None)
    results, ok_idx, ok_vals = [], [], []
    for i, row in df.iterrows():
        values, errs = ml.validate_inputs({f: row[f] for f in ml.FEATURES})
        label = str(row[label_col]) if label_col else f"Row {i + 1}"
        if errs:
            results.append({"label": label, "ok": False, "errors": errs})
        else:
            results.append({"label": label, "ok": True})
            ok_idx.append(len(results) - 1)
            ok_vals.append(values)
    if ok_vals:
        probs = ml.predict_many(bundle, pd.DataFrame(ok_vals))
        for idx, values, p in zip(ok_idx, ok_vals, probs):
            band = ml.band_for(float(p), t)
            results[idx].update(pct=round(float(p) * 100, 1), band=band, flagged=bool(p >= t))
            db.session.add(Assessment(source="batch", mode=mode, threshold=t, probability=float(p), band=band, flagged=bool(p >= t),
                                      model_version_id=version.id, shap_json="[]", **values))
        db.session.commit()
    core.audit("batch.run", f"{len(ok_vals)} scored of {len(df)}")
    summary = {"total": len(df), "scored": len(ok_vals), "flagged": sum(1 for r in results if r.get("flagged")),
               "threshold": t, "mode": mode, "model": version.name}
    return render_template("batch.html", results=results, summary=summary, error=None)


@bp.route("/batch/sample.csv")
def batch_sample():
    df = ml.load_builtin("cleveland").drop(columns="target").dropna().head(8).copy()
    df.insert(0, "name", [f"Sample {i + 1}" for i in range(len(df))])
    return Response(df.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=heartwise-sample.csv"})


# ------------------------------------------------------------------------- dashboard
@bp.route("/dashboard")
def dashboard():
    today, now = date.today(), utcnow()
    n_patients = db.session.scalar(select(func.count(Patient.id)).where(Patient.deleted_at.is_(None)))
    recent = db.session.scalars(select(Assessment).where(Assessment.created_at >= now - timedelta(days=30))).all()
    all_a = db.session.scalars(select(Assessment)).all()
    per_day = {(today - timedelta(days=i)): 0 for i in range(13, -1, -1)}
    for a in all_a:
        if a.created_at.date() in per_day:
            per_day[a.created_at.date()] += 1
    bands = {b: sum(1 for a in all_a if a.band == b) for b in ("low", "moderate", "high")}
    due = [p for p in active_patients() if p.next_checkup and p.next_checkup <= today + timedelta(days=7)]
    due.sort(key=lambda p: p.next_checkup)
    attention = [p for p in active_patients() if p.latest and p.latest.flagged]
    attention.sort(key=lambda p: -p.latest.probability)
    data = {"per_day": [{"x": d.strftime("%d %b"), "y": n} for d, n in per_day.items()], "bands": bands}
    return render_template("dashboard.html", n_patients=n_patients, n_recent=len(recent), flagged_pct=round(100 * sum(a.flagged for a in recent) / len(recent)) if recent else 0,
                           n_due=len(due), due=due[:6], attention=attention[:5], latest=sorted(all_a, key=lambda a: a.created_at, reverse=True)[:8],
                           data=data, today=today)


# ----------------------------------------------------------------------------- model
@bp.route("/model")
def model_page():
    v, bundle = core.active()
    return render_template("model.html", v=v, m=v.metrics, thresholds=v.metrics["thresholds"])


# ------------------------------------------------------------------- static-ish pages
@bp.route("/learn")
def learn():
    return render_template("learn.html", specs=ml.FIELD_SPECS, choices=ml.CHOICES, links=advice.LINKS)


@bp.route("/about")
def about():
    v = core.active_version()
    return render_template("about.html", v=v, m=v.metrics if v else {})


@bp.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


@bp.route("/docs")
def docs():
    return render_template("docs.html", endpoints=api.REGISTRY, example=api.EXAMPLE_BODY)


@bp.route("/health")
def health():
    v = core.active_version()
    return jsonify(status="ok", version=current_app.config["APP_VERSION"], model=v.name if v else None)


# ------------------------------------------------------------------------------ admin
@bp.route("/admin")
def admin_overview():
    v, bundle = core.active()
    now, today = utcnow(), date.today()
    items = db.session.scalars(select(Assessment)).all()
    days = {(today - timedelta(days=i)): {"web": 0, "api": 0, "batch": 0} for i in range(29, -1, -1)}
    for a in items:
        if a.created_at.date() in days:
            days[a.created_at.date()][a.source if a.source in ("web", "api", "batch") else "web"] += 1
    frame = core.recent_inputs()
    avg = []
    for f in ml.NUMERIC:
        if len(frame):
            avg.append({"label": ml.LABELS[f], "train": round(bundle["stats"][f]["mean"], 1), "recent": round(float(frame[f].mean()), 1), "unit": ml.UNITS.get(f, "")})
    drift = ml.compute_drift(bundle, frame)
    keys = db.session.scalars(select(ApiKey)).all()
    data = {"usage": [{"x": d.strftime("%d %b"), **c} for d, c in days.items()],
            "drift": [{"label": r["label"], "score": r["score"], "status": r["status"], "type": r["type"]} for r in drift["features"]]}
    return render_template("admin/overview.html", v=v, n_assess=len(items), n_patients=db.session.scalar(select(func.count(Patient.id))),
                           api_calls=sum(k.request_count or 0 for k in keys), n_models=db.session.scalar(select(func.count(ModelVersion.id))),
                           avg=avg, drift=drift, data=data, by_source={s: sum(1 for a in items if a.source == s) for s in ("web", "api", "batch")})


@bp.route("/admin/models")
def admin_models():
    versions = db.session.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())).all()
    active_v = next((v for v in versions if v.is_active), None)
    return render_template("admin/models.html", versions=versions, active_v=active_v, fast=current_app.config["FAST_TRAIN"])


@bp.route("/admin/models/retrain", methods=["POST"])
@admin_write
def admin_retrain():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Choose a CSV file with the 13 input columns and a target column.", "error")
        return redirect(url_for("web.admin_models"), code=303)
    try:
        raw = pd.read_csv(file, na_values=["?", ""])
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        missing = [c for c in ml.COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        df = ml.clean_frame(raw)
        if len(df) < 100:
            raise ValueError("Need at least 100 labelled rows.")
        if df["target"].value_counts().min() < 20:
            raise ValueError("Each class (disease / no disease) needs at least 20 rows.")
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("web.admin_models"), code=303)
    except Exception:
        flash("That file could not be read as CSV.", "error")
        return redirect(url_for("web.admin_models"), code=303)
    mv = core.train_and_register(df, f"Upload: {file.filename[:120]}", activate=False)
    core.audit("model.retrain", f"{mv.name} from {file.filename} ({len(df)} rows)")
    flash(f"Trained {mv.name}. Compare it with the active model below, then activate it if it is better.", "ok")
    return redirect(url_for("web.admin_models"), code=303)


@bp.route("/admin/models/<int:vid>/activate", methods=["POST"])
@admin_write
def admin_activate(vid):
    mv = db.session.get(ModelVersion, vid) or abort(404)
    core.activate_version(mv)
    core.audit("model.activate", mv.name)
    flash(f"{mv.name} is now the active model.", "ok")
    return redirect(url_for("web.admin_models"), code=303)


@bp.route("/admin/system")
def admin_system():
    keys = db.session.scalars(select(ApiKey).order_by(ApiKey.id.desc())).all()
    logs = db.session.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(100)).all()
    new_key = session.pop("new_key", None)
    n_demo = db.session.scalar(select(func.count(Patient.id)).where(Patient.is_demo.is_(True)))
    return render_template("admin/system.html", keys=keys, logs=logs, new_key=new_key, n_demo=n_demo,
                           db_url=current_app.config["SQLALCHEMY_DATABASE_URI"].split("@")[-1], cfg=current_app.config)


@bp.route("/admin/keys", methods=["POST"])
@admin_write
def admin_key_create():
    raw = api.create_key(request.form.get("name", "").strip() or "API key")
    core.audit("key.create", raw[:8])
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify(key=raw)
    session["new_key"] = raw
    return redirect(url_for("web.admin_system"), code=303)


@bp.route("/admin/keys/<int:kid>/revoke", methods=["POST"])
@admin_write
def admin_key_revoke(kid):
    k = db.session.get(ApiKey, kid) or abort(404)
    k.revoked = True
    db.session.commit()
    core.audit("key.revoke", k.prefix)
    flash("Key revoked.", "ok")
    return redirect(url_for("web.admin_system"), code=303)


@bp.route("/admin/demo", methods=["POST"])
@admin_write
def admin_demo():
    if request.form.get("action") == "clear":
        seed.clear_demo()
        core.audit("demo.clear")
        flash("Demo data removed.", "ok")
    else:
        if db.session.scalar(select(func.count(Patient.id)).where(Patient.is_demo.is_(True))):
            flash("Demo data is already loaded.", "warn")
        else:
            n = seed.load_demo()
            core.audit("demo.load", f"{n} assessments")
            flash(f"Demo data loaded ({n} assessments).", "ok")
    return redirect(url_for("web.admin_system"), code=303)
