"""Database tables (SQLAlchemy)."""
import json
from datetime import date, datetime, timezone

from flask_sqlalchemy import SQLAlchemy

from ml import FEATURES

db = SQLAlchemy()


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    sex = db.Column(db.Integer)                       # 1 male, 0 female, None unknown
    birth_year = db.Column(db.Integer)
    notes = db.Column(db.Text, default="")
    next_checkup = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=utcnow)
    deleted_at = db.Column(db.DateTime)               # soft delete
    is_demo = db.Column(db.Boolean, default=False)
    assessments = db.relationship("Assessment", back_populates="patient", order_by="Assessment.created_at.desc()")
    logs = db.relationship("HealthLog", back_populates="patient", order_by="HealthLog.logged_on.desc()",
                           cascade="all, delete-orphan")

    @property
    def age(self):
        return date.today().year - self.birth_year if self.birth_year else None

    @property
    def latest(self):
        return self.assessments[0] if self.assessments else None

    @property
    def initials(self):
        parts = [p for p in self.name.split() if p]
        return "".join(p[0] for p in parts[:2]).upper() or "?"


class HealthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False, index=True)
    logged_on = db.Column(db.Date, nullable=False, default=date.today)
    systolic_bp = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    weight_kg = db.Column(db.Float)
    resting_hr = db.Column(db.Integer)
    note = db.Column(db.String(200), default="")
    patient = db.relationship("Patient", back_populates="logs")


class ModelVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    source = db.Column(db.String(200), default="")
    n_train = db.Column(db.Integer)
    n_test = db.Column(db.Integer)
    model_name = db.Column(db.String(40))
    cv_auc = db.Column(db.Float)
    test_acc = db.Column(db.Float)
    test_recall = db.Column(db.Float)
    test_auc = db.Column(db.Float)
    screening_threshold = db.Column(db.Float, default=0.3)
    file_name = db.Column(db.String(120))
    metrics_json = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=False)

    @property
    def metrics(self):
        return json.loads(self.metrics_json or "{}")


class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), index=True)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    source = db.Column(db.String(20), default="web")          # web | api | batch
    mode = db.Column(db.String(12), default="standard")       # standard | screening
    threshold = db.Column(db.Float)
    probability = db.Column(db.Float, nullable=False)
    band = db.Column(db.String(10))
    flagged = db.Column(db.Boolean, default=False)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.id"))
    shap_json = db.Column(db.Text)
    is_demo = db.Column(db.Boolean, default=False)
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Float)
    restecg = db.Column(db.Float)
    thalach = db.Column(db.Float)
    exang = db.Column(db.Float)
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Float)
    ca = db.Column(db.Float)
    thal = db.Column(db.Float)
    patient = db.relationship("Patient", back_populates="assessments")
    model_version = db.relationship("ModelVersion")

    @property
    def inputs(self):
        return {f: getattr(self, f) for f in FEATURES}

    @property
    def contributions(self):
        return json.loads(self.shap_json or "[]")

    @property
    def pct(self):
        return round(self.probability * 100, 1)


class ApiKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    prefix = db.Column(db.String(12))
    key_hash = db.Column(db.String(64), unique=True, nullable=False)   # only a hash is stored
    created_at = db.Column(db.DateTime, default=utcnow)
    last_used_at = db.Column(db.DateTime)
    request_count = db.Column(db.Integer, default=0)
    revoked = db.Column(db.Boolean, default=False)


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    at = db.Column(db.DateTime, default=utcnow, index=True)
    action = db.Column(db.String(40), nullable=False)
    detail = db.Column(db.String(500), default="")
    ip = db.Column(db.String(60))
