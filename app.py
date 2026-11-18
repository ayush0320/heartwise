"""HeartWise - application factory and entry point.

    python app.py            # first run trains the model (about a minute), then serves on http://127.0.0.1:5000
"""
import json
import logging
import os
import secrets
import sys
import time
from datetime import date, timedelta

from flask import Flask, abort, g, jsonify, render_template, request, session
from sqlalchemy import func, select
from werkzeug.exceptions import HTTPException

import api as api_module
import core
import ml
import seed
import web as web_module
from config import Config
from models import Patient, db


class JsonFormatter(logging.Formatter):
    """One JSON object per log line - easy to search and to ship to a log service."""

    def format(self, record):
        out = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "message": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        return json.dumps(out)


def get_logger():
    logger = logging.getLogger("heartwise")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config.update(overrides or {})
    for folder in (app.config["MODEL_DIR"], app.config["UPLOAD_DIR"]):
        os.makedirs(folder, exist_ok=True)
    db.init_app(app)
    with app.app_context():
        db.create_all()
    app.register_blueprint(api_module.bp)
    app.register_blueprint(web_module.bp)
    log = get_logger()

    @app.before_request
    def before():
        g.t0 = time.perf_counter()
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and not request.path.startswith("/api/"):
            sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token") or ""
            expected = session.get("csrf") or ""
            if not sent or not expected or not secrets.compare_digest(sent, expected):   # empty never matches empty
                abort(400, description="Your security token is missing or expired. Go back, refresh the page and try again.")

    @app.after_request
    def after(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if not request.path.startswith("/static") and not app.config.get("TESTING"):
            log.info("request", extra={"fields": {"method": request.method, "path": request.path, "status": resp.status_code,
                                                  "ms": round((time.perf_counter() - g.get("t0", time.perf_counter())) * 1000, 1)}})
        return resp

    @app.context_processor
    def inject():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        ctx = dict(csrf_token=session["csrf"], cfg=app.config, ml=ml, nav_due=0, active_model=None,
                   read_only=app.config["READ_ONLY_ADMIN"], today=date.today())
        try:
            ctx["nav_due"] = db.session.scalar(select(func.count(Patient.id)).where(
                Patient.deleted_at.is_(None), Patient.next_checkup.is_not(None),
                Patient.next_checkup <= date.today() + timedelta(days=7))) or 0
            ctx["active_model"] = core.active_version()
        except Exception:
            db.session.rollback()
        return ctx

    @app.template_filter("dt")
    def fmt_dt(v):
        return v.strftime("%d %b %Y, %H:%M") if v else ""

    @app.template_filter("d")
    def fmt_d(v):
        return v.strftime("%d %b %Y") if v else ""

    def wants_json():
        return request.path.startswith("/api/") or request.path == "/live"

    @app.errorhandler(HTTPException)
    def http_error(e):
        if wants_json():
            return jsonify(error={"code": e.name.lower().replace(" ", "_"), "message": e.description}), e.code
        return render_template("error.html", code=e.code, title=e.name, message=e.description), e.code

    @app.errorhandler(core.NoModelError)
    def no_model(e):
        return render_template("error.html", code=503, title="No model yet", message=str(e)), 503

    @app.errorhandler(Exception)
    def server_error(e):
        db.session.rollback()
        log.error("unhandled error", extra={"fields": {"path": request.path, "error": repr(e)}})
        if wants_json():
            return jsonify(error={"code": "server_error", "message": "Something went wrong on our side."}), 500
        return render_template("error.html", code=500, title="Something went wrong",
                               message="An unexpected error occurred. It has been logged."), 500

    return app


def init_data(app, log=print):
    """First-run setup: train the first model and optionally load demo data."""
    with app.app_context():
        core.ensure_model(log)
        if app.config["LOAD_DEMO"] and not db.session.scalar(select(func.count(Patient.id))):
            log("Loading demo patients (fake people built from public records) ...")
            n = seed.load_demo()
            log(f"Demo data ready: {n} assessments.")


if __name__ == "__main__":
    application = create_app()
    init_data(application)
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5000"))
    print(f"\nHeartWise is running at http://{host}:{port}   (Ctrl+C to stop)\n")
    application.run(host=host, port=port, debug=False)
