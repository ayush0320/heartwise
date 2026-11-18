"""Configuration. Everything sensitive or environment-specific comes from environment variables."""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.environ.get("INSTANCE_DIR", os.path.join(BASE_DIR, "instance"))


def _flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _secret_key():
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    path = os.path.join(INSTANCE_DIR, "secret_key")        # generated once, kept out of git
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


def _db_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        return url.replace("postgres://", "postgresql://", 1)
    return "sqlite:///" + os.path.join(INSTANCE_DIR, "heart.db").replace("\\", "/")


class Config:
    APP_VERSION = "1.0.0"
    SECRET_KEY = _secret_key()
    SQLALCHEMY_DATABASE_URI = _db_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_DIR = os.path.join(INSTANCE_DIR, "models")
    UPLOAD_DIR = os.path.join(INSTANCE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024            # 5 MB uploads
    READ_ONLY_ADMIN = _flag("READ_ONLY_ADMIN")      # blocks retrain / activate / keys / demo data
    LOAD_DEMO = _flag("LOAD_DEMO", "1")             # seed demo patients on first run
    API_RATE_LIMIT = int(os.environ.get("API_RATE_LIMIT", "30"))   # requests per minute per API key
    FAST_TRAIN = _flag("FAST_TRAIN")                # small, quick models (used by the tests)
    TESTING = False
    PER_PAGE = 10
    # shown on the About page - edit freely
    PROJECT_TITLE = "HeartWise"
    TEAM = "Ayush Gautam"
    INSTITUTE = "B.Tech, Information Technology, Maharaja Agrasen Institute of Technology"
