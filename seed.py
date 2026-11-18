"""Demo data: fake patients built from public UCI rows. Everything created here is flagged is_demo."""
from datetime import date, timedelta

import numpy as np
from sqlalchemy import select

import core
import ml
from models import Assessment, HealthLog, Patient, db, utcnow

PEOPLE = [  # name, sex, source dataset, trend (-1 improving, 0 steady, +1 worsening), next check-up offset (days)
    ("Asha Verma", 0, "cleveland", -1, 21), ("Ravi Kumar", 1, "cleveland", 0, 3), ("Meera Nair", 0, "cleveland", +1, 45),
    ("Imran Sheikh", 1, "cleveland", -1, 90), ("Kavita Rao", 0, "cleveland", 0, -6), ("Lakshmi Iyer", 0, "cleveland", +1, None),
    ("Suresh Patel", 1, "switzerland", +1, 10), ("Arjun Singh", 1, "va", 0, 60),
]


def _complete(row, median):
    r = row[ml.FEATURES].copy().fillna(median)
    return {f: float(r[f]) for f in ml.FEATURES}


def _clip(values):
    for f, (lo, hi) in ml.LIMITS.items():
        values[f] = float(min(max(values[f], lo), hi))
    for f in ("age", "trestbps", "chol", "thalach"):
        values[f] = float(round(values[f]))
    values["oldpeak"] = round(values["oldpeak"], 1)
    return values


def clear_demo():
    for a in db.session.scalars(select(Assessment).where(Assessment.is_demo.is_(True))).all():
        db.session.delete(a)
    for p in db.session.scalars(select(Patient).where(Patient.is_demo.is_(True))).all():
        db.session.delete(p)
    db.session.commit()


def load_demo():
    """Create 8 demo patients with assessments and health logs, plus some quick assessments."""
    core.active()
    rng = np.random.default_rng(7)
    frames = {k: ml.load_builtin(k) for k in ("cleveland", "hungarian", "switzerland", "va")}
    median = frames["cleveland"][ml.FEATURES].median()
    today = date.today()
    n_assess = 0

    for name, sex, src, trend, nxt in PEOPLE:
        pool = frames[src]
        pool = pool[pool["sex"] == sex] if (pool["sex"] == sex).any() else pool
        base = _complete(pool.iloc[int(rng.integers(len(pool)))], median)
        p = Patient(name=name, sex=sex, birth_year=today.year - int(base["age"]), is_demo=True,
                    next_checkup=(today + timedelta(days=nxt)) if nxt is not None else None,
                    notes="Demo patient built from a public UCI record. Not a real person.")
        db.session.add(p)
        db.session.flush()

        for step, days_ago in enumerate([84, 56, 28, 3]):
            v = dict(base)
            v["trestbps"] += trend * 3.5 * step + rng.normal(0, 2)
            v["chol"] += trend * 9 * step + rng.normal(0, 5)
            v["thalach"] -= trend * 2.5 * step + rng.normal(0, 2)
            v["age"] = float(base["age"])
            v = _clip(v)
            mode = "screening" if step % 2 else "standard"
            res = core.analyze(v, mode, full=False)
            core.save_assessment(res, v, patient_id=p.id, is_demo=True, created_at=utcnow() - timedelta(days=days_ago, hours=int(rng.integers(0, 9))))
            n_assess += 1

        for i, d in enumerate(range(100, 0, -14)):
            step = i / 7.0
            db.session.add(HealthLog(
                patient_id=p.id, logged_on=today - timedelta(days=d),
                systolic_bp=int(round(base["trestbps"] + trend * 12 * step + rng.normal(0, 4))),
                cholesterol=int(round(base["chol"] + trend * 25 * step + rng.normal(0, 8))) if i % 3 == 0 else None,
                weight_kg=round(float(66 + 14 * rng.random() + trend * 2.2 * step + rng.normal(0, 0.5)), 1),
                resting_hr=int(round(72 + rng.normal(0, 4) + trend * 2 * step))))

    # quick assessments (no patient) spread over the last month - these feed the dashboard and drift view
    for key, count in (("cleveland", 6), ("hungarian", 4), ("va", 4), ("switzerland", 2)):
        pool = frames[key]
        for _ in range(count):
            v = _clip(_complete(pool.iloc[int(rng.integers(len(pool)))], median))
            res = core.analyze(v, "screening" if rng.random() < 0.4 else "standard", full=False)
            core.save_assessment(res, v, source="api" if rng.random() < 0.3 else "web", is_demo=True,
                                 created_at=utcnow() - timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 23))))
            n_assess += 1
    db.session.commit()
    return n_assess
