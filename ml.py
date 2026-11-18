"""Machine-learning core: data loading, training, prediction, SHAP, drift and similar patients.

Nothing in this file knows about Flask or the database, so it can be tested on its own.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 42

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [f for f in FEATURES if f not in NUMERIC]
COLUMNS = FEATURES + ["target"]

LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise-induced angina",
    "oldpeak": "ST depression", "slope": "ST slope",
    "ca": "Major vessels", "thal": "Thalassemia",
}
UNITS = {"age": "years", "trestbps": "mm Hg", "chol": "mg/dl", "thalach": "bpm", "oldpeak": "mm"}

CHOICES = {
    "sex": [(1, "Male"), (0, "Female")],
    "cp": [(1, "Typical angina"), (2, "Atypical angina"), (3, "Non-anginal pain"), (4, "Asymptomatic")],
    "fbs": [(0, "No (120 mg/dl or lower)"), (1, "Yes (above 120 mg/dl)")],
    "restecg": [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")],
    "exang": [(0, "No"), (1, "Yes")],
    "slope": [(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")],
    "ca": [(0, "0 vessels"), (1, "1 vessel"), (2, "2 vessels"), (3, "3 vessels")],
    "thal": [(3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")],
}
# hard limits accepted by the validator (the training data covers a narrower range)
LIMITS = {"age": (18, 100), "trestbps": (70, 250), "chol": (100, 600), "thalach": (60, 220), "oldpeak": (0, 7)}

# Everything the input forms need: label, help text, control type, defaults
FIELD_SPECS = {
    "age": dict(label="Age", help="Age in years.", kind="number", step=1, default=55),
    "sex": dict(label="Sex", help="Sex recorded at the check-up.", kind="select", default=1),
    "cp": dict(label="Chest pain type", help="Type of chest pain, if any. 'Asymptomatic' means no chest pain.", kind="select", default=3),
    "trestbps": dict(label="Resting blood pressure", help="Top (systolic) number while resting, in mm Hg.", kind="number", step=1, default=130),
    "chol": dict(label="Cholesterol", help="Total serum cholesterol in mg/dl.", kind="number", step=1, default=230),
    "fbs": dict(label="Fasting blood sugar", help="Was fasting blood sugar above 120 mg/dl?", kind="select", default=0),
    "restecg": dict(label="Resting ECG", help="Result of the resting electrocardiogram.", kind="select", default=0),
    "thalach": dict(label="Max heart rate", help="Highest heart rate reached during the exercise test, in beats per minute.", kind="number", step=1, default=150),
    "exang": dict(label="Exercise-induced angina", help="Did exercise bring on chest pain?", kind="select", default=0),
    "oldpeak": dict(label="ST depression", help="How far the ECG ST segment dropped during exercise compared with rest (mm).", kind="number", step=0.1, default=1.0),
    "slope": dict(label="ST slope", help="Shape of the ST segment at peak exercise.", kind="select", default=2),
    "ca": dict(label="Major vessels", help="Number of major blood vessels (0-3) coloured by fluoroscopy.", kind="select", default=0),
    "thal": dict(label="Thalassemia", help="Blood-flow test result.", kind="select", default=3),
}
EXTERNAL = {"hungarian": "Hungarian", "switzerland": "Switzerland", "va": "VA Long Beach"}
AGE_BANDS = [("Under 45", 0, 45), ("45 to 59", 45, 60), ("60 and over", 60, 200)]


# --------------------------------------------------------------------------- data
def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numbers, mark impossible zeros as missing, make the target 0/1."""
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    for col in ("trestbps", "chol"):
        df.loc[df[col] == 0, col] = np.nan      # zero means "not measured" in some source files
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df.reset_index(drop=True)


def load_builtin(name: str) -> pd.DataFrame:
    """Read one of the four bundled UCI files. (The Hungarian file starts with a header line;
    it has no numeric label, so clean_frame drops it, leaving the documented 294 patients.)"""
    raw = pd.read_csv(os.path.join(DATA_DIR, f"processed.{name}.data"), header=None,
                      names=COLUMNS, na_values=["?", "-9", "-9.0"])
    return clean_frame(raw)


def value_text(feature: str, value) -> str:
    """Human wording for one input value."""
    if feature in CHOICES:
        for v, t in CHOICES[feature]:
            if float(v) == float(value):
                return t
    s = f"{float(value):.1f}"
    s = s[:-2] if s.endswith(".0") else s
    return f"{s} {UNITS.get(feature, '')}".strip()


def validate_inputs(raw: dict) -> tuple[dict, list[str]]:
    """Check the 13 inputs. Returns (clean values, list of error messages)."""
    values, errors = {}, []
    for f in FEATURES:
        label = LABELS[f]
        try:
            x = float(raw.get(f))
        except (TypeError, ValueError):
            errors.append(f"{label} is missing or not a number.")
            continue
        if not np.isfinite(x):
            errors.append(f"{label} is not a valid number.")
            continue
        if f in CHOICES:
            if x not in [float(v) for v, _ in CHOICES[f]]:
                errors.append(f"{label} must be one of {[v for v, _ in CHOICES[f]]}.")
                continue
        else:
            lo, hi = LIMITS[f]
            if not lo <= x <= hi:
                errors.append(f"{label} must be between {lo} and {hi}.")
                continue
        values[f] = x
    return values, errors


# ------------------------------------------------------------------------ helpers
def band_edges(threshold: float) -> tuple[float, float]:
    """Low / moderate / high boundaries for a flagging threshold."""
    return round(0.6 * threshold, 4), threshold


def band_for(prob: float, threshold: float) -> str:
    low_max, high_min = band_edges(threshold)
    return "low" if prob < low_max else ("moderate" if prob < high_min else "high")


def _class1(sv):
    sv = np.array(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def make_explainer(model, model_name: str, X_background: np.ndarray):
    """SHAP explainer whose values are in probability units (percentage points)."""
    if model_name == "XGBoost":
        bg = shap.sample(X_background, min(100, len(X_background)), random_state=SEED)
        return shap.TreeExplainer(model, bg, model_output="probability")
    return shap.TreeExplainer(model)


def choose_threshold(y, prob, target_recall=0.90) -> float:
    """Highest threshold (max 0.50) that still catches `target_recall` of true cases."""
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    for t in np.round(np.arange(0.50, 0.05, -0.01), 2):
        if ((prob >= t) & (y == 1)).sum() / pos >= target_recall:
            return float(t)
    return 0.10


def _metrics(y, prob, t=0.5):
    y = np.asarray(y)
    pred = (prob >= t).astype(int)
    auc = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else None
    return {"accuracy": round(float(accuracy_score(y, pred)), 3),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "roc_auc": round(auc, 3) if auc is not None else None}


def _group_stats(prob, y, t_std, t_scr):
    prob, y = np.asarray(prob), np.asarray(y)
    pos = int((y == 1).sum())

    def rec(t):
        return round(float(((prob >= t) & (y == 1)).sum() / pos), 3) if pos else None

    auc = round(float(roc_auc_score(y, prob)), 3) if len(set(y)) > 1 else None
    return {"n": int(len(y)), "positives": pos, "recall_standard": rec(t_std),
            "recall_screening": rec(t_scr), "roc_auc": auc}


# ------------------------------------------------------------------------ training
def train_model(df: pd.DataFrame, source: str, fast: bool = False, log=print):
    """Train, compare, explain and validate. Returns (bundle, metrics).

    `bundle` is what gets pickled and used for predictions; `metrics` is JSON-safe.
    """
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    imputer = SimpleImputer(strategy="median").fit(X_train)
    Xtr, Xte = imputer.transform(X_train), imputer.transform(X_test)

    trees = 60 if fast else 300
    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=trees, max_depth=5, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=60 if fast else 200, max_depth=3, learning_rate=0.05,
                                 eval_metric="logloss", random_state=SEED),
    }
    cv = StratifiedKFold(3 if fast else 5, shuffle=True, random_state=SEED)
    comparison = {}
    for name, m in candidates.items():
        log(f"  training {name} ...")
        auc = cross_val_score(m, Xtr, y_train, cv=cv, scoring="roc_auc")
        rec = cross_val_score(m, Xtr, y_train, cv=cv, scoring="recall")
        m.fit(Xtr, y_train)
        comparison[name] = {"cv_auc": round(float(auc.mean()), 3), "cv_auc_std": round(float(auc.std()), 3),
                            "cv_recall": round(float(rec.mean()), 3),
                            "test": _metrics(y_test, m.predict_proba(Xte)[:, 1])}

    tree_names = ["Decision Tree", "Random Forest", "XGBoost"]      # SHAP's exact explainer needs a tree model
    best = max(tree_names, key=lambda n: comparison[n]["cv_auc"])
    model = candidates[best]
    oof = cross_val_predict(clone(model), Xtr, y_train, cv=cv, method="predict_proba")[:, 1]
    screening = choose_threshold(y_train, oof, 0.90)
    explainer = make_explainer(model, best, Xtr)

    # global SHAP importance on the held-out split, in percentage points
    sv_test = _class1(explainer.shap_values(Xte))
    importance = sorted(({"feature": f, "label": LABELS[f], "importance": round(float(v) * 100, 2)}
                         for f, v in zip(FEATURES, np.abs(sv_test).mean(axis=0))), key=lambda d: -d["importance"])

    # evaluation: held-out split of the training source + the three other hospitals
    datasets, pooled = {}, []

    def evaluate(key, title, Xraw, yv):
        yv = np.asarray(yv)
        prob = model.predict_proba(imputer.transform(Xraw))[:, 1]
        if len(set(yv)) > 1:
            fpr, tpr, _ = roc_curve(yv, prob)
        else:
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int))
        datasets[key] = {
            "name": title, "n": int(len(yv)), "positive_pct": round(float(np.mean(yv)) * 100, 1),
            "missing_pct": round(float(Xraw.isna().mean().mean()) * 100, 1),
            "standard": _metrics(yv, prob, 0.5), "screening": _metrics(yv, prob, screening),
            "roc": [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in keep],
            "probs": [round(float(p), 4) for p in prob], "y": [int(v) for v in yv],
        }
        pooled.append(pd.DataFrame({"prob": prob, "y": yv, "sex": Xraw["sex"].values, "age": Xraw["age"].values}))

    evaluate("heldout", "Held-out test split", X_test, y_test)
    for key, title in EXTERNAL.items():
        d = load_builtin(key)
        evaluate(key, title, d[FEATURES], d["target"])

    allp = pd.concat(pooled, ignore_index=True)
    fairness = {"sex": {}, "age": {}}
    for label, code in (("Male", 1), ("Female", 0)):
        g = allp[allp["sex"] == code]
        if len(g):
            fairness["sex"][label] = _group_stats(g["prob"], g["y"], 0.5, screening)
    for label, lo, hi in AGE_BANDS:
        g = allp[(allp["age"] >= lo) & (allp["age"] < hi)]
        if len(g):
            fairness["age"][label] = _group_stats(g["prob"], g["y"], 0.5, screening)

    # reference statistics for drift detection and range warnings
    stats = {}
    for f in FEATURES:
        col = X_train[f].dropna()
        entry = {"min": float(col.min()), "max": float(col.max())}
        if f in NUMERIC:
            entry.update(type="numeric", mean=float(col.mean()), std=float(col.std() or 1.0))
        else:
            entry.update(type="categorical", props={str(float(k)): float(v) for k, v in col.value_counts(normalize=True).items()})
        stats[f] = entry

    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    knn = {"X": Xtr.astype("float32"), "y": np.asarray(y_train), "mean": Xtr.mean(axis=0), "std": std}

    bundle = {"model": model, "imputer": imputer, "explainer": explainer, "model_name": best,
              "screening_threshold": screening, "stats": stats, "knn": knn}
    metrics = {
        "model_name": best, "source": source, "n_total": int(len(df)), "n_train": int(len(X_train)),
        "n_test": int(len(X_test)), "comparison": comparison,
        "thresholds": {"standard": 0.5, "screening": screening},
        "datasets": datasets, "importance": importance, "fairness": fairness,
    }
    return bundle, metrics


# ---------------------------------------------------------------------- prediction
def _row(bundle, values):
    return bundle["imputer"].transform(pd.DataFrame([{f: values[f] for f in FEATURES}], columns=FEATURES))


def predict_proba(bundle, values) -> float:
    return float(bundle["model"].predict_proba(_row(bundle, values))[0][1])


def predict_many(bundle, frame: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(bundle["imputer"].transform(frame[FEATURES]))[:, 1]


def explain(bundle, values) -> dict:
    """SHAP breakdown of one patient, in percentage points."""
    row = _row(bundle, values)
    prob = float(bundle["model"].predict_proba(row)[0][1])
    sv = _class1(bundle["explainer"].shap_values(row))[0]
    base = float(np.atleast_1d(bundle["explainer"].expected_value)[-1])
    items = [{"feature": f, "label": LABELS[f], "value": float(values[f]), "value_text": value_text(f, values[f]),
              "points": round(float(c) * 100, 2)} for f, c in zip(FEATURES, sv)]
    items.sort(key=lambda d: -abs(d["points"]))
    return {"prob": prob, "base_pct": round(base * 100, 1), "contributions": items,
            "check_pct": round((base + float(sv.sum())) * 100, 1)}


def range_warnings(bundle, values) -> list[str]:
    out = []
    for f in NUMERIC:
        s = bundle["stats"][f]
        if not s["min"] <= values[f] <= s["max"]:
            out.append(f"{LABELS[f]} ({values[f]:g}) is outside the range seen in the training data "
                       f"({s['min']:g} to {s['max']:g}), so the estimate may be less reliable.")
    return out


def similar_patients(bundle, values, k=15) -> dict:
    knn = bundle["knn"]
    z = (_row(bundle, values)[0] - knn["mean"]) / knn["std"]
    Z = (knn["X"] - knn["mean"]) / knn["std"]
    idx = np.argsort(np.linalg.norm(Z - z, axis=1))[:k]
    yy = knn["y"][idx]
    return {"k": int(k), "with_disease": int(yy.sum()), "pct": round(float(yy.mean()) * 100, 1)}


def compute_drift(bundle, frame: pd.DataFrame, min_rows=8) -> dict:
    """Compare recently logged inputs with the training data."""
    if len(frame) < min_rows:
        return {"status": "insufficient", "n": int(len(frame)), "min_rows": min_rows, "features": []}
    rows = []
    for f in FEATURES:
        s, col = bundle["stats"][f], frame[f].dropna().astype(float)
        if col.empty:
            continue
        if s["type"] == "numeric":
            score = float((col.mean() - s["mean"]) / s["std"])
            status = "drift" if abs(score) >= 0.5 else ("watch" if abs(score) >= 0.25 else "ok")
            detail = f"training mean {s['mean']:.1f}, recent mean {col.mean():.1f}"
        else:
            new = col.value_counts(normalize=True)
            keys = set(s["props"]) | {str(float(k)) for k in new.index}
            score = 0.5 * sum(abs(s["props"].get(k, 0.0) - float(new.get(float(k), 0.0))) for k in keys)
            status = "drift" if score >= 0.25 else ("watch" if score >= 0.15 else "ok")
            detail = "category mix changed" if status != "ok" else "category mix similar"
        rows.append({"feature": f, "label": LABELS[f], "type": s["type"], "score": round(score, 3), "status": status, "detail": detail})
    n_drift = sum(r["status"] == "drift" for r in rows)
    overall = "drift" if n_drift >= 2 else ("watch" if any(r["status"] != "ok" for r in rows) else "ok")
    return {"status": overall, "n": int(len(frame)), "min_rows": min_rows, "features": rows}
