"""Turns SHAP contributions into plain-language sentences and general advice cards.

This is general health information only - it is not medical advice.
"""
LINKS = [
    ("WHO: cardiovascular diseases fact sheet", "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"),
    ("American Heart Association", "https://www.heart.org/"),
]

# kind: lifestyle = things you can influence, doctor = needs a clinician, fixed = cannot be changed
ADVICE = {
    "age": dict(kind="fixed", title="Age", text="Risk naturally rises with age. You can't change this, but regular check-ups and keeping the other numbers in a healthy range matter more as the years go on."),
    "sex": dict(kind="fixed", title="Sex", text="Heart-disease patterns differ between men and women. This is not something you can change; it is one reason doctors look at the whole picture instead of a single number."),
    "cp": dict(kind="doctor", title="Chest pain", text="Chest pain that is new, pressing, or spreads to the arm or jaw should be checked by a doctor promptly. If it is severe or comes with breathlessness or sweating, call your local emergency number (112 in India) right away."),
    "trestbps": dict(kind="lifestyle", title="Blood pressure", text="Your resting blood pressure is pushing the estimate up. Have it measured regularly. Less salt, regular activity, a healthy weight and limiting alcohol usually help. Ask your doctor what target is right for you."),
    "chol": dict(kind="lifestyle", title="Cholesterol", text="Cholesterol is pushing the estimate up. Ask for a full lipid profile. More vegetables, fruit, whole grains and pulses, and less fried and processed food, usually help. Your doctor decides whether medicine is needed."),
    "fbs": dict(kind="doctor", title="Blood sugar", text="A fasting blood sugar above 120 mg/dl is worth following up. Ask about an HbA1c or a repeat glucose test, because diabetes strongly affects heart risk."),
    "restecg": dict(kind="doctor", title="Resting ECG", text="The resting ECG shows a change that a doctor should interpret. Keep the report and share it at your next visit."),
    "thalach": dict(kind="lifestyle", title="Peak heart rate", text="A low peak heart rate during the exercise test can reflect fitness or how the heart responds to effort. Regular, doctor-approved physical activity usually helps, and the result should be reviewed by your doctor."),
    "exang": dict(kind="doctor", title="Pain during exercise", text="Chest pain brought on by exercise is an important symptom. Don't push through it. Discuss it with a doctor soon."),
    "oldpeak": dict(kind="doctor", title="ST depression", text="ST depression during exercise is an ECG finding that needs a cardiologist's interpretation. Bring the full test report."),
    "slope": dict(kind="doctor", title="ST slope", text="The ST slope at peak exercise is an ECG pattern. A cardiologist can tell you what it means for you."),
    "ca": dict(kind="doctor", title="Major vessels", text="The number of narrowed vessels comes from an imaging test. It is the kind of result a cardiologist should explain and follow up."),
    "thal": dict(kind="doctor", title="Blood-flow test", text="A blood-flow defect on the stress test should be explained by a cardiologist and, if needed, investigated further."),
}
KIND_LABEL = {"lifestyle": "Lifestyle", "doctor": "Talk to a doctor", "fixed": "Can't be changed"}


def build_advice(contributions, min_points=2.0, limit=4):
    """Cards for the inputs that pushed the estimate up the most."""
    cards = []
    for c in contributions:
        if c["points"] >= min_points and c["feature"] in ADVICE:
            a = ADVICE[c["feature"]]
            cards.append({**a, "feature": c["feature"], "label": c["label"], "points": c["points"],
                          "value_text": c["value_text"], "kind_label": KIND_LABEL[a["kind"]]})
        if len(cards) >= limit:
            break
    return cards


def plain_sentences(contributions, n=3):
    out = []
    for c in contributions[:n]:
        verb = "raised" if c["points"] > 0 else "lowered"
        out.append(f"{c['label']} ({c['value_text']}) {verb} the estimate by {abs(c['points']):.1f} points.")
    return out
