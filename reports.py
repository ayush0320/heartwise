"""One-page PDF report for an assessment (reportlab)."""
from io import BytesIO

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import ml

NAVY, RED, GREEN, AMBER = colors.HexColor("#10263f"), colors.HexColor("#c8283d"), colors.HexColor("#18a36b"), colors.HexColor("#d99100")
INK, MUTED, LINE, SOFT = colors.HexColor("#14202e"), colors.HexColor("#6b7a90"), colors.HexColor("#dde3ec"), colors.HexColor("#f3f6fa")
BAND_COLOR = {"low": GREEN, "moderate": AMBER, "high": RED}
BAND_TEXT = {"low": "Low", "moderate": "Moderate", "high": "High"}


def _styles():
    base = dict(fontName="Helvetica", textColor=INK, leading=13, alignment=TA_LEFT)
    return {
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=18, textColor=colors.white, leading=22),
        "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#c9d6e6"), leading=12),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, textColor=NAVY, spaceBefore=10, spaceAfter=4, leading=14),
        "body": ParagraphStyle("body", fontSize=9, **base),
        "small": ParagraphStyle("small", fontSize=8, **{**base, "textColor": MUTED, "leading": 11}),
        "big": ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=INK),
    }


def _bars(contribs, width=170 * mm, n=8):
    rows = contribs[:n]
    row_h, label_w = 14, 62 * mm
    h = row_h * len(rows) + 8
    d = Drawing(width, h)
    mx = max((abs(r["points"]) for r in rows), default=1) or 1
    cx = label_w + (width - label_w) / 2
    half = (width - label_w) / 2 - 24
    d.add(Line(cx, 2, cx, h - 2, strokeColor=LINE, strokeWidth=0.8))
    for i, r in enumerate(rows):
        y = h - 6 - (i + 1) * row_h + 3
        d.add(String(0, y + 2, f"{r['label']}: {r['value_text']}"[:44], fontName="Helvetica", fontSize=7.5, fillColor=INK))
        w = abs(r["points"]) / mx * half
        up = r["points"] > 0
        d.add(Rect(cx if up else cx - w, y, max(w, 0.8), 9, fillColor=RED if up else GREEN, strokeColor=None))
        d.add(String(cx + w + 3 if up else cx - w - 3, y + 2, f"{r['points']:+.1f}", fontName="Helvetica-Bold", fontSize=7.5,
                     fillColor=INK, textAnchor="start" if up else "end"))
    return d


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "HeartWise - educational screening aid. Not a medical diagnosis.")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report(assessment, patient, result) -> bytes:
    s = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=18 * mm,
                            title=f"Heart risk assessment #{assessment.id}", author="HeartWise")
    W = A4[0] - 36 * mm
    story = []

    head = Table([[Paragraph("Heart risk assessment", s["h1"])],
                  [Paragraph(f"Report #{assessment.id} &nbsp;|&nbsp; {assessment.created_at:%d %b %Y, %H:%M} UTC", s["sub"])]], colWidths=[W])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                              ("TOPPADDING", (0, 0), (0, 0), 12), ("BOTTOMPADDING", (0, -1), (-1, -1), 12)]))
    story += [head, Spacer(1, 8)]

    who = patient.name if patient else "Quick assessment (no patient record)"
    meta = [["Patient", who], ["Model", f"{result['model']['name']} ({result['model']['algorithm']})"],
            ["Mode", f"{assessment.mode.title()} (flags at {round(result['threshold'] * 100)}% or above)"]]
    t = Table(meta, colWidths=[28 * mm, W - 28 * mm])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t)

    band, c = result["band"], BAND_COLOR[result["band"]]
    verdict = ("Flagged: further cardiac investigation suggested" if result["flagged"] else "Not flagged: lower estimated likelihood")
    box = Table([[Paragraph(f"{result['pct']}%", s["big"]),
                  [Paragraph(f"<b>{BAND_TEXT[band]} estimated likelihood</b>", ParagraphStyle("b", parent=s["body"], fontSize=11, textColor=c)),
                   Paragraph(verdict, s["body"]),
                   Paragraph(f"The average patient in the training data starts at {result['base_pct']}%.", s["small"])]]],
                colWidths=[48 * mm, W - 48 * mm])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("LINEBEFORE", (0, 0), (0, -1), 4, c), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [Spacer(1, 8), box]

    story.append(KeepTogether([Paragraph("What moved this estimate (percentage points)", s["h2"]),
                               Paragraph("Red bars raised the estimate, green bars lowered it. Values come from SHAP.", s["small"]),
                               Spacer(1, 4), _bars(result["contributions"])]))
    if result["sentences"]:
        story.append(Paragraph("In plain words", s["h2"]))
        story += [Paragraph("&bull; " + x, s["body"]) for x in result["sentences"]]

    inputs = assessment.inputs
    cells = [[Paragraph(f"<font color='#6b7a90'>{ml.LABELS[f]}</font>", s["small"]), Paragraph(ml.value_text(f, inputs[f]), s["body"])] for f in ml.FEATURES]
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:] + [["", ""]] * (half - len(cells[half:]))
    grid = Table([l + r for l, r in zip(left, right)], colWidths=[34 * mm, W / 2 - 34 * mm, 34 * mm, W / 2 - 34 * mm])
    grid.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [Paragraph("Inputs used", s["h2"]), grid]

    if result["advice"]:
        story.append(Paragraph("General information on the top factors", s["h2"]))
        for a in result["advice"][:3]:
            story.append(Paragraph(f"<b>{a['title']}</b> ({a['kind_label']}): {a['text']}", s["body"]))
            story.append(Spacer(1, 3))

    warn = Table([[Paragraph("<b>Important.</b> This report comes from a student project trained on 303 patients. It is a screening aid and "
                             "must not be used to diagnose or rule out heart disease. In testing on other hospitals' data it missed many "
                             "patients with disease. Please talk to a doctor about your health. For severe chest pain, breathlessness or "
                             "fainting, call your local emergency number (112 in India).", s["small"])]], colWidths=[W])
    warn.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, AMBER), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf3dc")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [Spacer(1, 10), warn]
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
