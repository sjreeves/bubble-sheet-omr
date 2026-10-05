#!/usr/bin/env python3
"""
make_sheet.py - draw the demo answer sheet as a print-ready PDF.

    pip install reportlab
    python3 tools/make_sheet.py --form A              # sheets/demo-form-a.pdf
    python3 tools/make_sheet.py --form A --ink red   # sheets/demo-form-a-red.pdf
    python3 tools/make_sheet.py --form B             # ... and the same for form B

The sheet is an original design, not a reproduction of any published answer
document, but it is deliberately built the way real ones are, because every
one of those choices is what makes recognition hard:

  * Sections with different geometry within a sheet (4- and 5-choice, 9, 10 and
    12 rows per column) so the recognizer cannot assume one grid, and two forms
    that differ throughout so it has to work out which sheet it is reading.
  * Questions numbered down each column, not across - column-major, as printed.
  * Alternating letter sets (A B C D / F G H J) so adjacent rows do not share
    a glyph. This is also why the engine corrects a per-letter darkness bias:
    "B" and "G" put more ink inside the oval than "A" or "J".
  * The choice letter printed *inside* the oval, which is why the fill measure
    samples a dark percentile rather than a mean.
  * A tight row pitch - 8.28pt on form A, about 23px at the 200dpi reference
    scale - so neighbouring rows are close enough that a few pixels of
    misregistration would read the wrong row. Form B uses a different pitch and
    a different margin, so the two grids never coincide and choosing between
    them is a measurement rather than a guess.
  * Timing marks down the left edge and solid corner fiducials, which give the
    feature matcher something unambiguous to lock onto.

Two inks are produced. Real answer documents are often printed in a drop-out
red that a scanner's red channel cannot see, so pencil is all that remains;
photocopies of those sheets come out as grey printing that cannot be dropped
out at all. The engine handles both, and the eval harness needs both.

Alongside the PDF this writes sheets/<id>-geometry.json: every oval centre as
drawn, in PDF points from the top-left. tools/make_template.py then recovers
the same geometry from the PDF alone and asserts the two agree, so a template
can be built for any sheet, not only one whose generator we happen to own.
"""
import argparse
import json
import os

from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = 612.0, 792.0          # US Letter, points

OVAL_W, OVAL_H = 8.9, 6.2              # a wide, shallow oval: hard to centre, like the real thing
OPT_PITCH = 11.04                       # horizontal centre-to-centre within a question
LABEL_W = 24.0                           # room for the question number left of the first oval
COL_GAP = 14.0                           # gap between answer columns

INKS = {
    # name: (print colour, a human description for the PDF title)
    "black": (Color(0.1, 0.1, 0.1), "black"),
    # drop-out red: bright in the red channel (so the red channel sees only pencil),
    # dark in green (so the green channel still shows the printing for alignment).
    "red": (Color(0.96, 0.53, 0.57), "drop-out red"),
}

# Two printed forms, so the recognizer has to work out which sheet it is looking
# at rather than being told. They differ in section count, choice count, rows per
# column and vertical position - everything the matcher could otherwise assume.
# name, label, questions, options, rows per column, y of the first oval row (from page top)
FORMS = {
    "A": {
        "left": 76.0, "row_pitch": 8.28, "timing_top": 170.0, "footer": 690.0,
        "sections": [
            ("part1", "Section 1", 50, 4, 10, 180.0),
            ("part2", "Section 2", 45, 5, 9, 316.0),
            ("part3", "Section 3", 36, 4, 9, 444.0),
            ("part4", "Section 4", 40, 4, 10, 572.0),
        ],
    },
    "B": {
        # a different margin and a different row pitch, not just different counts:
        # if one form's grid landed on top of another's, choosing between them
        # would be guesswork rather than measurement
        "left": 92.0, "row_pitch": 9.6, "timing_top": 165.0, "footer": 710.0,
        "sections": [
            ("part1", "Section 1", 60, 5, 12, 176.0),
            ("part2", "Section 2", 60, 5, 12, 320.0),
            ("part3", "Section 3", 40, 4, 10, 464.0),
            ("part4", "Section 4", 40, 4, 10, 588.0),
        ],
    },
}

ODD_LETTERS = "ABCDE"
EVEN_LETTERS = "FGHJK"


def oval_centres(count, options, rows_per_col, y_top, left, row_pitch):
    """Every oval centre for one section, in points from the top-left of the page.

    Numbering is column-major: question 1 is the top of the first column, and
    question rows_per_col + 1 is the top of the second column.
    """
    col_w = LABEL_W + (options - 1) * OPT_PITCH + OVAL_W + COL_GAP
    out = []
    for n in range(1, count + 1):
        ci, ri = divmod(n - 1, rows_per_col)
        letters = ODD_LETTERS if n % 2 else EVEN_LETTERS
        x0 = left + ci * col_w + LABEL_W + OVAL_W / 2.0
        y = y_top + ri * row_pitch
        out.append({
            "n": n,
            "label_x": left + ci * col_w + LABEL_W - 4.0,
            "choices": [{"l": letters[i], "x": round(x0 + i * OPT_PITCH, 2), "y": round(y, 2)}
                        for i in range(options)],
        })
    return out


def draw(path, ink_name, form):
    ink, ink_desc = INKS[ink_name]
    spec = FORMS[form]
    LEFT = spec["left"]
    # invariant=1 fixes the embedded timestamp and document id, so regenerating
    # a sheet produces a byte-identical PDF and the repository stays clean
    c = canvas.Canvas(path, pagesize=(PAGE_W, PAGE_H), invariant=1)
    c.setTitle("OMR demo answer sheet, form %s (%s)" % (form, ink_desc))

    def top(y):
        """PDF points from the page top -> reportlab's bottom-left origin."""
        return PAGE_H - y

    # --- corner fiducials: solid squares the feature matcher can rely on ----
    c.setFillColor(ink)
    for x in (36.0, PAGE_W - 46.0):
        for y in (36.0, PAGE_H - 46.0):
            c.rect(x, y, 10.0, 10.0, stroke=0, fill=1)

    # --- timing marks down the left edge ------------------------------------
    for i in range(50):
        c.rect(46.0, top(spec["timing_top"] + i * 10.0), 8.0, 4.0, stroke=0, fill=1)

    # --- heading ------------------------------------------------------------
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(LEFT, top(52.0), "OMR DEMO ANSWER SHEET")
    c.setFont("Helvetica", 8)
    c.drawString(LEFT, top(66.0), "Form %s  -  synthetic sheet for optical mark recognition testing" % form)

    c.setLineWidth(0.6)
    c.setStrokeColor(ink)
    for label, x, w in (("NAME", LEFT, 230.0), ("DATE", LEFT + 300.0, 130.0)):
        c.setFont("Helvetica", 7)
        c.drawString(x, top(88.0), label)
        c.line(x + 30.0, top(90.0), x + 30.0 + w, top(90.0))

    c.setFont("Helvetica", 7.5)
    c.drawString(LEFT, top(112.0), "Fill the oval completely with a soft pencil. Make no stray marks on this sheet.")
    c.drawString(LEFT, top(124.0), "Erase cleanly to change an answer. Mark only one oval for each question.")

    # --- answer grid --------------------------------------------------------
    geometry = {"page": {"width": PAGE_W, "height": PAGE_H},
                "oval": {"w": OVAL_W, "h": OVAL_H}, "rowPitch": spec["row_pitch"],
                "sections": {}}
    for name, label, count, options, rows_per_col, y_top in spec["sections"]:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(LEFT, top(y_top - 13.0), label.upper())
        c.setFont("Helvetica", 7)
        c.drawString(LEFT + 70.0, top(y_top - 13.0),
                     "%d questions, %d choices" % (count, options))
        c.line(LEFT, top(y_top - 8.5), PAGE_W - 76.0, top(y_top - 8.5))

        questions = oval_centres(count, options, rows_per_col, y_top, LEFT, spec["row_pitch"])
        c.setLineWidth(0.5)
        for q in questions:
            c.setFont("Helvetica", 5.5)
            c.drawRightString(q["label_x"], top(q["choices"][0]["y"] + 1.9), str(q["n"]))
            for ch in q["choices"]:
                x, y = ch["x"], ch["y"]
                c.ellipse(x - OVAL_W / 2.0, top(y + OVAL_H / 2.0),
                          x + OVAL_W / 2.0, top(y - OVAL_H / 2.0), stroke=1, fill=0)
                # the choice letter sits inside the oval, as on a real sheet
                c.setFont("Helvetica", 4.2)
                c.drawCentredString(x, top(y + 1.5), ch["l"])
        geometry["sections"][name] = {
            "label": label, "count": count, "options": options,
            "rowsPerColumn": rows_per_col,
            "questions": [{"n": q["n"], "choices": q["choices"]} for q in questions],
        }

    c.setFont("Helvetica", 6.5)
    c.drawString(LEFT, top(spec["footer"]),
                 "Demo sheet - original design, released with the recognizer under the MIT licence.")
    c.showPage()
    c.save()
    return geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ink", choices=sorted(INKS), default="black")
    ap.add_argument("--form", choices=sorted(FORMS), default="A")
    args = ap.parse_args()
    args.id = "demo-form-%s" % args.form.lower()

    here = os.path.dirname(os.path.abspath(__file__))
    sheets = os.path.join(os.path.dirname(here), "sheets")
    os.makedirs(sheets, exist_ok=True)
    suffix = "" if args.ink == "black" else "-" + args.ink
    pdf = os.path.join(sheets, "%s%s.pdf" % (args.id, suffix))
    geometry = draw(pdf, args.ink, args.form)
    print("wrote", pdf)

    # the as-drawn geometry, for make_template.py to check its extraction against
    geo_path = os.path.join(sheets, "%s-geometry.json" % args.id)
    with open(geo_path, "w") as fh:
        json.dump(geometry, fh, separators=(",", ":"))
    print("wrote", geo_path,
          "(%d ovals)" % sum(len(q["choices"]) for s in geometry["sections"].values()
                             for q in s["questions"]))


if __name__ == "__main__":
    main()
