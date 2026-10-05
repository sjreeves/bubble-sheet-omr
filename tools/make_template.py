#!/usr/bin/env python3
"""
make_template.py - build a sheet template and its alignment reference from a
blank answer sheet PDF.

    pip install pdfplumber pillow        # plus `pdftoppm` (poppler-utils) on PATH
    python3 tools/make_template.py demo-form-a sheets/demo-form-a.pdf
    python3 tools/make_template.py demo-form-b sheets/demo-form-b.pdf

Writes sheets/<id>.json (the geometry the recognizer reads) and
sheets/<id>-ref.png (the image it aligns photos against).

Why this exists: the recognizer knows nothing about any particular sheet. It is
handed a list of oval centres in PDF points and a reference render, and it will
read any sheet described that way. Adding a layout therefore means describing
it here - a y-band per section, how many rows run down a column, how many
choices - and letting the extractor find the ovals, rather than typing several
hundred coordinates by hand.

How ovals are found: every vector curve on the page whose bounding box matches
the oval size, to a tolerance. They are bucketed into rows by y, grouped into
columns by x (a gap wider than COLUMN_GAP starts a new column), and then
numbered column-major - 1..rowsPerColumn down the first column, as printed.
Odd questions get A-B-C-D(-E) and even questions F-G-H-J(-K), which is the
convention the demo sheet prints and the recognizer's per-letter bias
correction assumes.

Both demo forms are listed in sheets/sheets.json; the recognizer tries every
sheet it finds there and keeps whichever one actually aligns with the photo, so
a student never has to say which form they filled in.

The reference image is the green channel of a 200dpi render, auto-contrasted.
Green is the useful channel because the printing is dark in it whether the
sheet was printed in black or in a drop-out red that the red channel cannot
see; auto-contrast then puts both on the same footing, so one reference serves
both printings.

If the PDF was produced by tools/make_sheet.py, the as-drawn geometry sitting
next to it is used to check the extraction, oval for oval. That check is the
reason to trust this tool on a sheet whose generator we do not own.
"""
import collections
import json
import os
import subprocess
import sys

import pdfplumber
from PIL import Image, ImageOps

REF_DPI = 200
COLUMN_GAP = 15.0        # a horizontal gap this wide starts a new answer column

OVAL = {"w": 8.9, "h": 6.2}
TOL = 0.3

SHEETS = {
    "demo-form-a": {
        "name": "OMR demo answer sheet, form A",
        "row_pitch": 8.28,
        # name, y0, y1 (points from the page top), rows per column, options, prefix
        "sections": [
            ("part1", 172, 262, 10, 4, "p1"),
            ("part2", 308, 392, 9, 5, "p2"),
            ("part3", 436, 520, 9, 4, "p3"),
            ("part4", 564, 656, 10, 4, "p4"),
        ],
    },
    "demo-form-b": {
        "name": "OMR demo answer sheet, form B",
        "row_pitch": 9.6,
        "sections": [
            ("part1", 168, 290, 12, 5, "p1"),
            ("part2", 312, 434, 12, 5, "p2"),
            ("part3", 456, 560, 10, 4, "p3"),
            ("part4", 580, 684, 10, 4, "p4"),
        ],
    },
}

ODD_LETTERS = "ABCDE"
EVEN_LETTERS = "FGHJK"


def extract_ovals(page, oval, tol):
    return [((c["x0"] + c["x1"]) / 2.0, (c["top"] + c["bottom"]) / 2.0)
            for c in page.curves
            if abs(c["width"] - oval["w"]) < tol and abs(c["height"] - oval["h"]) < tol]


def build_section(points, y0, y1, rows_per_col, options, row_pitch, name):
    band = [p for p in points if y0 < p[1] < y1]
    rows = collections.defaultdict(list)
    for x, y in band:
        rows[round(y / row_pitch)].append((x, y))
    row_keys = sorted(rows)

    columns = []
    for x in sorted(set(round(x, 1) for x, _ in band)):
        if columns and x - columns[-1][-1] < COLUMN_GAP:
            columns[-1].append(x)
        else:
            columns.append([x])
    bad = [len(g) for g in columns if len(g) != options]
    if bad:
        raise SystemExit("%s: expected %d ovals per column, found groups of %s"
                         % (name, options, bad))

    questions = []
    for ci, group in enumerate(columns):
        for ri, rk in enumerate(row_keys):
            n = ci * rows_per_col + ri + 1
            row = sorted((x, y) for x, y in rows[rk] if group[0] - 1 < x < group[-1] + 1)
            if len(row) != options:
                continue
            letters = ODD_LETTERS if n % 2 else EVEN_LETTERS
            questions.append({"n": n, "choices": [
                {"l": letters[i], "x": round(row[i][0], 2), "y": round(row[i][1], 2)}
                for i in range(options)]})
    questions.sort(key=lambda q: q["n"])
    if [q["n"] for q in questions] != list(range(1, len(questions) + 1)):
        raise SystemExit("%s: question numbering is not contiguous" % name)
    return questions


def check_against_drawing(template, geometry_path):
    """Compare the extracted geometry with what make_sheet.py drew."""
    with open(geometry_path) as fh:
        drawn = json.load(fh)
    worst = 0.0
    checked = 0
    for name, sec in template["sections"].items():
        ref = drawn["sections"].get(name)
        if ref is None:
            raise SystemExit("%s: not in %s" % (name, geometry_path))
        if ref["count"] != sec["count"]:
            raise SystemExit("%s: drew %d questions, extracted %d"
                             % (name, ref["count"], sec["count"]))
        for a, b in zip(ref["questions"], sec["questions"]):
            if a["n"] != b["n"] or len(a["choices"]) != len(b["choices"]):
                raise SystemExit("%s: question %d does not line up" % (name, a["n"]))
            for ca, cb in zip(a["choices"], b["choices"]):
                if ca["l"] != cb["l"]:
                    raise SystemExit("%s q%d: drew %s, extracted %s"
                                     % (name, a["n"], ca["l"], cb["l"]))
                worst = max(worst, abs(ca["x"] - cb["x"]), abs(ca["y"] - cb["y"]))
                checked += 1
    print("  checked %d ovals against the drawing, worst disagreement %.3f pt"
          % (checked, worst))
    if worst > 0.05:
        raise SystemExit("extraction does not match the drawing")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in SHEETS:
        sys.exit(__doc__)
    sheet_id, pdf_path = sys.argv[1], sys.argv[2]
    spec = SHEETS[sheet_id]
    here = os.path.dirname(os.path.abspath(__file__))
    sheets_dir = os.path.join(os.path.dirname(here), "sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    page = pdfplumber.open(pdf_path).pages[0]
    points = extract_ovals(page, OVAL, TOL)
    print("%d ovals on the page" % len(points))

    template = {"id": sheet_id, "name": spec["name"],
                "page": {"width": float(page.width), "height": float(page.height)},
                "oval": OVAL, "sections": {}}
    for name, y0, y1, rows_per_col, options, prefix in spec["sections"]:
        questions = build_section(points, y0, y1, rows_per_col, options,
                                  spec["row_pitch"], name)
        xs = [c["x"] for q in questions for c in q["choices"]]
        ys = [c["y"] for q in questions for c in q["choices"]]
        template["sections"][name] = {
            "prefix": prefix, "count": len(questions), "options": options,
            "rowsPerColumn": rows_per_col,
            # the bbox reaches left of the first oval to take in the printed
            # question number, which gives cross-correlation something to bite on
            "bbox": [round(min(xs) - 30, 1), round(min(ys) - 8, 1),
                     round(max(xs) + 8, 1), round(max(ys) + 8, 1)],
            "questions": questions,
        }
        print("  %s: %d questions x %d" % (name, len(questions), options))

    geometry_path = os.path.join(sheets_dir, "%s-geometry.json" % sheet_id)
    if os.path.exists(geometry_path):
        check_against_drawing(template, geometry_path)

    out_json = os.path.join(sheets_dir, "%s.json" % sheet_id)
    with open(out_json, "w") as fh:
        json.dump(template, fh, separators=(",", ":"))
    print("wrote", out_json)

    tmp_prefix = os.path.join(sheets_dir, "_ref")
    subprocess.check_call(["pdftoppm", "-r", str(REF_DPI), "-png", "-f", "1", "-l", "1",
                           pdf_path, tmp_prefix])
    rendered = [f for f in os.listdir(sheets_dir)
                if f.startswith("_ref") and f.endswith(".png")][0]
    rendered = os.path.join(sheets_dir, rendered)
    green = ImageOps.autocontrast(Image.open(rendered).convert("RGB").split()[1], cutoff=0.5)
    out_png = os.path.join(sheets_dir, "%s-ref.png" % sheet_id)
    green.save(out_png, optimize=True)
    os.remove(rendered)
    print("wrote", out_png, green.size)

    index = os.path.join(sheets_dir, "sheets.json")
    entry = {"id": sheet_id, "template": "%s.json" % sheet_id,
             "reference": "%s-ref.png" % sheet_id}
    listing = []
    if os.path.exists(index):
        with open(index) as fh:
            listing = [e for e in json.load(fh) if e["id"] != sheet_id]
    listing.append(entry)
    listing.sort(key=lambda e: e["id"])   # stable order, so rebuilding one sheet
    with open(index, "w") as fh:          # does not reshuffle the file
        json.dump(listing, fh, indent=1)
    print("wrote", index)


if __name__ == "__main__":
    main()
