#!/usr/bin/env python3
"""
gen_synth.py - synthesise photographs of filled answer sheets, with ground truth.

    pip install pillow numpy opencv-python-headless   # plus `pdftoppm` on PATH
    python3 eval/gen_synth.py                         # -> eval/photos/*.jpg + *.json

There is no public corpus of phone photos of bubble sheets, and photographing a
few hundred by hand is not a test suite - it is an anecdote. So the sheets are
filled and photographed in software, which makes the ground truth exact and the
hard cases reproducible.

What each photo goes through, in order:

  1. The blank sheet, rendered from its own PDF at 200dpi.
  2. Marks drawn at the template's oval centres, jittered a couple of points off
     centre, with a random size, a random opacity and a few scribble strokes, so
     that a mark is never a clean filled ellipse. A fixed share of questions are
     left blank, double-marked, marked faintly, or double-marked with one of the
     two faint - the cases that decide whether a flag is raised.
  3. Optionally a photocopier: desaturate and crush the tone curve, which is what
     turns a drop-out red sheet into the grey one a student actually brings in.
  4. A photograph: the page laid on a desk, rotated, warped by a random
     perspective, lit by a gradient, sometimes half in shadow, blurred, grained,
     sometimes shot in landscape, then JPEG'd down to phone resolution.

The seed is an argument, so a run is reproducible; `--seed 7` gives a different
suite of the same shape for checking that a threshold was not tuned to one draw.
"""
import argparse
import json
import math
import os
import random
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHEETS = os.path.join(ROOT, "sheets")
RENDER = os.path.join(HERE, "_render")
PHOTOS = os.path.join(HERE, "photos")
S = 200 / 72.0                       # the 200dpi reference scale, px per point

# name, form, print, long side px, rotation deg, perspective, light falloff, blur,
# landscape, white desk, raking shadow
CASES = [
    ("a_red_clean",         "a", "red",       2600,  1.5, 0.010, 0.15, 0.8, False, False, False),
    ("a_red_tilted",        "a", "red",       3000,  6.0, 0.040, 0.35, 1.2, False, False, False),
    ("a_red_landscape_pen", "a", "red",       2400, -3.0, 0.030, 0.25, 1.0, True,  False, False),
    ("a_red_lowres",        "a", "red",       1500,  3.0, 0.020, 0.20, 0.7, False, False, False),
    ("a_red_whitedesk",     "a", "red",       2400,  4.0, 0.030, 0.20, 0.9, False, True,  False),
    ("a_red_shadow",        "a", "red",       2600, -2.0, 0.020, 0.20, 0.9, False, False, True),
    ("a_copy_clean",        "a", "photocopy", 2600,  2.0, 0.015, 0.20, 0.9, False, False, False),
    ("a_copy_tilted_light", "a", "photocopy", 2200, -5.0, 0.040, 0.40, 1.4, False, False, False),
    ("a_copy_whitedesk",    "a", "photocopy", 2600,  3.0, 0.030, 0.25, 0.7, False, True,  False),
    ("a_black_clean",       "a", "black",     2600,  2.5, 0.020, 0.20, 0.9, False, False, False),
    ("a_black_shadow",      "a", "black",     2000, -4.0, 0.035, 0.30, 1.1, False, False, True),
    ("b_red_clean",         "b", "red",       2600,  1.0, 0.015, 0.15, 0.8, False, False, False),
    ("b_copy_tilted",       "b", "photocopy", 2400, -6.0, 0.040, 0.35, 1.2, False, False, False),
    ("b_black_landscape",   "b", "black",     2800,  4.0, 0.030, 0.25, 1.0, True,  False, False),
]

PENCILS = [(55, 55, 60), (70, 70, 75), (40, 40, 45), (90, 90, 95), (20, 25, 90)]


def render(form, ink):
    """200dpi RGB render of a blank sheet, cached under eval/_render."""
    os.makedirs(RENDER, exist_ok=True)
    out = os.path.join(RENDER, "demo-form-%s-%s.png" % (form, ink))
    if not os.path.exists(out):
        suffix = "" if ink == "black" else "-red"
        pdf = os.path.join(SHEETS, "demo-form-%s%s.pdf" % (form, suffix))
        prefix = os.path.join(RENDER, "_tmp")
        subprocess.check_call(["pdftoppm", "-r", "200", "-png", "-f", "1", "-l", "1", pdf, prefix])
        tmp = [f for f in os.listdir(RENDER) if f.startswith("_tmp")][0]
        os.rename(os.path.join(RENDER, tmp), out)
    return Image.open(out).convert("RGB")


def photocopy(img):
    """A drop-out red sheet run through a copier: no colour left to drop out."""
    a = np.asarray(ImageOps.grayscale(img)).astype(np.float32)
    a = 255 - np.clip((255 - a) * 2.6, 0, 255)
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def fill_sheet(img, template, pencil, rng):
    """Draw pencil marks on the rendered sheet; return the ground truth."""
    draw = ImageDraw.Draw(img, "RGBA")
    ow, oh = template["oval"]["w"] * S, template["oval"]["h"] * S
    truth = {}
    for sec in template["sections"].values():
        for q in sec["questions"]:
            r = rng.random()
            nchoices = len(q["choices"])
            if r < 0.05:
                picks, status = [], "blank"
            elif r < 0.08:
                picks, status = rng.sample(range(nchoices), 2), "multiple"
            elif r < 0.10:
                picks, status = rng.sample(range(nchoices), 2), "double-light"
            elif r < 0.12:
                picks, status = [rng.randrange(nchoices)], "light"
            else:
                picks, status = [rng.randrange(nchoices)], "ok"
            truth[sec["prefix"] + "_" + str(q["n"])] = {
                "pick": [q["choices"][i]["l"] for i in picks], "status": status}
            for i in picks:
                c = q["choices"][i]
                cx = c["x"] * S + rng.uniform(-2, 2)
                cy = c["y"] * S + rng.uniform(-1.5, 1.5)
                rx = ow / 2 * rng.uniform(0.8, 1.05)
                ry = oh / 2 * rng.uniform(0.8, 1.1)
                if status == "light":
                    alpha = 60
                elif status == "double-light" and i == picks[1]:
                    alpha = 110
                else:
                    alpha = rng.randint(190, 255)
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=pencil + (alpha,))
                if status != "light":
                    for _ in range(6):   # scribble texture, so no mark is a clean ellipse
                        x0 = cx + rng.uniform(-rx, rx)
                        y0 = cy + rng.uniform(-ry, ry)
                        draw.line([x0, y0, x0 + rng.uniform(-4, 4), y0 + rng.uniform(-2, 2)],
                                  fill=pencil + (120,), width=1)
    return truth


def photograph(page, long_side, rot_deg, persp, light, blur,
               landscape=False, white_desk=False, shadow=False, rng=random):
    pw, ph = page.size
    scene_w, scene_h = int(pw * 1.25), int(ph * 1.18)

    if white_desk:
        bg = np.random.normal(238, 4, (scene_h, scene_w, 3)).astype(np.float32)
    else:
        bg = np.random.normal(120, 8, (scene_h, scene_w, 3)).astype(np.float32)
        bg[:, :, 0] += 25
        bg[:, :, 2] -= 20            # a brown desk, so the page edge is a real edge
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    base = np.float32([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    th = math.radians(rot_deg)
    R = np.float32([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    jitter = np.float32([[rng.uniform(-persp, persp) * pw, rng.uniform(-persp, persp) * ph]
                         for _ in range(4)])
    dst = (base @ R.T) * 0.92 + jitter + np.float32([scene_w / 2, scene_h / 2])
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    pg = cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(pg, M, (scene_w, scene_h), borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (scene_w, scene_h))
    comp = np.where(mask[:, :, None] > 0, warped,
                    cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)).astype(np.float32)

    yy, xx = np.mgrid[0:scene_h, 0:scene_w]
    comp *= (1 - light * ((xx / scene_w) * 0.6 + (yy / scene_h) * 0.4))[:, :, None]
    if shadow:
        sh = np.ones((scene_h, scene_w), np.float32)
        sh[(xx + yy * 0.7) > scene_w * 0.75] = 0.55      # the photographer's own shadow
        comp *= cv2.GaussianBlur(sh, (0, 0), 25)[:, :, None]
    comp = np.clip(comp + np.random.normal(0, 4, comp.shape), 0, 255).astype(np.uint8)
    if blur > 0:
        comp = cv2.GaussianBlur(comp, (0, 0), blur)
    if landscape:
        comp = cv2.rotate(comp, cv2.ROTATE_90_CLOCKWISE)
    h, w = comp.shape[:2]
    f = long_side / max(w, h)
    return cv2.resize(comp, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def negative_control(rng):
    """A photo that is not an answer sheet at all. The engine must refuse it."""
    page = Image.new("RGB", (1700, 2200), (250, 249, 246))
    d = ImageDraw.Draw(page)
    for i in range(60):
        y = 180 + i * 30
        d.line([150, y, 150 + rng.randint(600, 1400), y], fill=(70, 70, 70), width=4)
    return photograph(page, 2400, 3.0, 0.03, 0.2, 0.9, rng=rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    os.makedirs(PHOTOS, exist_ok=True)

    templates = {f: json.load(open(os.path.join(SHEETS, "demo-form-%s.json" % f)))
                 for f in ("a", "b")}

    for (name, form, ink, long_side, rot, persp, light, blur,
         landscape, white_desk, shadow) in CASES:
        page = render(form, "black" if ink == "black" else "red")
        if ink == "photocopy":
            page = photocopy(page)
        else:
            page = page.copy()
        pencil = PENCILS[rng.randrange(len(PENCILS))]
        truth = fill_sheet(page, templates[form], pencil, rng)
        photo = photograph(page, long_side, rot, persp, light, blur,
                           landscape, white_desk, shadow, rng)
        cv2.imwrite(os.path.join(PHOTOS, name + ".jpg"), photo,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        json.dump({"sheet": "demo-form-" + form, "expect": "read", "questions": truth},
                  open(os.path.join(PHOTOS, name + ".json"), "w"))
        print("%-22s %s  %dx%d  %d questions" % (name, ink, photo.shape[1], photo.shape[0],
                                                 len(truth)))

    photo = negative_control(rng)
    cv2.imwrite(os.path.join(PHOTOS, "z_not_a_sheet.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 82])
    json.dump({"expect": "reject"}, open(os.path.join(PHOTOS, "z_not_a_sheet.json"), "w"))
    print("%-22s negative control" % "z_not_a_sheet")


if __name__ == "__main__":
    main()
