#!/usr/bin/env python3
"""
run_test.py - run the recognizer over eval/photos and score it against truth.

    pip install playwright && playwright install chromium
    python3 eval/gen_synth.py
    python3 eval/run_test.py                      # all photos
    python3 eval/run_test.py a_copy_clean         # one photo
    python3 eval/run_test.py --show p1_17,p2_3    # dump the measured values for a question

The engine is browser code, so it is tested in a browser: a headless Chromium
loads src/scanEngine.js from a local static server and the photos are handed to
it exactly as a file input would. No part of the pipeline is stubbed.

A question counts as an error when:

  * truth marked one oval and the engine picked a different one, or none;
  * truth left it blank and the engine picked something - a false positive, the
    expensive kind, because a student skimming a review screen will not notice
    an answer they never gave;
  * truth marked two ovals and the engine did not flag it.

Note what is *not* an error: a correct pick that the engine also flagged for
checking. Flags are cheap and a human clears them, so the flag count is
reported next to the error count rather than folded into it. That asymmetry is
the whole design: never be quietly wrong, be loudly unsure.
"""
import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.request

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHOTOS = os.path.join(HERE, "photos")
CACHE = os.path.join(HERE, "_cache")
OPENCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js"

# Ask for more questions than either form prints. The engine clamps each section
# to what the sheet actually has, which is the real case: one host application,
# several sheet layouts, tests of different lengths.
WANTED = {"part1": {"count": 60, "options": 5}, "part2": {"count": 60, "options": 5},
          "part3": {"count": 40, "options": 4}, "part4": {"count": 40, "options": 4}}

PAGE = """<!doctype html><html><body>
<script src="/src/scanEngine.js"></script>
<script>
window.runScan = async function (url, wanted) {
  const blob = await (await fetch(url)).blob();
  const t0 = performance.now();
  ScanEngine.onStatus = function (s) { console.log('status: ' + s); };
  const r = await ScanEngine.scan({
    file: blob, sections: wanted,
    urls: { opencv: '/eval/_cache/opencv.js', sheets: '/sheets/sheets.json' }
  });
  return {
    ms: performance.now() - t0, sheet: r.sheet, printType: r.printType,
    registration: r.registration, threshold: r.threshold, clusters: r.clusters,
    sections: r.sections.map(function (s) {
      return { name: s.name, score: s.score, aligned: s.aligned, columns: s.columns,
        questions: s.questions.map(function (q) {
          return { name: q.name, pick: q.pick, status: q.status,
                   values: q.values.map(function (v) { return +v.toFixed(3); }) };
        }) };
    })
  };
};
</script></body></html>"""


def ensure_opencv():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "opencv.js")
    if not os.path.exists(path):
        print("fetching opencv.js (once, ~10MB) ...")
        urllib.request.urlretrieve(OPENCV_URL, path)
    return path


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=ROOT, **k)

        def log_message(self, *a):
            pass

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = Server(("127.0.0.1", 0), Quiet)   # port 0: let the OS pick a free one
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def score(result, truth):
    """-> (errors, flags, question count). See the module docstring for the rules."""
    errors, flags, total = [], 0, 0
    for sec in result["sections"]:
        for q in sec["questions"]:
            t = truth.get(q["name"])
            if t is None:
                continue
            total += 1
            if q["status"] in ("uncertain", "multiple", "unreadable"):
                flags += 1
            if t["status"] in ("ok", "light"):
                if q["pick"] != t["pick"][0]:
                    errors.append((q["name"], "wanted " + t["pick"][0], q))
            elif t["status"] == "blank":
                if q["pick"] is not None:
                    errors.append((q["name"], "wanted blank", q))
            else:                                  # multiple / double-light
                if q["status"] != "multiple":
                    errors.append((q["name"], "wanted a multiple-mark flag", q))
    return errors, flags, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="photo names to run (default: all)")
    ap.add_argument("--show", default="", help="comma-separated question names to dump")
    ap.add_argument("--verbose", action="store_true", help="print engine status lines")
    args = ap.parse_args()

    ensure_opencv()
    with open(os.path.join(HERE, "_page.html"), "w") as fh:
        fh.write(PAGE)
    httpd = serve()

    names = args.names or sorted(f[:-4] for f in os.listdir(PHOTOS) if f.endswith(".jpg"))
    show = set(n for n in args.show.split(",") if n)
    total_err = total_q = total_flag = 0
    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--no-proxy-server"])
        page = browser.new_page()
        page.set_default_timeout(300000)
        if args.verbose:
            page.on("console", lambda m: print("   ", m.text))
        page.goto("http://127.0.0.1:%d/eval/_page.html" % httpd.server_address[1])

        for name in names:
            with open(os.path.join(PHOTOS, name + ".json")) as fh:
                spec = json.load(fh)
            url = "/eval/photos/%s.jpg" % name
            try:
                r = page.evaluate("([u, w]) => runScan(u, w)", [url, WANTED])
            except Exception as exc:
                if spec.get("expect") == "reject":
                    print("%-22s correctly refused" % name)
                else:
                    print("%-22s FAILED: %s" % (name, str(exc).split("\n")[0][:120]))
                    failures.append(name)
                continue
            if spec.get("expect") == "reject":
                print("%-22s SHOULD HAVE BEEN REFUSED (read as %s)" % (name, r["sheet"]["id"]))
                failures.append(name)
                continue

            errors, flags, nq = score(r, spec["questions"])
            right_sheet = r["sheet"]["id"] == spec["sheet"]
            total_err += len(errors)
            total_q += nq
            total_flag += flags
            if not right_sheet:
                failures.append(name)
            print("%-22s %s%s  %4.1fs  %s  reg=%s/%d fit=%.2f  T=%.2f  errors=%d/%d flagged=%d"
                  % (name, r["sheet"]["id"], "" if right_sheet else " (WRONG SHEET)",
                     r["ms"] / 1000, r["printType"], r["registration"]["method"],
                     r["registration"]["inliers"], r["registration"]["fit"],
                     r["threshold"], len(errors), nq, flags))
            if os.environ.get("COLUMNS_DEBUG"):
                for sec in r["sections"]:
                    print("   ", sec["name"], json.dumps(sec["columns"]))
            for sec in r["sections"]:
                if not sec["aligned"]:
                    print("    section %s did not align (fit %.2f)" % (sec["name"], sec["score"]))
            for qname, want, q in errors[:10]:
                print("    %-8s %s, read %s (%s) %s" % (qname, want, q["pick"], q["status"],
                                                        q["values"]))
            if show:
                for sec in r["sections"]:
                    for q in sec["questions"]:
                        if q["name"] in show:
                            print("    %-8s %s %s %s" % (q["name"], q["pick"], q["status"],
                                                         q["values"]))
        browser.close()
    httpd.shutdown()

    print("\n%d errors in %d questions (%.3f%%), %d flagged for checking (%.1f%%)"
          % (total_err, total_q, 100.0 * total_err / max(total_q, 1),
             total_flag, 100.0 * total_flag / max(total_q, 1)))
    if failures:
        print("failed photos: " + ", ".join(failures))
    sys.exit(1 if (failures or total_err) else 0)


if __name__ == "__main__":
    main()
