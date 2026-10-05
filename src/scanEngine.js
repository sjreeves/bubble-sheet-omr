/*
 * scanEngine.js
 * Bubble-sheet recognition engine: a photograph of a filled answer sheet in,
 * a per-question answer with a confidence flag out, entirely in the browser.
 *
 * Pipeline (all in the browser, OpenCV.js):
 *   1. Photo -> canvas (EXIF orientation handled by the browser), long side capped.
 *   2. Registration: ORB features on the photo's green channel (printing is dark
 *      in green whether the sheet is red-ink or black) matched against a
 *      reference render of the blank sheet; RANSAC homography; warp photo into
 *      the reference frame. Fallback: page-quad detection then ORB again.
 *   3. Per-section refinement: normalized cross-correlation of the reference
 *      section against the warped photo gives a residual (dx,dy) and a fit score.
 *   4. Print-type detection (red drop-out ink vs black) decides the mark channel:
 *      red channel (printing vanishes, pencil stays) or grayscale.
 *   5. Background-normalized darkness sampled inside each oval -> fill ratio.
 *   6. Otsu threshold over all ovals; per-question decision with confidence flags.
 *
 * Exposes window.ScanEngine. No framework, no build step, no network beyond
 * the one OpenCV.js fetch.
 */
(function (global) {
	"use strict";

	var REF_DPI = 200;
	var REF_SCALE = REF_DPI / 72;          // px per PDF point
	var PHOTO_MAX_SIDE = 3000;             // measured on real photos: 3000 beats 2200 for alignment
	var REF_FEATURES = 8000;               // ORB features kept per reference sheet
	var PHOTO_FEATURES = 12000;            // ORB features kept for the photo
	var SELECT_FEATURES = 3000;            // cheap pass that only decides which sheet layout it is
	var RATIO_TEST = 0.85;                 // Lowe ratio; 0.75 threw away too much on photocopies
	var MIN_MATCHES = 30;
	var MIN_INLIERS = 40;
	var SECTION_MIN_SCORE = 0.22;          // TM_CCOEFF_NORMED below this = column not aligned
	var GOOD_FIT = 0.45;                   // mean column fit at which a sheet candidate is accepted outright
	// Faintest mark accepted (row-relative fill, after per-letter bias correction). An oval
	// this far above its row-mates, and 2.5x the runner-up, is pre-selected and flagged
	// "check". Measured on a real photocopy: blank rows peak ~0.03 after correction, faded
	// pencil marks 0.08-0.16, full marks ~0.5. Lower = catches fainter marks, more flags.
	var MIN_MARK = 0.09;
	var OPENCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js";

	var cvReadyPromise = null;
	// Known sheet layouts (scan/sheets.json). Each: {id, template, refMat, refFeatures}.
	// The photo is matched against every sheet and the best alignment wins.
	var sheets = null;
	var template = null;                   // the selected sheet's template
	var refMat = null;                     // the selected sheet's reference (green channel, CV_8UC1)

	function log(msg) {
		if (global.ScanEngine.onStatus) {
			global.ScanEngine.onStatus(msg);
		}
	}

	/* ------------------------------------------------------------------ */
	/* Loading                                                            */
	/* ------------------------------------------------------------------ */

	function loadOpenCV(url) {
		if (cvReadyPromise) {
			return cvReadyPromise;
		}
		cvReadyPromise = new Promise(function (resolve, reject) {
			if (global.cv && global.cv.Mat) {
				resolve(true);
				return;
			}
			var s = document.createElement("script");
			s.async = true;
			s.src = url || OPENCV_URL;
			s.onload = function () {
				var c = global.cv;
				// NB: the Emscripten module is itself a "thenable"; never resolve a
				// Promise with it or the resolution recurses forever.
				if (c && typeof c.then === "function") {
					c.then(function (m) { global.cv = m; resolve(true); });
				}
				else if (c && c.Mat) {
					resolve(true);
				}
				else if (c) {
					c.onRuntimeInitialized = function () { resolve(true); };
				}
				else {
					reject(new Error("OpenCV failed to initialize"));
				}
			};
			s.onerror = function () { reject(new Error("Could not download the image library. Check your connection and try again.")); };
			document.head.appendChild(s);
		});
		return cvReadyPromise;
	}

	function loadImageElement(src) {
		return new Promise(function (resolve, reject) {
			var img = new Image();
			img.onload = function () { resolve(img); };
			img.onerror = function () { reject(new Error("Could not load image")); };
			img.src = src;
		});
	}

	function fetchJson(url, what) {
		return fetch(url).then(function (r) {
			if (!r.ok) { throw new Error("Could not load " + what); }
			return r.json();
		});
	}

	/* Load scan/sheets.json and every sheet's template + reference features (once). */
	function loadSheets(url) {
		if (sheets) {
			return Promise.resolve(sheets);
		}
		var listUrl = new URL(url || "scan/sheets.json", document.baseURI);
		// entries in sheets.json are relative to sheets.json itself
		return fetchJson(listUrl.href, "sheet list").then(function (list) {
			return Promise.all(list.map(function (entry) {
				var tUrl = new URL(entry.template, listUrl).href, rUrl = new URL(entry.reference, listUrl).href;
				return Promise.all([fetchJson(tUrl, "sheet template"), loadImageElement(rUrl)])
					.then(function (parts) {
						var img = parts[1];
						var c = document.createElement("canvas");
						c.width = img.naturalWidth;
						c.height = img.naturalHeight;
						c.getContext("2d").drawImage(img, 0, 0);
						var rgba = cv.imread(c);
						var gray = new cv.Mat();
						cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
						rgba.delete();
						// Features are detected on a slightly blurred copy: the render is crisper than
						// any photo, and matching improved measurably (real photocopy: 93 -> 118 inliers).
						var soft = new cv.Mat();
						cv.GaussianBlur(gray, soft, new cv.Size(0, 0), 1.0);
						var feats = detectFeatures(soft, REF_FEATURES);
						soft.delete();
						return { id: entry.id, template: parts[0], refMat: gray, refFeatures: feats };
					});
			}));
		}).then(function (loaded) {
			if (!loaded.length) { throw new Error("No sheet layouts configured"); }
			sheets = loaded;
			return sheets;
		});
	}

	function selectSheet(sheet) {
		template = sheet.template;
		refMat = sheet.refMat;
	}

	/* ------------------------------------------------------------------ */
	/* Photo handling                                                      */
	/* ------------------------------------------------------------------ */

	function fileToCanvas(file) {
		var url = URL.createObjectURL(file);
		return loadImageElement(url).then(function (img) {
			URL.revokeObjectURL(url);
			var w = img.naturalWidth, h = img.naturalHeight;
			var scale = Math.min(1, PHOTO_MAX_SIDE / Math.max(w, h));
			var c = document.createElement("canvas");
			c.width = Math.round(w * scale);
			c.height = Math.round(h * scale);
			var ctx = c.getContext("2d");
			ctx.drawImage(img, 0, 0, c.width, c.height);
			return c;
		});
	}

	/* ------------------------------------------------------------------ */
	/* Registration                                                        */
	/* ------------------------------------------------------------------ */

	function detectFeatures(gray8u, nfeatures) {
		// (opencv.js cannot bind the ScoreType enum, so only the first six ctor args are usable.)
		// Measured: many features + a low FAST threshold on the *raw* channel is what works on
		// real photocopies; histogram equalization cuts inliers ~5x, so it is not used.
		var orb = new cv.ORB(nfeatures || PHOTO_FEATURES, 1.2, 8, 31, 0, 2);
		orb.setFastThreshold(7);
		var kp = new cv.KeyPointVector();
		var desc = new cv.Mat();
		var mask = new cv.Mat();
		orb.detectAndCompute(gray8u, mask, kp, desc);
		mask.delete();
		orb.delete();
		return { kp: kp, desc: desc };
	}

	/* Match photo features (f = {kp, desc}) to one sheet's reference;
	   returns {H (photo->ref), inliers, matches} or null */
	function matchToReference(f, photoCols, photoRows, sheet) {
		var refFeatures = sheet.refFeatures;
		var result = null;
		if (f.desc.rows >= MIN_MATCHES && refFeatures.desc.rows >= MIN_MATCHES) {
			var matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
			var knn = new cv.DMatchVectorVector();
			matcher.knnMatch(f.desc, refFeatures.desc, knn, 2);
			var src = [], dst = [];
			for (var i = 0; i < knn.size(); i++) {
				var pair = knn.get(i);
				if (pair.size() < 2) { continue; }
				var m = pair.get(0), n = pair.get(1);
				if (m.distance < RATIO_TEST * n.distance) {
					var p = f.kp.get(m.queryIdx).pt;
					var q = refFeatures.kp.get(m.trainIdx).pt;
					src.push(p.x, p.y);
					dst.push(q.x, q.y);
				}
			}
			knn.delete();
			matcher.delete();
			var count = src.length / 2;
			if (count >= MIN_MATCHES) {
				var srcMat = cv.matFromArray(count, 1, cv.CV_32FC2, src);
				var dstMat = cv.matFromArray(count, 1, cv.CV_32FC2, dst);
				var mask = new cv.Mat();
				var H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 4.0, mask, 5000, 0.995);
				var inliers = 0;
				for (var k = 0; k < mask.rows; k++) { if (mask.data[k]) { inliers++; } }
				srcMat.delete(); dstMat.delete(); mask.delete();
				if (!H.empty() && inliers >= MIN_INLIERS && saneHomography(H, photoCols, photoRows, sheet.refMat)) {
					result = { H: H, inliers: inliers, matches: count };
				}
				else {
					H.delete();
				}
			}
		}
		return result;
	}

	/* Reject homographies that fold or wildly distort the page */
	function saneHomography(H, w, h, ref) {
		var d = H.data64F;
		function map(x, y) {
			var z = d[6] * x + d[7] * y + d[8];
			return [(d[0] * x + d[1] * y + d[2]) / z, (d[3] * x + d[4] * y + d[5]) / z];
		}
		var c = [map(0, 0), map(w, 0), map(w, h), map(0, h)];
		// signed area of mapped image quad must be positive-ish and finite; orientation consistent
		var area = 0;
		for (var i = 0; i < 4; i++) {
			var a = c[i], b = c[(i + 1) % 4];
			area += a[0] * b[1] - b[0] * a[1];
		}
		if (!isFinite(area) || Math.abs(area) < 0.05 * ref.cols * ref.rows) { return false; }
		if (Math.abs(area) > 40 * ref.cols * ref.rows) { return false; }
		return true;
	}

	/* Fallback: find the sheet as the largest 4-corner contour, warp to reference size */
	function findPageQuad(gray) {
		var small = new cv.Mat();
		var f = 800 / Math.max(gray.cols, gray.rows);
		cv.resize(gray, small, new cv.Size(0, 0), f, f, cv.INTER_AREA);
		var blur = new cv.Mat();
		cv.GaussianBlur(small, blur, new cv.Size(5, 5), 0);
		var edges = new cv.Mat();
		cv.Canny(blur, edges, 40, 120);
		var kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
		cv.dilate(edges, edges, kernel);
		var contours = new cv.MatVector();
		var hier = new cv.Mat();
		cv.findContours(edges, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
		var best = null, bestArea = 0.2 * small.cols * small.rows;
		for (var i = 0; i < contours.size(); i++) {
			var cnt = contours.get(i);
			var area = cv.contourArea(cnt);
			if (area < bestArea) { continue; }
			var peri = cv.arcLength(cnt, true);
			var approx = new cv.Mat();
			cv.approxPolyDP(cnt, approx, 0.03 * peri, true);
			if (approx.rows === 4) {
				var pts = [];
				for (var j = 0; j < 4; j++) { pts.push([approx.data32S[j * 2] / f, approx.data32S[j * 2 + 1] / f]); }
				best = pts; bestArea = area;
			}
			approx.delete();
		}
		small.delete(); blur.delete(); edges.delete(); kernel.delete(); contours.delete(); hier.delete();
		if (!best) { return null; }
		// order corners: tl, tr, br, bl
		best.sort(function (a, b) { return (a[0] + a[1]) - (b[0] + b[1]); });
		var tl = best[0], br = best[3];
		var rest = [best[1], best[2]].sort(function (a, b) { return (a[0] - a[1]) - (b[0] - b[1]); });
		var bl = rest[0], tr = rest[1];
		var W = sheets[0].refMat.cols, Hh = sheets[0].refMat.rows;
		var src = cv.matFromArray(4, 1, cv.CV_32FC2, [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]);
		var dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W, 0, W, Hh, 0, Hh]);
		var P = cv.getPerspectiveTransform(src, dst);
		src.delete(); dst.delete();
		return P;
	}

	/* Matrix product A*B for 3x3 CV_64F mats */
	function composeH(A, B) {
		var a = A.data64F, b = B.data64F, out = [];
		for (var r = 0; r < 3; r++) {
			for (var c = 0; c < 3; c++) {
				out.push(a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]);
			}
		}
		return cv.matFromArray(3, 3, cv.CV_64F, out);
	}

	function warpTo(refSizeSrc, H) {
		var out = new cv.Mat();
		var border = refSizeSrc.channels() === 1 ? new cv.Scalar(255) : new cv.Scalar(255, 255, 255, 255);
		var ref = refMat || sheets[0].refMat;
		cv.warpPerspective(refSizeSrc, out, H, new cv.Size(ref.cols, ref.rows), cv.INTER_LINEAR, cv.BORDER_CONSTANT, border);
		return out;
	}

	/* Register photo (RGBA mat). Returns {H, inliers, method} or throws. */
	/*
	 * Registration candidates for a photo, produced lazily (each full feature match
	 * costs a few seconds on a phone, so the caller stops as soon as one fits well):
	 *   1. direct ORB match against each sheet, most likely layout first;
	 *   2. page-quad pre-warp, then ORB against each sheet again.
	 * next() returns {H (photo->ref), inliers, sheet, method} or null; close() frees.
	 */
	function createRegistrar(rgba) {
		var chans = new cv.MatVector();
		cv.split(rgba, chans);
		var green = chans.get(1).clone();
		chans.delete();
		log("Locating the answer sheet...");
		var feats = detectFeatures(green);
		var order = orderSheets(feats, green.cols, green.rows);
		var stage = 0, idx = 0, quad = null, pre = null, preFeats = null, preOrder = null;

		function nextDirect() {
			while (idx < order.length) {
				var sheet = order[idx++];
				var r = matchToReference(feats, green.cols, green.rows, sheet);
				if (r) { return { H: r.H, inliers: r.inliers, sheet: sheet, method: "direct" }; }
			}
			return null;
		}
		function nextQuad() {
			if (stage === 1) {
				stage = 2; idx = 0;
				quad = findPageQuad(green);
				if (!quad) { return null; }
				pre = warpTo(green, quad);
				preFeats = detectFeatures(pre);
				preOrder = orderSheets(preFeats, pre.cols, pre.rows);
			}
			if (!quad) { return null; }
			while (idx < preOrder.length) {
				var sheet = preOrder[idx++];
				var r = matchToReference(preFeats, pre.cols, pre.rows, sheet);
				if (r) {
					var H = composeH(r.H, quad);
					r.H.delete();
					return { H: H, inliers: r.inliers, sheet: sheet, method: "quad+orb" };
				}
			}
			return null;
		}
		return {
			next: function () {
				var c = null;
				if (stage === 0) { c = nextDirect(); if (!c) { stage = 1; } }
				if (!c && stage >= 1) { c = nextQuad(); }
				return c;
			},
			close: function () {
				feats.kp.delete(); feats.desc.delete(); green.delete();
				if (quad) { quad.delete(); }
				if (pre) { pre.delete(); }
				if (preFeats) { preFeats.kp.delete(); preFeats.desc.delete(); }
			}
		};
	}

	/* Cheap pass deciding which sheet layout to try first (subset of descriptors). */
	function orderSheets(f, cols, rows) {
		if (sheets.length < 2) { return sheets.slice(); }
		var sub = { kp: f.kp, desc: f.desc.rowRange(0, Math.min(SELECT_FEATURES, f.desc.rows)) };
		var scores = sheets.map(function (sheet) {
			var refSub = { kp: sheet.refFeatures.kp, desc: sheet.refFeatures.desc.rowRange(0, Math.min(SELECT_FEATURES, sheet.refFeatures.desc.rows)) };
			var r = matchToReference(sub, cols, rows, { refFeatures: refSub, refMat: sheet.refMat });
			var n = r ? r.inliers : 0;
			if (r) { r.H.delete(); }
			refSub.desc.delete();
			return { sheet: sheet, n: n };
		});
		sub.desc.delete();
		scores.sort(function (x, y) { return y.n - x.n; });
		return scores.map(function (x) { return x.sheet; });
	}

	/* ------------------------------------------------------------------ */
	/* Section refinement                                                  */
	/* ------------------------------------------------------------------ */

	function sectionRectPx(sec) {
		var b = sec.bbox;
		return {
			x: Math.round(b[0] * REF_SCALE), y: Math.round(b[1] * REF_SCALE),
			w: Math.round((b[2] - b[0]) * REF_SCALE), h: Math.round((b[3] - b[1]) * REF_SCALE)
		};
	}

	/* Sections of the selected sheet that the test type uses. */
	function buildSections(wanted) {
		var secs = [];
		Object.keys(template.sections).forEach(function (name) {
			var ts = template.sections[name];
			var want = wanted && wanted[name];
			if (!want || !want.count) { return; }
			var count = Math.min(want.count, ts.count);
			secs.push({ name: name, prefix: ts.prefix, tmpl: ts, count: count, sheetCount: ts.count, sheetOptions: ts.options, formOptions: want.options || 4, rect: sectionRectPx(ts) });
		});
		return secs;
	}

	/* Row and choice spacing of a printed section, in reference pixels. */
	function gridPitch(ts) {
		var q0 = ts.questions[0];
		var q1 = ts.questions.length > 1 ? ts.questions[1] : q0;   // questions 1 and 2 share a column
		var row = Math.abs(q1.choices[0].y - q0.choices[0].y) * REF_SCALE;
		var option = q0.choices.length > 1 ? Math.abs(q0.choices[1].x - q0.choices[0].x) * REF_SCALE : 20;
		return { row: row || 20, option: option || 20 };
	}

	/* Refine every printed column of every section against the reference; returns the
	   mean column fit (TM_CCOEFF_NORMED) as the candidate's quality. */
	function refineAll(secs, warpedGreen) {
		var total = 0, n = 0;
		secs.forEach(function (s) {
			var cols = {};
			s.tmpl.questions.forEach(function (q) {
				var ci = Math.floor((q.n - 1) / s.tmpl.rowsPerColumn);
				var c = cols[ci] || (cols[ci] = { xs: [], ys: [] });
				q.choices.forEach(function (ch) { c.xs.push(ch.x); c.ys.push(ch.y); });
			});
			s.columns = {};
			var bestScore = 0;
			Object.keys(cols).forEach(function (ci) {
				var c = cols[ci];
				var x0 = Math.min.apply(null, c.xs), x1 = Math.max.apply(null, c.xs), y0 = Math.min.apply(null, c.ys), y1 = Math.max.apply(null, c.ys);
				var rect = { x: Math.round((x0 - 28) * REF_SCALE), y: Math.round((y0 - 7) * REF_SCALE), w: Math.round((x1 - x0 + 36) * REF_SCALE), h: Math.round((y1 - y0 + 14) * REF_SCALE) };
				var r = refineSection(warpedGreen, rect, 24);
				s.columns[ci] = { dx: r.dx, dy: r.dy, score: r.score, ok: r.score >= SECTION_MIN_SCORE };
				bestScore = Math.max(bestScore, r.score); total += r.score; n++;
			});
			s.aligned = bestScore >= SECTION_MIN_SCORE;
			var med = function (arr) { arr = arr.slice().sort(function (a, b) { return a - b; }); return arr.length ? arr[Math.floor(arr.length / 2)] : 0; };
			var okCols = Object.keys(s.columns).filter(function (k) { return s.columns[k].ok; });
			var consensus = function () {
				s.dx = med(okCols.map(function (k) { return s.columns[k].dx; }));
				s.dy = med(okCols.map(function (k) { return s.columns[k].dy; }));
			};
			consensus();
			// A column of ovals repeats every row, so the correlation surface has a
			// second peak one row pitch away that is almost as strong as the right
			// one. A column that latches onto it scores well and reads every answer
			// in that column off by a row - the worst kind of failure, because
			// nothing about it looks uncertain. The columns of one section were
			// printed on one plate, so their residuals have to agree: any column
			// further from the section median than half a row, or half an oval pitch
			// across, is rejected and takes the median instead.
			var pitch = gridPitch(s.tmpl);
			okCols = okCols.filter(function (k) {
				var c = s.columns[k];
				if (Math.abs(c.dy - s.dy) <= pitch.row / 2 && Math.abs(c.dx - s.dx) <= pitch.option / 2) { return true; }
				c.ok = false;
				c.outlier = true;
				return false;
			});
			consensus();
			var scores = okCols.map(function (k) { return s.columns[k].score; });
			s.score = scores.length ? scores.reduce(function (a, b) { return a + b; }, 0) / scores.length : 0;
			Object.keys(s.columns).forEach(function (k) {
				if (!s.columns[k].ok) { s.columns[k].dx = s.dx; s.columns[k].dy = s.dy; }
			});
		});
		return n ? total / n : 0;
	}

	/* Cross-correlate the reference section crop against the warped green channel.
	   Returns {dx, dy, score}. */
	function refineSection(warpedGreen, rect, margin) {
		var half = 2;
		function ccorr(img, tmplImg, r, mg, scale) {
			var rx = Math.max(0, Math.round((r.x - mg) * scale)), ry = Math.max(0, Math.round((r.y - mg) * scale));
			var rw = Math.min(img.cols - rx, Math.round((r.w + 2 * mg) * scale));
			var rh = Math.min(img.rows - ry, Math.round((r.h + 2 * mg) * scale));
			var tx = Math.round(r.x * scale), ty = Math.round(r.y * scale);
			var tw = Math.round(r.w * scale), th = Math.round(r.h * scale);
			var search = img.roi(new cv.Rect(rx, ry, rw, rh));
			var tmpl = tmplImg.roi(new cv.Rect(tx, ty, tw, th));
			var res = new cv.Mat();
			cv.matchTemplate(search, tmpl, res, cv.TM_CCOEFF_NORMED);
			var mm = cv.minMaxLoc(res);
			search.delete(); tmpl.delete(); res.delete();
			return { dx: (rx + mm.maxLoc.x) / scale - r.x, dy: (ry + mm.maxLoc.y) / scale - r.y, score: mm.maxVal };
		}
		// coarse at half resolution
		var wSmall = new cv.Mat(), rSmall = new cv.Mat();
		cv.resize(warpedGreen, wSmall, new cv.Size(0, 0), 1 / half, 1 / half, cv.INTER_AREA);
		cv.resize(refMat, rSmall, new cv.Size(0, 0), 1 / half, 1 / half, cv.INTER_AREA);
		var coarse = ccorr(wSmall, rSmall, rect, margin, 1 / half);
		wSmall.delete(); rSmall.delete();
		// fine at full resolution around the coarse peak
		var shifted = { x: rect.x + Math.round(coarse.dx), y: rect.y + Math.round(coarse.dy), w: rect.w, h: rect.h };
		var fine = ccorrFine(warpedGreen, rect, shifted, 4);
		return { dx: shifted.x - rect.x + fine.dx, dy: shifted.y - rect.y + fine.dy, score: Math.max(coarse.score, fine.score) };
	}

	function ccorrFine(img, tmplRect, aroundRect, mg) {
		var rx = Math.max(0, aroundRect.x - mg), ry = Math.max(0, aroundRect.y - mg);
		var rw = Math.min(img.cols - rx, aroundRect.w + 2 * mg), rh = Math.min(img.rows - ry, aroundRect.h + 2 * mg);
		var search = img.roi(new cv.Rect(rx, ry, rw, rh));
		var tmpl = refMat.roi(new cv.Rect(tmplRect.x, tmplRect.y, tmplRect.w, tmplRect.h));
		var res = new cv.Mat();
		cv.matchTemplate(search, tmpl, res, cv.TM_CCOEFF_NORMED);
		var mm = cv.minMaxLoc(res);
		search.delete(); tmpl.delete(); res.delete();
		return { dx: rx + mm.maxLoc.x - aroundRect.x, dy: ry + mm.maxLoc.y - aroundRect.y, score: mm.maxVal };
	}

	/* ------------------------------------------------------------------ */
	/* Print type + mark channel                                           */
	/* ------------------------------------------------------------------ */

	function detectPrintType(warpedRGBA, sectionsPx) {
		// Compare (R - G) on printed pixels vs paper pixels, using the reference as the print mask.
		var sumPrint = 0, nPrint = 0, sumPaper = 0, nPaper = 0;
		var W = warpedRGBA.cols;
		var d = warpedRGBA.data, r = refMat.data;
		sectionsPx.forEach(function (rect) {
			for (var y = rect.y; y < rect.y + rect.h; y += 2) {
				for (var x = rect.x; x < rect.x + rect.w; x += 2) {
					var i = y * W + x;
					var rv = d[i * 4], gv = d[i * 4 + 1];
					var diff = rv - gv;
					if (r[i] < 110) { sumPrint += diff; nPrint++; }
					else if (r[i] > 200) { sumPaper += diff; nPaper++; }
				}
			}
		});
		var printDiff = nPrint ? sumPrint / nPrint : 0;
		var paperDiff = nPaper ? sumPaper / nPaper : 0;
		var sep = printDiff - paperDiff;
		// Bias toward "red": on a truly black-and-white sheet the red channel is just gray, so a
		// false "red" costs nothing, while a false "bw" on a red sheet re-admits the printing.
		return { type: sep > 10 ? "red" : "bw", separation: sep };
	}

	/* Returns {darkness: CV_8UC1 (bg - img), bg: CV_8UC1} */
	function markDarkness(warpedRGBA, printType) {
		var chans = new cv.MatVector();
		cv.split(warpedRGBA, chans);
		var mark;
		if (printType === "red") {
			mark = chans.get(0).clone();
		}
		else {
			mark = new cv.Mat();
			cv.cvtColor(warpedRGBA, mark, cv.COLOR_RGBA2GRAY);
		}
		chans.delete();
		// background: close at quarter resolution (removes marks + print), back up to full size
		var small = new cv.Mat(), closed = new cv.Mat(), bg = new cv.Mat();
		cv.resize(mark, small, new cv.Size(0, 0), 0.25, 0.25, cv.INTER_AREA);
		var kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(11, 11));
		cv.morphologyEx(small, closed, cv.MORPH_CLOSE, kernel);
		cv.GaussianBlur(closed, closed, new cv.Size(9, 9), 0);
		cv.resize(closed, bg, new cv.Size(mark.cols, mark.rows), 0, 0, cv.INTER_LINEAR);
		var dark = new cv.Mat();
		cv.subtract(bg, mark, dark);
		small.delete(); closed.delete(); kernel.delete(); mark.delete();
		return { darkness: dark, bg: bg };
	}

	/* Sorted normalized-darkness samples inside the ellipse (rx, ry) at (cx, cy), optionally
	   excluding an inner ellipse (rxIn, ryIn) to sample an annulus. */
	/* mask (if given) is {mat, dx, dy}: a print mask in reference coordinates whose printing
	   sits at (+dx,+dy) in the sampled image. */
	function ovalSamples(dark, bg, mask, cx, cy, rx, ry, rxIn, ryIn) {
		var W = dark.cols, Hh = dark.rows;
		var d = dark.data, b = bg.data, mk = mask ? mask.mat.data : null;
		var mdx = mask ? Math.round(mask.dx) : 0, mdy = mask ? Math.round(mask.dy) : 0;
		var vals = [], skipped = 0;
		var y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(Hh - 1, Math.ceil(cy + ry));
		var x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(W - 1, Math.ceil(cx + rx));
		for (var y = y0; y <= y1; y++) {
			var fy = (y - cy) / ry, fyIn = rxIn ? (y - cy) / ryIn : 0;
			for (var x = x0; x <= x1; x++) {
				var fx = (x - cx) / rx;
				if (fx * fx + fy * fy <= 1) {
					if (rxIn) {
						var fxIn = (x - cx) / rxIn;
						if (fxIn * fxIn + fyIn * fyIn <= 1) { continue; }
					}
					var i = y * W + x;
					if (mk) {
						var my = y - mdy, mx = x - mdx;
						if (my >= 0 && my < Hh && mx >= 0 && mx < W && mk[my * W + mx]) { skipped++; continue; }
					}
					vals.push(b[i] > 0 ? d[i] / b[i] : 0);
				}
			}
		}
		if (mk && vals.length < 0.3 * (vals.length + skipped)) {
			// mask ate nearly everything (misalignment?) - fall back to unmasked sampling
			return ovalSamples(dark, bg, null, cx, cy, rx, ry, rxIn, ryIn);
		}
		vals.sort(function (p, q) { return p - q; });
		return vals;
	}

	function percentile(sorted, f) {
		if (sorted.length === 0) { return 0; }
		return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * f))];
	}

	/*
	 * Fill value for one oval = (30th percentile of darkness inside the inner ellipse)
	 *                          - (30th percentile of darkness in a surrounding annulus).
	 * Why a low percentile and not the mean: an unfilled oval still contains its printed
	 * letter (black, on a photocopy), but more than half of its interior is bare paper, so a
	 * low percentile of an unfilled oval is ~paper while a filled oval - even a lightly
	 * filled one - has no bare paper at all.
	 * Why subtract the annulus: the shaded bands, shadows and copier gray all shift the
	 * local floor; the annulus (which mostly sees paper/band between ovals) measures it.
	 */
	function sampleOval(dark, bg, mask, cx, cy, ow, oh) {
		var inner = ovalSamples(dark, bg, mask, cx, cy, ow * 0.40, oh * 0.40);
		// No annulus subtraction: the row-relative step cancels band shading and shadows
		// for all ovals of a row alike, whereas an annulus around the last oval of a shaded
		// row reaches past the band into white paper and made that oval read darker
		// (measured: false marks of 0.10-0.15 on a real photocopy).
		// Two measures. "fill": 20th percentile - high only when the oval is dark throughout,
		// robust for evenly filled or photocopier-faded marks. "ink": mean - sees any stroke,
		// which is what catches the check marks and slashes students actually make. Each is
		// made row-relative and letter-bias-corrected in decide(); the larger wins.
		var sum = 0;
		for (var i = 0; i < inner.length; i++) { sum += inner[i]; }
		return { fill: percentile(inner, 0.20), ink: inner.length ? sum / inner.length : 0 };
	}

	/* Pixels where the blank sheet has printing (letters, rings), dilated 2px: excluded from
	   sampling so a black-and-white copy's letters can't masquerade as pencil. */
	function buildPrintMask() {
		var m = new cv.Mat();
		cv.threshold(refMat, m, 150, 255, cv.THRESH_BINARY_INV);
		var k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
		cv.dilate(m, m, k);
		k.delete();
		return m;
	}

	/* ------------------------------------------------------------------ */
	/* Decisions                                                           */
	/* ------------------------------------------------------------------ */

	function otsu(values) {
		var bins = 256, hist = new Array(bins).fill(0);
		var n = values.length;
		values.forEach(function (v) { hist[Math.min(bins - 1, Math.max(0, Math.round(v * (bins - 1))))]++; });
		var sum = 0;
		for (var i = 0; i < bins; i++) { sum += i * hist[i]; }
		var sumB = 0, wB = 0, best = 0, thr = 0;
		for (var t = 0; t < bins; t++) {
			wB += hist[t];
			if (wB === 0) { continue; }
			var wF = n - wB;
			if (wF === 0) { break; }
			sumB += t * hist[t];
			var mB = sumB / wB, mF = (sum - sumB) / wF;
			var between = wB * wF * (mB - mF) * (mB - mF);
			if (between > best) { best = between; thr = t; }
		}
		return thr / (bins - 1);
	}

	/*
	 * Row-relative fill: each oval minus the lower-median oval of its own row. The
	 * ovals of one question share their band shading, shadow, blur and copier tone,
	 * so this cancels all of them; a mark is whatever stands above its neighbours.
	 * (Lower median: for 4 ovals the 2nd lowest, for 5 the 3rd - so even two marks
	 * in a row are measured against an unmarked oval.)
	 */
	function rowRelative(values) {
		var sorted = values.slice().sort(function (a, b) { return a - b; });
		var base = sorted[Math.floor((values.length - 1) / 2)];
		return values.map(function (v) { return Math.max(0, v - base); });
	}

	/*
	 * Per-letter bias: on a black-and-white copy the heavier letters (B, G) darken their
	 * ovals a little even when unmarked. Since most ovals are unmarked, the median of the
	 * row-relative value at each letter position (per row parity: A-D rows vs F-J rows)
	 * is that letter's unmarked baseline; subtract it.
	 */
	function removeLetterBias(questions, field) {
		var groups = {};
		questions.forEach(function (q) {
			var key = (q.n % 2) + ":" + q[field].length;
			var g = groups[key] || (groups[key] = []);
			q[field].forEach(function (v, i) { (g[i] || (g[i] = [])).push(v); });
		});
		var bias = {};
		Object.keys(groups).forEach(function (key) {
			bias[key] = groups[key].map(function (arr) {
				arr = arr.slice().sort(function (a, b) { return a - b; });
				return arr.length ? arr[Math.floor(arr.length / 2)] : 0;
			});
		});
		questions.forEach(function (q) {
			var b = bias[(q.n % 2) + ":" + q[field].length];
			q[field] = q[field].map(function (v, i) { return Math.max(0, v - (b[i] || 0)); });
		});
	}

	function decide(questions) {
		var all = [];
		questions.forEach(function (q) {
			q.abs = q.values;
			q.values = rowRelative(q.values);
			q.ink = rowRelative(q.ink);
		});
		removeLetterBias(questions, "values");
		removeLetterBias(questions, "ink");
		questions.forEach(function (q) {
			q.values = q.values.map(function (v, i) { return Math.max(v, q.ink[i]); });
			q.values.forEach(function (v) { all.push(v); });
		});
		var T = otsu(all);
		var lo = all.filter(function (v) { return v <= T; }), hi = all.filter(function (v) { return v > T; });
		function mean(a) { return a.length ? a.reduce(function (x, y) { return x + y; }, 0) / a.length : 0; }
		var mu0 = mean(lo), mu1 = mean(hi);
		// Guard against a sheet with no/few marks: Otsu will split noise.
		if (hi.length < 3 || mu1 < 0.18 || T < 0.08 || mu1 - mu0 < 0.10) {
			T = Math.max(T, 0.30);
			mu0 = Math.min(mu0, 0.5 * T);
			mu1 = Math.max(mu1, 2 * T);
		}
		// Confidence bands adapt to this sheet's own two clusters (paper-ish vs marked):
		// a confident single answer sits well inside the marked cluster and its runner-up
		// well inside the unmarked cluster.
		var sureHigh = T + 0.25 * (mu1 - T);
		var sureLow = T - 0.35 * (T - mu0);
		questions.forEach(function (q) {
			var idx = q.values.map(function (v, i) { return i; }).sort(function (a, b) { return q.values[b] - q.values[a]; });
			var v1 = q.values[idx[0]], v2 = q.values[idx[1]];
			var above = q.values.filter(function (v) { return v > T; }).length;
			q.pick = null;
			var faint = Math.max(MIN_MARK, mu0 + 0.25 * (T - mu0));
			if (above === 0) {
				// A faint but unambiguous mark (photocopier-faded pencil, light pencil): the
				// oval stands well above its row-mates even though it is under the sheet-wide
				// threshold. Pre-select it and ask the student to confirm. Measured on a real
				// photocopy: blank rows peak at ~0.07, faded marks at 0.12-0.16, full at 0.5.
				if (v1 >= faint && v2 >= faint) {
					// two faint marks (e.g. two light ticks): the student must decide
					q.status = "multiple";
					if (v1 > 1.7 * v2) { q.pick = q.choices[idx[0]].l; }
				}
				else if (v1 >= faint && v1 >= 2.5 * v2) {
					q.pick = q.choices[idx[0]].l;
					q.status = "uncertain";
				}
				else {
					q.status = v1 > T - 0.15 * (T - mu0) ? "uncertain" : "blank";
				}
			}
			else if (above === 1) {
				q.pick = q.choices[idx[0]].l;
				if (v2 >= faint) {
					// a second, lighter mark (typically a half-erased earlier answer): the
					// student must decide, so treat it as a double mark
					q.status = "multiple";
				}
				else {
					q.status = (v1 >= sureHigh && v2 <= sureLow) ? "ok" : "uncertain";
				}
			}
			else {
				q.status = "multiple";
				if (v1 - v2 > 0.6 * (mu1 - mu0)) { q.pick = q.choices[idx[0]].l; }
			}
		});
		return { T: T, mu0: mu0, mu1: mu1 };
	}

	/* ------------------------------------------------------------------ */
	/* Main entry                                                          */
	/* ------------------------------------------------------------------ */

	/*
	 * opts: { file: File|Blob|HTMLCanvasElement, sections: {english:{count,options},...},
	 *         urls: {opencv, sheets} }
	 * resolves { warpedCanvas, printType, registration, threshold, sections:[{name,prefix,score,aligned,questions:[...]}] }
	 */
	function scan(opts) {
		var urls = opts.urls || {};
		var photoCanvas, rgba, warped, warpedGreen, dark = null, printMask = null;
		return loadOpenCV(urls.opencv)
			.then(function () { log("Loading sheet layouts..."); return loadSheets(urls.sheets); })
			.then(function () {
				log("Reading photo...");
				if (opts.file instanceof HTMLCanvasElement) { return opts.file; }
				return fileToCanvas(opts.file);
			})
			.then(function (c) {
				photoCanvas = c;
				rgba = cv.imread(c);
				// Try registration candidates until one fits the answer grid well.
				var reg = createRegistrar(rgba);
				var best = null, cand, tries = 0;
				while ((cand = reg.next()) && tries < 4) {
					tries++;
					selectSheet(cand.sheet);
					log("Aligning " + template.name + " (" + cand.method + ", " + cand.inliers + " matches)...");
					var w = warpTo(rgba, cand.H);
					cand.H.delete();
					var ch = new cv.MatVector();
					cv.split(w, ch);
					var g = ch.get(1).clone();
					ch.delete();
					var sx = buildSections(opts.sections);
					var quality = refineAll(sx, g);
					if (!best || quality > best.quality) {
						if (best) { best.warped.delete(); best.green.delete(); }
						best = { cand: cand, warped: w, green: g, secs: sx, quality: quality };
					}
					else {
						w.delete(); g.delete();
					}
					if (best.quality >= GOOD_FIT) { break; }
				}
				reg.close();
				rgba.delete(); rgba = null;
				if (!best) {
					throw new Error("Couldn't find an answer sheet in the photo. Lay the sheet flat, fill the frame, keep the camera square to the page, and try again.");
				}
				selectSheet(best.cand.sheet);
				var reg2 = best.cand;
				warped = best.warped;
				warpedGreen = best.green;
				var secs = best.secs;
				if (secs.length === 0) { throw new Error("No sections to scan for this test type."); }

				var pt = detectPrintType(warped, secs.map(function (s) { return s.rect; }));
				log("Reading marks (" + (pt.type === "red" ? "red-ink sheet" : "black-and-white sheet") + ")...");
				dark = markDarkness(warped, pt.type);

				var ow = template.oval.w * REF_SCALE, oh = template.oval.h * REF_SCALE;
				// printMask = buildPrintMask();  // measured: at 200 dpi the letters leave too few
				// paper pixels inside an oval for masking to help; percentile sampling does better.
				var allQ = [];
				secs.forEach(function (s) {
					s.questions = [];
					for (var i = 0; i < s.count; i++) {
						var tq = s.tmpl.questions[i];
						var q = { n: tq.n, name: s.prefix + "_" + tq.n, choices: [], values: [], ink: [] };
						var col = s.columns[Math.floor((tq.n - 1) / s.tmpl.rowsPerColumn)] || { dx: s.dx, dy: s.dy };
						tq.choices.forEach(function (c) {
							var cx = c.x * REF_SCALE + col.dx, cy = c.y * REF_SCALE + col.dy;
							q.choices.push({ l: c.l, x: cx, y: cy });
							var m = s.aligned ? sampleOval(dark.darkness, dark.bg, null, cx, cy, ow, oh) : { fill: 0, ink: 0 };
							q.values.push(m.fill);
							q.ink.push(m.ink);
						});
						s.questions.push(q);
						if (s.aligned) { allQ.push(q); }
					}
				});
				if (allQ.length === 0) {
					throw new Error("Couldn't recognize the answer grid in this photo. Make sure this is the right answer sheet, the whole page is in view, and it is flat and evenly lit, then try again.");
				}
				var stats = decide(allQ);
				secs.forEach(function (s) {
					if (!s.aligned) {
						s.questions.forEach(function (q) { q.status = "unreadable"; q.pick = null; });
					}
				});

				var out = document.createElement("canvas");
				out.width = warped.cols; out.height = warped.rows;
				cv.imshow(out, warped);

				var result = {
					warpedCanvas: out,
					photoCanvas: photoCanvas,
					printType: pt.type,
					printSeparation: pt.separation,
					registration: { method: reg2.method, inliers: reg2.inliers, fit: Math.round(best.quality * 100) / 100 },
					sheet: { id: template.id, name: template.name },
					threshold: stats.T,
					clusters: { unmarked: stats.mu0, marked: stats.mu1 },
					ovalPx: { w: template.oval.w * REF_SCALE, h: template.oval.h * REF_SCALE },
					sections: secs.map(function (s) {
						return { name: s.name, prefix: s.prefix, count: s.count, sheetCount: s.sheetCount, sheetOptions: s.sheetOptions, formOptions: s.formOptions, score: s.score, aligned: s.aligned, dx: s.dx, dy: s.dy, columns: s.columns, questions: s.questions };
					})
				};
				return result;
			})
			.finally(function () {
				if (rgba) { rgba.delete(); }
				if (warped) { warped.delete(); }
				if (warpedGreen) { warpedGreen.delete(); }
				if (dark) { dark.darkness.delete(); dark.bg.delete(); }
				if (printMask) { printMask.delete(); }
			});
	}

	global.ScanEngine = {
		scan: scan,
		preload: function (urls) {
			urls = urls || {};
			return loadOpenCV(urls.opencv).then(function () { return loadSheets(urls.sheets); });
		},
		onStatus: null,
		REF_SCALE: REF_SCALE
	};
})(window);
