import 'dart:math' as math;
import 'dart:typed_data';

import 'grayscale.dart';

/// Tunables for pipeline step 3 (adaptive thresholding) and blob selection.
class SegmenterParams {
  const SegmenterParams({
    this.windowFraction = 0.5,
    this.localContrast = 0.15,
    this.maxPupilFraction = 0.15,
    this.minAreaFraction = 0.002,
    this.maxAreaFraction = 0.35,
  });

  /// Local-mean window side as a fraction of the ROI side. It must be wider
  /// than the pupil, otherwise the pupil's own interior drags the local mean
  /// down and its centre is not classified as foreground.
  final double windowFraction;

  /// A pixel is "dark" when it is this fraction below the local window mean
  /// (Bradley–Roth adaptive threshold).
  final double localContrast;

  /// The global darkness cap comes from Otsu's threshold (dark tissue vs.
  /// sclera/skin). If that dark class covers more than this fraction of the
  /// ROI it must contain the iris too, so Otsu is re-applied inside the dark
  /// class to split pupil from iris (works for dark-brown irides as long as
  /// the two are bimodal).
  final double maxPupilFraction;

  final double minAreaFraction;
  final double maxAreaFraction;
}

/// A dark connected component that looks like a pupil, with its hole-filled
/// outer contour sampled at pixel-edge midpoints (sub-pixel unbiased).
class PupilBlob {
  PupilBlob({
    required this.area,
    required this.centroidX,
    required this.centroidY,
    required this.score,
    required this.meanLuma,
    required this.contourX,
    required this.contourY,
  });

  final int area;
  final double centroidX;
  final double centroidY;

  /// 0..1 pupil-likeness (fill ratio × aspect × darkness × proximity).
  final double score;
  final double meanLuma;
  final Float64List contourX;
  final Float64List contourY;
}

/// Summary-area table with one row/column of zero padding.
class IntegralImage {
  IntegralImage(GrayImage img)
      : width = img.width,
        height = img.height,
        _sum = Int32List((img.width + 1) * (img.height + 1)) {
    final w1 = width + 1;
    for (var y = 0; y < height; y++) {
      var row = 0;
      final src = y * width;
      final dst = (y + 1) * w1;
      for (var x = 0; x < width; x++) {
        row += img.pixels[src + x];
        _sum[dst + x + 1] = _sum[dst - w1 + x + 1] + row;
      }
    }
  }

  final int width;
  final int height;
  final Int32List _sum;

  /// Mean over the inclusive rectangle, clipped to the image.
  double mean(int x0, int y0, int x1, int y1) {
    x0 = math.max(0, x0);
    y0 = math.max(0, y0);
    x1 = math.min(width - 1, x1);
    y1 = math.min(height - 1, y1);
    final w1 = width + 1;
    final s = _sum[(y1 + 1) * w1 + x1 + 1] -
        _sum[y0 * w1 + x1 + 1] -
        _sum[(y1 + 1) * w1 + x0] +
        _sum[y0 * w1 + x0];
    return s / ((x1 - x0 + 1) * (y1 - y0 + 1));
  }

  double get globalMean => mean(0, 0, width - 1, height - 1);
}

class PupilSegmenter {
  PupilSegmenter([this.params = const SegmenterParams()]);

  final SegmenterParams params;

  /// Adaptive threshold → 8-connected components → best pupil-like blob.
  ///
  /// [priorX]/[priorY] (e.g. the Kalman-predicted centre) bias selection
  /// toward temporal continuity; without a prior the ROI centre is used,
  /// matching where the targeting reticle is drawn.
  PupilBlob? segment(
    GrayImage img, {
    IntegralImage? integral,
    double? priorX,
    double? priorY,
  }) {
    final w = img.width;
    final h = img.height;
    final n = w * h;
    final ii = integral ?? IntegralImage(img);
    final mask = threshold(img, ii);

    // ---- Two-pass 8-connected labelling with union–find.
    final labels = Int32List(n);
    final parent = <int>[0];
    int find(int a) {
      while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
      }
      return a;
    }

    void union(int a, int b) {
      final ra = find(a);
      final rb = find(b);
      if (ra != rb) parent[math.max(ra, rb)] = math.min(ra, rb);
    }

    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final i = y * w + x;
        if (mask[i] == 0) continue;
        var l = 0;
        void merge(int v) {
          if (v == 0) return;
          if (l == 0) {
            l = v;
          } else if (v != l) {
            union(l, v);
          }
        }

        // Previously-visited neighbours: W, NW, N, NE.
        if (x > 0) merge(labels[i - 1]);
        if (y > 0) {
          if (x > 0) merge(labels[i - w - 1]);
          merge(labels[i - w]);
          if (x < w - 1) merge(labels[i - w + 1]);
        }
        if (l == 0) {
          l = parent.length;
          parent.add(l);
        }
        labels[i] = l;
      }
    }

    // ---- Per-component statistics on resolved roots.
    final count = parent.length;
    final area = Int32List(count);
    final sumX = Float64List(count);
    final sumY = Float64List(count);
    final sumLuma = Float64List(count);
    final minX = Int32List(count)..fillRange(0, count, w);
    final minY = Int32List(count)..fillRange(0, count, h);
    final maxX = Int32List(count)..fillRange(0, count, -1);
    final maxY = Int32List(count)..fillRange(0, count, -1);
    for (var i = 0; i < n; i++) {
      if (labels[i] == 0) continue;
      final r = find(labels[i]);
      labels[i] = r;
      final x = i % w;
      final y = i ~/ w;
      area[r]++;
      sumX[r] += x;
      sumY[r] += y;
      sumLuma[r] += img.pixels[i];
      if (x < minX[r]) minX[r] = x;
      if (y < minY[r]) minY[r] = y;
      if (x > maxX[r]) maxX[r] = x;
      if (y > maxY[r]) maxY[r] = y;
    }

    // ---- Score candidates.
    final minArea = params.minAreaFraction * n;
    final maxArea = params.maxAreaFraction * n;
    final globalMean = ii.globalMean;
    final px = priorX ?? w / 2;
    final py = priorY ?? h / 2;
    final diag = math.sqrt(w * w + h * h.toDouble());
    var best = -1;
    var bestScore = 0.0;
    for (var r = 1; r < count; r++) {
      final a = area[r];
      if (a < minArea || a > maxArea) continue;
      final bw = maxX[r] - minX[r] + 1;
      final bh = maxY[r] - minY[r] + 1;
      // Blobs clipped by the ROI border cannot be fitted reliably.
      if (minX[r] == 0 || minY[r] == 0 || maxX[r] == w - 1 || maxY[r] == h - 1) {
        continue;
      }
      final aspect = math.min(bw, bh) / math.max(bw, bh);
      if (aspect < 0.45) continue; // eyelashes, lid creases
      // An ellipse fills π/4 of its bounding box.
      final fill = (a / (bw * bh)) / (math.pi / 4);
      final fillScore = math.max(0.0, 1 - (1 - fill).abs() * 1.5);
      final darkness = 1 - (sumLuma[r] / a) / math.max(1.0, globalMean);
      final cx = sumX[r] / a;
      final cy = sumY[r] / a;
      final dist = math.sqrt((cx - px) * (cx - px) + (cy - py) * (cy - py));
      final proximity = math.exp(-math.pow(dist / (0.25 * diag), 2));
      final score = fillScore * aspect * darkness.clamp(0.0, 1.0) * proximity;
      if (score > bestScore) {
        bestScore = score;
        best = r;
      }
    }
    if (best < 0) return null;

    final contour = _outerContour(labels, w, best, minX[best], minY[best], maxX[best], maxY[best]);
    return PupilBlob(
      area: area[best],
      centroidX: sumX[best] / area[best],
      centroidY: sumY[best] / area[best],
      score: bestScore.clamp(0.0, 1.0),
      meanLuma: sumLuma[best] / area[best],
      contourX: contour.$1,
      contourY: contour.$2,
    );
  }

  /// Binary mask (1 = candidate pupil pixel).
  Uint8List threshold(GrayImage img, IntegralImage ii) {
    final w = img.width;
    final h = img.height;
    final half = math.max(7, (math.min(w, h) * params.windowFraction) ~/ 2);
    final darkCap = _darkCap(img.pixels, w * h);
    final k = 1 - params.localContrast;
    final mask = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        // Raw pixel, not a blurred one: blurring would dilate the blob and
        // bias the diameter outward. Noise is handled by sub-pixel refinement
        // and the robust fit downstream.
        final p = img.pixels[y * w + x];
        if (p > darkCap) continue;
        final mw = ii.mean(x - half, y - half, x + half, y + half);
        if (p < mw * k) mask[y * w + x] = 1;
      }
    }
    return mask;
  }

  int _darkCap(Uint8List px, int n) {
    final hist = Int32List(256);
    for (var i = 0; i < n; i++) {
      hist[px[i]]++;
    }
    final t1 = _otsu(hist, 255);
    var dark = 0;
    for (var v = 0; v <= t1; v++) {
      dark += hist[v];
    }
    if (dark / n <= params.maxPupilFraction) return t1;
    return _otsu(hist, t1);
  }

  /// Otsu's threshold over histogram bins [0, hi]; returns the last bin of
  /// the dark class.
  static int _otsu(Int32List hist, int hi) {
    var total = 0;
    var sum = 0.0;
    for (var v = 0; v <= hi; v++) {
      total += hist[v];
      sum += v * hist[v];
    }
    if (total == 0) return hi;
    var wB = 0, sumB = 0.0, best = 0.0, thr = 0;
    for (var v = 0; v < hi; v++) {
      wB += hist[v];
      if (wB == 0) continue;
      final wF = total - wB;
      if (wF == 0) break;
      sumB += v * hist[v];
      final mB = sumB / wB, mF = (sum - sumB) / wF;
      final between = wB.toDouble() * wF * (mB - mF) * (mB - mF);
      if (between > best) {
        best = between;
        thr = v;
      }
    }
    return thr;
  }

  /// Fills interior holes (e.g. the corneal reflection of the torch) by
  /// flood-filling the exterior inside the padded bounding box, then emits a
  /// point at the midpoint of every edge between the filled blob and the
  /// exterior.
  static (Float64List, Float64List) _outerContour(
    Int32List labels,
    int w,
    int label,
    int x0,
    int y0,
    int x1,
    int y1,
  ) {
    final bw = x1 - x0 + 3;
    final bh = y1 - y0 + 3;
    // 0 = unvisited, 1 = blob, 2 = exterior.
    final grid = Uint8List(bw * bh);
    for (var y = y0; y <= y1; y++) {
      for (var x = x0; x <= x1; x++) {
        if (labels[y * w + x] == label) grid[(y - y0 + 1) * bw + (x - x0 + 1)] = 1;
      }
    }
    final stack = <int>[0];
    grid[0] = 2;
    while (stack.isNotEmpty) {
      final i = stack.removeLast();
      final x = i % bw;
      final y = i ~/ bw;
      void visit(int nx, int ny) {
        if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) return;
        final j = ny * bw + nx;
        if (grid[j] != 0) return;
        grid[j] = 2;
        stack.add(j);
      }

      visit(x + 1, y);
      visit(x - 1, y);
      visit(x, y + 1);
      visit(x, y - 1);
    }

    final xs = <double>[];
    final ys = <double>[];
    const dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (var y = 1; y < bh - 1; y++) {
      for (var x = 1; x < bw - 1; x++) {
        if (grid[y * bw + x] == 2) continue; // blob or filled hole
        for (final (dx, dy) in dirs) {
          if (grid[(y + dy) * bw + x + dx] == 2) {
            xs.add(x + x0 - 1 + 0.5 * dx);
            ys.add(y + y0 - 1 + 0.5 * dy);
          }
        }
      }
    }
    return (Float64List.fromList(xs), Float64List.fromList(ys));
  }
}
