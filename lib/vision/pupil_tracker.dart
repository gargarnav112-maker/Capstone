import 'dart:math' as math;
import 'dart:typed_data';

import 'ellipse_fit.dart';
import 'grayscale.dart';
import 'kalman.dart';
import 'pixel_scale.dart';
import 'pupil_segmenter.dart';

/// Per-frame output of the edge-AI pipeline.
class PupilObservation {
  const PupilObservation({
    required this.timestampUs,
    required this.frameIndex,
    required this.meanLuma,
    required this.roiWidth,
    required this.roiHeight,
    this.ellipse,
    this.confidence = 0,
    this.irisDiameterPx,
    this.stabilizedX,
    this.stabilizedY,
    this.diameterMm,
  });

  final int timestampUs;
  final int frameIndex;

  /// Mean ROI luminance – used to detect the true optical flash onset.
  final double meanLuma;
  final int roiWidth;
  final int roiHeight;

  /// Raw least-squares fit (ROI pixel coordinates), null if no pupil.
  final Ellipse? ellipse;

  /// 0..1 combined blob score × contour inlier ratio.
  final double confidence;
  final double? irisDiameterPx;

  /// Kalman-stabilised pupil centre (ROI px) – drives the reticle.
  final double? stabilizedX;
  final double? stabilizedY;

  /// Raw (unsmoothed) pupil diameter in mm for this frame. Temporal
  /// smoothing is applied later, offline, by the RTS smoother.
  final double? diameterMm;

  bool get detected => ellipse != null;

  Map<String, Object?> toMessage() => {
        'ts': timestampUs,
        'i': frameIndex,
        'l': meanLuma,
        'w': roiWidth,
        'h': roiHeight,
        if (ellipse case final e?) 'e': [e.cx, e.cy, e.a, e.b, e.theta],
        'c': confidence,
        'iris': irisDiameterPx,
        'sx': stabilizedX,
        'sy': stabilizedY,
        'd': diameterMm,
      };

  factory PupilObservation.fromMessage(Map<Object?, Object?> m) {
    final e = m['e'] as List<Object?>?;
    return PupilObservation(
      timestampUs: m['ts']! as int,
      frameIndex: m['i']! as int,
      meanLuma: m['l']! as double,
      roiWidth: m['w']! as int,
      roiHeight: m['h']! as int,
      ellipse: e == null
          ? null
          : Ellipse(
              cx: e[0]! as double,
              cy: e[1]! as double,
              a: e[2]! as double,
              b: e[3]! as double,
              theta: e[4]! as double,
            ),
      confidence: m['c']! as double,
      irisDiameterPx: m['iris'] as double?,
      stabilizedX: m['sx'] as double?,
      stabilizedY: m['sy'] as double?,
      diameterMm: m['d'] as double?,
    );
  }
}

/// Moves each contour point along the ray from the ellipse centre to where the
/// luminance crosses halfway between the local inside (pupil) and outside
/// (iris) levels, with linear interpolation between 0.25 px samples. Points
/// with weak contrast (eyelid, glint touching the edge) are dropped.
(Float64List, Float64List) refineSubpixel(
  GrayImage img,
  Ellipse e,
  Float64List xs,
  Float64List ys, {
  double reach = 3.0,
  double minContrast = 12,
}) {
  final outX = <double>[];
  final outY = <double>[];
  const step = 0.25;
  final n = (2 * reach / step).round() + 1;
  final profile = Float64List(n);
  for (var i = 0; i < xs.length; i++) {
    final dx = xs[i] - e.cx, dy = ys[i] - e.cy;
    final len = math.sqrt(dx * dx + dy * dy);
    if (len < 1) continue;
    final nx = dx / len, ny = dy / len;
    for (var k = 0; k < n; k++) {
      final s = -reach + k * step;
      profile[k] = img.sample(xs[i] + nx * s, ys[i] + ny * s);
    }
    final inside = profile[0], outside = profile[n - 1];
    if (outside - inside < minContrast) continue;
    final half = (inside + outside) / 2;
    for (var k = 1; k < n; k++) {
      if (profile[k - 1] < half && profile[k] >= half) {
        final f = (half - profile[k - 1]) / (profile[k] - profile[k - 1]);
        final s = -reach + (k - 1 + f) * step;
        outX.add(xs[i] + nx * s);
        outY.add(ys[i] + ny * s);
        break;
      }
    }
  }
  return (Float64List.fromList(outX), Float64List.fromList(outY));
}

/// Stateful per-frame pipeline:
///
///   luma ROI → adaptive threshold → connected components → hole-filled
///   contour → robust direct least-squares ellipse → Kalman-stabilised
///   centre (tremor) → iris-referenced mm conversion.
class PupilTracker {
  PupilTracker({
    SegmenterParams segmenter = const SegmenterParams(),
    this.scale = const IrisReferenceScale(),
    this.iris = const IrisEstimator(),
    this.minConfidence = 0.25,
    this.maxEccentricity = 0.8,
  }) : _segmenter = PupilSegmenter(segmenter);

  final PupilSegmenter _segmenter;
  final PixelScale scale;
  final IrisEstimator iris;
  final double minConfidence;

  /// Reject fits more elongated than this (partial blink, lid occlusion).
  final double maxEccentricity;

  final CenterKalman _center = CenterKalman();

  /// Slowly-varying iris size; its noise would otherwise be injected 1:1 into
  /// every mm measurement.
  final ConstantVelocityKalman _irisPx = ConstantVelocityKalman(q: 50, r: 9, gateSigma: 4);

  void reset() {
    _center.reset();
    _irisPx.reset();
  }

  PupilObservation process(GrayImage img, {required int timestampUs, required int frameIndex}) {
    final t = timestampUs / 1e6;
    final ii = IntegralImage(img);
    final meanLuma = ii.globalMean;

    double? priorX, priorY;
    if (_center.initialized) (priorX, priorY) = _center.predictAt(t);

    final blob = _segmenter.segment(img, integral: ii, priorX: priorX, priorY: priorY);
    Ellipse? ellipse;
    var confidence = 0.0;
    if (blob != null && blob.contourX.length >= 12) {
      var fit = EllipseFitter.fitRobust(blob.contourX, blob.contourY);
      // Second pass on sub-pixel edges: the binary contour is quantised to
      // half-pixels and its position depends on the threshold level; the
      // half-intensity crossing does not.
      if (fit != null) {
        final (rx, ry) = refineSubpixel(img, fit.$1, blob.contourX, blob.contourY);
        if (rx.length >= 12) fit = EllipseFitter.fitRobust(rx, ry) ?? fit;
      }
      if (fit != null) {
        final (e, inliers) = fit;
        final plausible = e.eccentricity <= maxEccentricity &&
            e.cx > 0 &&
            e.cy > 0 &&
            e.cx < img.width &&
            e.cy < img.height &&
            // Ellipse area should agree with blob area (π·a·b vs pixel count).
            (math.pi * e.a * e.b / blob.area - 1).abs() < 0.35;
        confidence = blob.score * inliers;
        if (plausible && confidence >= minConfidence) ellipse = e;
      }
    }

    _center.update(t, ellipse?.cx, ellipse?.cy);

    double? irisPx;
    double? diameterMm;
    if (ellipse != null) {
      final raw = iris.estimate(img, ellipse.cx, ellipse.cy, ellipse.a);
      _irisPx.update(t, raw);
      irisPx = _irisPx.initialized ? _irisPx.position : null;
      final mmPerPx = scale.mmPerPixel(irisDiameterPx: irisPx);
      if (mmPerPx != null) diameterMm = ellipse.majorDiameter * mmPerPx;
    } else {
      _irisPx.update(t, null);
    }

    return PupilObservation(
      timestampUs: timestampUs,
      frameIndex: frameIndex,
      meanLuma: meanLuma,
      roiWidth: img.width,
      roiHeight: img.height,
      ellipse: ellipse,
      confidence: confidence,
      irisDiameterPx: irisPx,
      stabilizedX: _center.initialized ? _center.x : null,
      stabilizedY: _center.initialized ? _center.y : null,
      diameterMm: diameterMm,
    );
  }
}
