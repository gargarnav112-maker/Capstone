import 'dart:math' as math;
import 'dart:typed_data';

import 'grayscale.dart';

/// Converts pixel lengths to millimetres.
///
/// A handheld phone's working distance changes continuously with tremor, so a
/// fixed px→mm factor would convert distance changes into fake pupil size
/// changes. The default strategy therefore measures the iris in every frame
/// and uses the horizontal visible iris diameter (HVID, "white-to-white") as
/// an internal ruler: mm = px × HVID_mm / iris_px. The adult HVID is very
/// stable (≈11.7 ± 0.5 mm); when a patient's own HVID is known (biometry),
/// pass it in for a per-patient calibration.
abstract class PixelScale {
  /// Millimetres per pixel for this frame, or null if it cannot be determined.
  double? mmPerPixel({double? irisDiameterPx});
}

class IrisReferenceScale implements PixelScale {
  const IrisReferenceScale({this.hvidMm = 11.7});

  final double hvidMm;

  @override
  double? mmPerPixel({double? irisDiameterPx}) =>
      (irisDiameterPx == null || irisDiameterPx <= 0) ? null : hvidMm / irisDiameterPx;
}

/// Fixed factor, e.g. from a printed calibration target at a fixed-distance
/// eyecup, or from lens intrinsics × known working distance.
class FixedScale implements PixelScale {
  const FixedScale(this.mmPerPx);

  final double mmPerPx;

  @override
  double? mmPerPixel({double? irisDiameterPx}) => mmPerPx;
}

/// Estimates the limbus (iris/sclera boundary) diameter by casting rays from
/// the pupil centre into the nasal and temporal sectors (±[sectorDeg] around
/// horizontal – the vertical meridian is unreliable because of the eyelids)
/// and taking the strongest dark→bright luminance step along each ray.
class IrisEstimator {
  const IrisEstimator({
    this.raysPerSide = 9,
    this.sectorDeg = 30,
    this.minRatio = 1.6,
    this.maxRatio = 5.0,
  });

  final int raysPerSide;
  final double sectorDeg;

  /// Iris/pupil radius ratio search range.
  final double minRatio;
  final double maxRatio;

  /// Returns the iris diameter in pixels, or null when fewer than a third of
  /// the rays find a consistent edge (iris not fully in the ROI, lid, glare).
  double? estimate(GrayImage img, double cx, double cy, double pupilRadius) {
    if (pupilRadius <= 1) return null;
    final rMin = pupilRadius * minRatio;
    final rMax = pupilRadius * maxRatio;
    final radii = <double>[];
    final halfSector = sectorDeg * math.pi / 180;
    for (final base in const [0.0, math.pi]) {
      for (var i = 0; i < raysPerSide; i++) {
        final ang = base - halfSector + 2 * halfSector * i / math.max(1, raysPerSide - 1);
        final r = _edgeAlongRay(img, cx, cy, math.cos(ang), math.sin(ang), rMin, rMax);
        if (r != null) radii.add(r);
      }
    }
    if (radii.length < (2 * raysPerSide) / 3) return null;
    radii.sort();
    // The median over all rays is robust to eyelash / glint / lid rays.
    return 2 * radii[radii.length ~/ 2];
  }

  double? _edgeAlongRay(
    GrayImage img,
    double cx,
    double cy,
    double dx,
    double dy,
    double rMin,
    double rMax,
  ) {
    // Stop before leaving the ROI.
    var limit = rMax;
    if (dx > 1e-9) limit = math.min(limit, (img.width - 2 - cx) / dx);
    if (dx < -1e-9) limit = math.min(limit, (1 - cx) / dx);
    if (dy > 1e-9) limit = math.min(limit, (img.height - 2 - cy) / dy);
    if (dy < -1e-9) limit = math.min(limit, (1 - cy) / dy);
    if (limit <= rMin + 4) return null;

    final steps = (limit - rMin).ceil();
    final profile = Float64List(steps);
    for (var s = 0; s < steps; s++) {
      final r = rMin + s;
      // 3-tap perpendicular average suppresses iris texture.
      final px = cx + dx * r, py = cy + dy * r;
      profile[s] = (img.sample(px, py) +
              img.sample(px - dy, py + dx) +
              img.sample(px + dy, py - dx)) /
          3;
    }
    // Smoothed derivative (central difference over ±2 px).
    var best = 0.0;
    var bestIdx = -1;
    for (var s = 2; s < steps - 2; s++) {
      final g = (profile[s + 1] + profile[s + 2]) - (profile[s - 1] + profile[s - 2]);
      if (g > best) {
        best = g;
        bestIdx = s;
      }
    }
    // Require a meaningful step (≥ 8 grey levels on the 2-px average).
    if (bestIdx < 0 || best < 16) return null;
    return rMin + bestIdx;
  }
}
