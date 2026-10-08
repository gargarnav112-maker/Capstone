/// Data model for one pupillary light reflex (PLR) recording.
library;

/// One point on the kinematic curve; time is relative to optical stimulus
/// onset (negative = resting baseline).
class KinematicPoint {
  const KinematicPoint(this.tMs, this.diameterMm, [this.velocityMmS]);

  final double tMs;
  final double diameterMm;

  /// d(diameter)/dt from the RTS smoother; null for raw/live points.
  final double? velocityMmS;

  Map<String, Object?> toJson() => {
        't': double.parse(tMs.toStringAsFixed(1)),
        'd': double.parse(diameterMm.toStringAsFixed(3)),
        if (velocityMmS case final v?) 'v': double.parse(v.toStringAsFixed(3)),
      };
}

enum Eye {
  right('OD', 'Right eye'),
  left('OS', 'Left eye');

  const Eye(this.code, this.label);

  /// HL7 / ophthalmic laterality code (oculus dexter / sinister).
  final String code;
  final String label;
}

enum NpiCategory {
  brisk('Normal', 'Brisk reactivity'),
  sluggish('Abnormal', 'Sluggish reactivity'),
  nonReactive('Non-reactive', 'No measurable reflex');

  const NpiCategory(this.label, this.description);

  final String label;
  final String description;
}

class PlrMetrics {
  const PlrMetrics({
    required this.baselineDiameterMm,
    required this.minDiameterMm,
    required this.constrictionPercent,
    required this.latencyMs,
    required this.maxConstrictionVelocityMmS,
    required this.avgConstrictionVelocityMmS,
    required this.dilationVelocityMmS,
    required this.peakDilationVelocityMmS,
    required this.t75Ms,
    required this.npi,
    required this.npiCategory,
  });

  final double baselineDiameterMm;
  final double minDiameterMm;

  /// %CH = (baseline − min) / baseline × 100.
  final double constrictionPercent;

  /// Stimulus onset → start of constriction. Null for a non-reactive pupil.
  final double? latencyMs;

  /// MCV – peak inward speed (magnitude of the most negative dD/dt).
  final double maxConstrictionVelocityMmS;

  /// CV – mean constriction speed from onset to minimum diameter.
  final double avgConstrictionVelocityMmS;

  /// DV – mean re-dilation speed from the minimum to 75 % recovery (or to the
  /// end of the recording when 75 % recovery is not reached).
  final double dilationVelocityMmS;
  final double peakDilationVelocityMmS;

  /// Time from minimum diameter to 75 % recovery; null if not reached.
  final double? t75Ms;

  /// Synapse composite pupil index, 0.0–5.0 (see `npi.dart`).
  final double npi;
  final NpiCategory npiCategory;

  Map<String, Object?> toJson() => {
        'baselineDiameterMm': baselineDiameterMm,
        'minDiameterMm': minDiameterMm,
        'constrictionPercent': constrictionPercent,
        'latencyMs': latencyMs,
        'maxConstrictionVelocityMmS': maxConstrictionVelocityMmS,
        'avgConstrictionVelocityMmS': avgConstrictionVelocityMmS,
        'dilationVelocityMmS': dilationVelocityMmS,
        'peakDilationVelocityMmS': peakDilationVelocityMmS,
        't75Ms': t75Ms,
        'npi': npi,
        'npiCategory': npiCategory.name,
      };
}

class ScanQuality {
  const ScanQuality({
    required this.frames,
    required this.detectedFraction,
    required this.effectiveFps,
    required this.maxGapMs,
    required this.blinkCount,
    required this.stimulusFromOptics,
    required this.flashOpticalDurationMs,
    required this.warnings,
  });

  final int frames;
  final double detectedFraction;
  final double effectiveFps;
  final double maxGapMs;
  final int blinkCount;

  /// True if stimulus onset came from the measured luminance step, false if
  /// it fell back to the torch command timestamp.
  final bool stimulusFromOptics;
  final double? flashOpticalDurationMs;
  final List<String> warnings;

  bool get acceptable => warnings.isEmpty;

  Map<String, Object?> toJson() => {
        'frames': frames,
        'detectedFraction': detectedFraction,
        'effectiveFps': effectiveFps,
        'maxGapMs': maxGapMs,
        'blinkCount': blinkCount,
        'stimulusFromOptics': stimulusFromOptics,
        'flashOpticalDurationMs': flashOpticalDurationMs,
        'warnings': warnings,
      };
}

class ScanResult {
  const ScanResult({
    required this.scanId,
    required this.eye,
    required this.recordedAt,
    required this.metrics,
    required this.curve,
    required this.rawCurve,
    required this.quality,
    required this.sensor,
  });

  final String scanId;
  final Eye eye;
  final DateTime recordedAt;
  final PlrMetrics metrics;

  /// RTS-smoothed curve with velocities, t relative to stimulus onset.
  final List<KinematicPoint> curve;

  /// Per-frame measured diameters (only detected frames).
  final List<KinematicPoint> rawCurve;
  final ScanQuality quality;

  /// Frozen sensor parameters (ISO, exposure, focus, fps) for audit.
  final Map<String, Object> sensor;
}
