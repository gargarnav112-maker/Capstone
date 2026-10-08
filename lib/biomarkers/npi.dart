import 'dart:math' as math;

import 'plr_models.dart';

/// Normative reference for one PLR parameter.
class Norm {
  const Norm(this.mean, this.sd, {required this.higherIsWorse, this.weight = 1});

  final double mean;
  final double sd;
  final bool higherIsWorse;
  final double weight;

  /// One-sided z-score: only deviation in the pathological direction counts.
  double penalty(double value) {
    final z = (value - mean) / sd;
    return math.max(0, higherIsWorse ? z : -z);
  }
}

/// Synapse composite pupil index ("NPi-style"), 0.0–5.0.
///
/// IMPORTANT: the NeurOptics NPi® algorithm is proprietary and is NOT
/// reproduced here. This is a transparent surrogate with the same scale and
/// clinical cut-offs (≥ 3.0 normal, < 3.0 abnormal, 0 non-reactive): each
/// parameter is converted to a one-sided z-score against [norms], the weighted
/// mean penalty z̄ is computed, and index = 5 − [slope]·z̄ (clamped).
///
/// The default norms are literature-informed placeholders for healthy adults
/// under bright-flash stimulation (latency ≈ 200–240 ms, average constriction
/// velocity > 0.8 mm/s abnormal threshold). They MUST be re-derived from a
/// validation cohort recorded with this device against a reference
/// pupillometer before any clinical use.
class NpiCalculator {
  const NpiCalculator({
    this.norms = defaultNorms,
    this.slope = 1.25,
    this.nonReactiveAmplitudeMm = 0.15,
  });

  static const defaultNorms = <String, Norm>{
    'latencyMs': Norm(220, 40, higherIsWorse: true),
    // z = 2 at 0.8 mm/s: the conventional "abnormal CV" cut-off.
    'avgConstrictionVelocityMmS': Norm(1.8, 0.5, higherIsWorse: false, weight: 1.5),
    'maxConstrictionVelocityMmS': Norm(3.2, 0.8, higherIsWorse: false),
    'constrictionPercent': Norm(30, 7, higherIsWorse: false),
    'dilationVelocityMmS': Norm(0.7, 0.25, higherIsWorse: false, weight: 0.5),
  };

  final Map<String, Norm> norms;

  /// z̄ = 1.6 maps to 3.0 (the abnormal cut-off) with the default slope.
  final double slope;

  /// Constriction amplitude below which the pupil is called non-reactive.
  final double nonReactiveAmplitudeMm;

  (double, NpiCategory) compute({
    required double? latencyMs,
    required double avgConstrictionVelocityMmS,
    required double maxConstrictionVelocityMmS,
    required double constrictionPercent,
    required double dilationVelocityMmS,
    required double amplitudeMm,
  }) {
    if (latencyMs == null || amplitudeMm < nonReactiveAmplitudeMm) {
      return (0.0, NpiCategory.nonReactive);
    }
    final values = {
      'latencyMs': latencyMs,
      'avgConstrictionVelocityMmS': avgConstrictionVelocityMmS,
      'maxConstrictionVelocityMmS': maxConstrictionVelocityMmS,
      'constrictionPercent': constrictionPercent,
      'dilationVelocityMmS': dilationVelocityMmS,
    };
    var sum = 0.0, wsum = 0.0;
    for (final MapEntry(:key, value: norm) in norms.entries) {
      final v = values[key];
      if (v == null) continue;
      sum += norm.weight * norm.penalty(v);
      wsum += norm.weight;
    }
    final zbar = wsum == 0 ? 0.0 : sum / wsum;
    final npi = ((5 - slope * zbar).clamp(0.0, 5.0) * 10).round() / 10;
    return (npi, npi >= 3.0 ? NpiCategory.brisk : NpiCategory.sluggish);
  }
}
