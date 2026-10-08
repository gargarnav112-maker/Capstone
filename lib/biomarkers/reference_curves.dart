import 'dart:math' as math;

import 'plr_models.dart';

/// Parametric PLR waveform used for the chart's reference overlays and for
/// synthetic test fixtures.
///
///   D(t) = D0                                   t < L
///   D(t) = D0 − A · g(t − L) / g_max            t ≥ L
///   g(s) = (1 − exp(−s/τc))² · exp(−s/τd)
///
/// The squared rise gives the physiological S-shaped onset (zero velocity at
/// the end of latency), τc sets the constriction speed and τd the slower
/// re-dilation. Normalised so the trough depth is exactly A.
class PlrModel {
  const PlrModel({
    required this.baselineMm,
    required this.latencyMs,
    required this.amplitudeMm,
    required this.tauConstrictMs,
    required this.tauDilateMs,
  });

  /// Illustrative healthy adult response to a bright 100 ms flash.
  static const healthy = PlrModel(
    baselineMm: 5.0,
    latencyMs: 220,
    amplitudeMm: 1.6,
    tauConstrictMs: 280,
    tauDilateMs: 1500,
  );

  /// Illustrative abnormal (sluggish) response: delayed, shallow, slow.
  static const abnormal = PlrModel(
    baselineMm: 4.6,
    latencyMs: 380,
    amplitudeMm: 0.5,
    tauConstrictMs: 450,
    tauDilateMs: 2500,
  );

  final double baselineMm;
  final double latencyMs;
  final double amplitudeMm;
  final double tauConstrictMs;
  final double tauDilateMs;

  /// dg/ds = 0  ⇔  exp(−s/τc) = τc / (2τd + τc).
  double get _peakS => tauConstrictMs * math.log((2 * tauDilateMs + tauConstrictMs) / tauConstrictMs);

  double _g(double s) {
    final rise = 1 - math.exp(-s / tauConstrictMs);
    return rise * rise * math.exp(-s / tauDilateMs);
  }

  double diameterAt(double tMs) {
    if (tMs < latencyMs) return baselineMm;
    return baselineMm - amplitudeMm * _g(tMs - latencyMs) / _g(_peakS);
  }

  /// Analytic dD/dt in mm/s.
  double velocityAt(double tMs) {
    if (tMs < latencyMs) return 0;
    final s = tMs - latencyMs;
    final u = math.exp(-s / tauConstrictMs);
    final decay = math.exp(-s / tauDilateMs);
    final dg = 2 * (1 - u) * (u / tauConstrictMs) * decay - (1 - u) * (1 - u) * decay / tauDilateMs;
    return -amplitudeMm * dg / _g(_peakS) * 1000;
  }

  /// Time of minimum diameter (ms after stimulus).
  double get troughMs => latencyMs + _peakS;

  List<KinematicPoint> sample({double fromMs = -500, double toMs = 3000, double stepMs = 1000 / 120}) {
    final n = ((toMs - fromMs) / stepMs).floor() + 1;
    return [
      for (var i = 0; i < n; i++)
        KinematicPoint(fromMs + i * stepMs, diameterAt(fromMs + i * stepMs)),
    ];
  }
}
