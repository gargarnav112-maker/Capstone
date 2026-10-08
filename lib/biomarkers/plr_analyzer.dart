import 'dart:math' as math;

import '../vision/kalman.dart';
import 'npi.dart';
import 'plr_models.dart';

/// Minimal per-frame record the analyzer needs (decoupled from the vision
/// types so recorded sessions can be re-analysed offline).
class FrameSample {
  const FrameSample({required this.timestampUs, required this.meanLuma, this.diameterMm});

  final int timestampUs;
  final double meanLuma;
  final double? diameterMm;
}

/// Optical stimulus timing recovered from the frames themselves.
class StimulusTiming {
  const StimulusTiming({required this.onsetUs, required this.fromOptics, this.opticalDurationMs});

  final int onsetUs;
  final bool fromOptics;
  final double? opticalDurationMs;
}

class PlrAnalyzerConfig {
  const PlrAnalyzerConfig({
    this.kalmanQ = 10,
    this.kalmanR = 0.0025,
    this.baselineWindowMs = 500,
    this.constrictionSearchMs = 1500,
    this.onsetVelocityFraction = 0.15,
    this.minDetectedFraction = 0.7,
    this.minFps = 55,
    this.maxGapMs = 50,
    this.npi = const NpiCalculator(),
  });

  /// Diameter Kalman process noise ((mm/s²)²·s) and measurement noise (mm²).
  /// q = 10 was chosen by sweeping synthetic healthy/sluggish reflexes at 60
  /// and 120 fps with 0.03 mm noise: it keeps MCV within ~2 % while holding
  /// latency error to about one frame for brisk responses.
  final double kalmanQ;
  final double kalmanR;
  final double baselineWindowMs;
  final double constrictionSearchMs;

  /// Latency onset = last time before peak constriction velocity at which the
  /// constriction speed was still below this fraction of MCV.
  final double onsetVelocityFraction;
  final double minDetectedFraction;
  final double minFps;
  final double maxGapMs;
  final NpiCalculator npi;
}

class PlrAnalysisException implements Exception {
  PlrAnalysisException(this.message);

  final String message;

  @override
  String toString() => message;
}

class PlrAnalysis {
  const PlrAnalysis({
    required this.metrics,
    required this.curve,
    required this.rawCurve,
    required this.quality,
  });

  final PlrMetrics metrics;
  final List<KinematicPoint> curve;
  final List<KinematicPoint> rawCurve;
  final ScanQuality quality;
}

/// Clinical biomarker extraction from a recorded PLR.
class PlrAnalyzer {
  const PlrAnalyzer([this.config = const PlrAnalyzerConfig()]);

  final PlrAnalyzerConfig config;

  /// The torch takes several ms to reach full output and the commanded time
  /// is not when photons reach the retina, so the stimulus onset is measured
  /// from the ROI luminance step (the sclera/iris brighten under a locked
  /// exposure). Sub-frame precision: the onset is placed inside the first lit
  /// frame's interval in proportion to how lit that frame is.
  static StimulusTiming detectStimulus(
    List<FrameSample> frames, {
    required int commandOnUs,
    int searchWindowUs = 250000,
  }) {
    final pre = frames.where((f) => f.timestampUs < commandOnUs).toList();
    if (pre.length < 3) return StimulusTiming(onsetUs: commandOnUs, fromOptics: false);
    final mu = pre.map((f) => f.meanLuma).reduce((a, b) => a + b) / pre.length;
    final variance =
        pre.map((f) => (f.meanLuma - mu) * (f.meanLuma - mu)).reduce((a, b) => a + b) / pre.length;
    final threshold = mu + math.max(6 * math.sqrt(variance), math.max(4.0, 0.08 * mu));

    final window = frames
        .where((f) =>
            f.timestampUs >= commandOnUs - 20000 && f.timestampUs <= commandOnUs + searchWindowUs)
        .toList();
    final k = window.indexWhere((f) => f.meanLuma > threshold);
    if (k < 0) return StimulusTiming(onsetUs: commandOnUs, fromOptics: false);

    final plateau = window.skip(k).take(6).map((f) => f.meanLuma).reduce(math.max);
    final first = window[k];
    final prevIdx = frames.indexOf(first) - 1;
    var onset = first.timestampUs;
    if (prevIdx >= 0 && plateau > mu) {
      final dt = first.timestampUs - frames[prevIdx].timestampUs;
      final frac = ((first.meanLuma - mu) / (plateau - mu)).clamp(0.0, 1.0);
      onset = first.timestampUs - (dt * frac).round();
    }

    double? durationMs;
    final half = mu + 0.5 * (plateau - mu);
    final firstIdx = frames.indexOf(first);
    for (var i = firstIdx + 1; i < frames.length; i++) {
      if (frames[i].meanLuma < half) {
        durationMs = (frames[i].timestampUs - onset) / 1000;
        break;
      }
    }
    return StimulusTiming(onsetUs: onset, fromOptics: true, opticalDurationMs: durationMs);
  }

  PlrAnalysis analyze(List<FrameSample> frames, StimulusTiming stimulus) {
    if (frames.length < 20) throw PlrAnalysisException('Recording too short');
    final c = config;
    final warnings = <String>[];

    // ---- Quality: sampling.
    final dts = <double>[
      for (var i = 1; i < frames.length; i++)
        (frames[i].timestampUs - frames[i - 1].timestampUs) / 1000,
    ]..sort();
    final medianDt = dts[dts.length ~/ 2];
    final fps = medianDt > 0 ? 1000 / medianDt : 0.0;
    final maxGap = dts.last;
    if (fps < c.minFps) warnings.add('Frame rate ${fps.toStringAsFixed(0)} fps below ${c.minFps} fps');
    if (maxGap > c.maxGapMs) warnings.add('Frame gap of ${maxGap.toStringAsFixed(0)} ms');

    // ---- Relative time, Hampel outlier rejection on raw diameters.
    final t = [for (final f in frames) (f.timestampUs - stimulus.onsetUs) / 1000.0];
    final d = [for (final f in frames) f.diameterMm];
    _hampel(d, halfWindow: 4, k: 3.5);

    final detected = d.where((v) => v != null).length;
    final detectedFraction = detected / d.length;
    if (detectedFraction < c.minDetectedFraction) {
      warnings.add('Pupil tracked in only ${(detectedFraction * 100).round()} % of frames');
    }
    // A tracking gap of ≥ 40 ms between detections is a blink / lid occlusion.
    var blinks = 0;
    int? lastSeen;
    for (var i = 0; i < d.length; i++) {
      if (d[i] == null) continue;
      if (lastSeen != null && i - lastSeen > 1 && t[i] - t[lastSeen] >= 40) blinks++;
      lastSeen = i;
    }

    // ---- Zero-lag temporal smoothing: forward Kalman + RTS.
    final kf = ConstantVelocityKalman(q: c.kalmanQ, r: c.kalmanR, gateSigma: 5, record: true);
    for (var i = 0; i < t.length; i++) {
      kf.update(t[i] / 1000, d[i]);
    }
    final smoothed = kf.smooth();
    if (smoothed.length < 10) throw PlrAnalysisException('Pupil not tracked');
    // The filter only records steps from the first detection onward.
    final offset = t.length - smoothed.length;
    final curve = [
      for (var i = 0; i < smoothed.length; i++)
        KinematicPoint(t[i + offset], smoothed[i].x, smoothed[i].v),
    ];
    final rawCurve = [
      for (var i = 0; i < t.length; i++)
        if (d[i] case final v?) KinematicPoint(t[i], v),
    ];

    // ---- Baseline.
    final base = curve.where((p) => p.tMs < 0 && p.tMs >= -c.baselineWindowMs).toList();
    if (base.length < 3) throw PlrAnalysisException('No resting baseline before the flash');
    final baseline = base.map((p) => p.diameterMm).reduce((a, b) => a + b) / base.length;

    // ---- Constriction phase.
    final post = curve.where((p) => p.tMs >= 0 && p.tMs <= c.constrictionSearchMs).toList();
    if (post.length < 10) throw PlrAnalysisException('Response window not captured');
    var minIdx = 0;
    for (var i = 1; i < post.length; i++) {
      if (post[i].diameterMm < post[minIdx].diameterMm) minIdx = i;
    }
    final dMin = post[minIdx].diameterMm;
    final tMin = post[minIdx].tMs;
    final amplitude = math.max(0.0, baseline - dMin);

    var peakIdx = 0;
    for (var i = 1; i <= minIdx; i++) {
      if (post[i].velocityMmS! < post[peakIdx].velocityMmS!) peakIdx = i;
    }
    final mcv = math.max(0.0, -post[peakIdx].velocityMmS!);

    double? latency;
    final reactive = amplitude >= c.npi.nonReactiveAmplitudeMm && mcv > 0.2;
    if (reactive) {
      final thr = -c.onsetVelocityFraction * mcv;
      var k = peakIdx;
      while (k > 0 && post[k].velocityMmS! < thr) {
        k--;
      }
      if (k == peakIdx) {
        latency = post[k].tMs;
      } else {
        // Linear interpolation of the threshold crossing between k and k+1.
        final v0 = post[k].velocityMmS!, v1 = post[k + 1].velocityMmS!;
        final f = v1 == v0 ? 0.0 : ((thr - v0) / (v1 - v0)).clamp(0.0, 1.0);
        latency = post[k].tMs + f * (post[k + 1].tMs - post[k].tMs);
      }
    }
    final avgCv = (latency != null && tMin > latency) ? amplitude / ((tMin - latency) / 1000) : 0.0;

    // ---- Re-dilation phase.
    final recovery = curve.where((p) => p.tMs > tMin).toList();
    var peakDv = 0.0;
    double? t75;
    var dv = 0.0;
    if (recovery.length >= 3 && amplitude > 0) {
      for (final p in recovery) {
        peakDv = math.max(peakDv, p.velocityMmS!);
      }
      final target = dMin + 0.75 * amplitude;
      final hit = recovery.indexWhere((p) => p.diameterMm >= target);
      final end = hit >= 0 ? recovery[hit] : recovery.last;
      if (hit >= 0) t75 = end.tMs - tMin;
      final span = (end.tMs - tMin) / 1000;
      if (span > 0) dv = math.max(0.0, (end.diameterMm - dMin) / span);
    }

    final pct = baseline > 0 ? amplitude / baseline * 100 : 0.0;
    final (npi, category) = c.npi.compute(
      latencyMs: latency,
      avgConstrictionVelocityMmS: avgCv,
      maxConstrictionVelocityMmS: mcv,
      constrictionPercent: pct,
      dilationVelocityMmS: dv,
      amplitudeMm: amplitude,
    );

    if (!stimulus.fromOptics) {
      warnings.add('Flash onset not visible in frames; using torch command time');
    }
    if (stimulus.opticalDurationMs case final dur? when (dur - 100).abs() > 40) {
      warnings.add('Measured flash width ${dur.toStringAsFixed(0)} ms (expected 100 ms)');
    }

    return PlrAnalysis(
      metrics: PlrMetrics(
        baselineDiameterMm: baseline,
        minDiameterMm: dMin,
        constrictionPercent: pct,
        latencyMs: latency,
        maxConstrictionVelocityMmS: mcv,
        avgConstrictionVelocityMmS: avgCv,
        dilationVelocityMmS: dv,
        peakDilationVelocityMmS: peakDv,
        t75Ms: t75,
        npi: npi,
        npiCategory: category,
      ),
      curve: curve,
      rawCurve: rawCurve,
      quality: ScanQuality(
        frames: frames.length,
        detectedFraction: detectedFraction,
        effectiveFps: fps,
        maxGapMs: maxGap,
        blinkCount: blinks,
        stimulusFromOptics: stimulus.fromOptics,
        flashOpticalDurationMs: stimulus.opticalDurationMs,
        warnings: warnings,
      ),
    );
  }

  /// Hampel identifier: nulls samples further than k·1.4826·MAD from the
  /// local median (specular glints, partial-lid frames).
  static void _hampel(List<double?> d, {required int halfWindow, required double k}) {
    final orig = List<double?>.of(d);
    for (var i = 0; i < orig.length; i++) {
      final x = orig[i];
      if (x == null) continue;
      final w = <double>[
        for (var j = math.max(0, i - halfWindow); j <= math.min(orig.length - 1, i + halfWindow); j++)
          ?orig[j],
      ];
      if (w.length < 5) continue;
      w.sort();
      final med = w[w.length ~/ 2];
      final dev = [for (final v in w) (v - med).abs()]..sort();
      final mad = math.max(dev[dev.length ~/ 2] * 1.4826, 0.02);
      if ((x - med).abs() > k * mad) d[i] = null;
    }
  }
}
