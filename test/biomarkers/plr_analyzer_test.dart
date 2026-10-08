import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/biomarkers/plr_analyzer.dart';
import 'package:synapse/biomarkers/plr_models.dart';
import 'package:synapse/biomarkers/reference_curves.dart';
import 'package:synapse/vision/kalman.dart';

/// Simulates the controller's recording: 500 ms baseline, 100 ms flash
/// (luminance step that lags the command by [torchLagMs]), response to 3 s.
List<FrameSample> _record(
  PlrModel model, {
  double fps = 120,
  double noiseMm = 0.03,
  double torchLagMs = 18,
  int stimulusAtUs = 1000000,
  int seed = 11,
  Set<int> blinkFrames = const {},
}) {
  final rnd = math.Random(seed);
  double gauss() =>
      math.sqrt(-2 * math.log(rnd.nextDouble() + 1e-12)) * math.cos(2 * math.pi * rnd.nextDouble());
  final out = <FrameSample>[];
  final dtUs = 1e6 / fps;
  final opticalOnUs = stimulusAtUs + torchLagMs * 1000;
  for (var i = 0;; i++) {
    final ts = (stimulusAtUs - 500000 + i * dtUs).round();
    final rel = (ts - opticalOnUs) / 1000; // ms from optical onset
    if (rel > 3000) break;
    final lit = rel >= 0 && rel < 100;
    out.add(FrameSample(
      timestampUs: ts,
      meanLuma: (lit ? 160 : 110) + gauss(),
      diameterMm: blinkFrames.contains(i) ? null : model.diameterAt(rel) + noiseMm * gauss(),
    ));
  }
  return out;
}

double _analyticMcv(PlrModel m) {
  var best = 0.0;
  for (var t = m.latencyMs; t < m.troughMs; t += 0.5) {
    best = math.max(best, -m.velocityAt(t));
  }
  return best;
}

void main() {
  const analyzer = PlrAnalyzer();

  test('detects the optical flash onset, not the torch command time', () {
    final frames = _record(PlrModel.healthy, torchLagMs: 18);
    final s = PlrAnalyzer.detectStimulus(frames, commandOnUs: 1000000);
    expect(s.fromOptics, isTrue);
    expect((s.onsetUs - 1018000).abs(), lessThan(9000)); // within ~1 frame
    expect(s.opticalDurationMs, closeTo(100, 12));
  });

  test('healthy reflex: latency, velocities and index', () {
    final frames = _record(PlrModel.healthy);
    final stim = PlrAnalyzer.detectStimulus(frames, commandOnUs: 1000000);
    final a = analyzer.analyze(frames, stim);
    final m = a.metrics;
    expect(m.baselineDiameterMm, closeTo(5.0, 0.03));
    expect(m.latencyMs, closeTo(220, 25));
    expect(m.minDiameterMm, closeTo(3.4, 0.05));
    expect(m.constrictionPercent, closeTo(32, 1.5));
    expect(m.maxConstrictionVelocityMmS, closeTo(_analyticMcv(PlrModel.healthy), 0.15 * _analyticMcv(PlrModel.healthy)));
    expect(m.avgConstrictionVelocityMmS, greaterThan(0.8));
    expect(m.dilationVelocityMmS, greaterThan(0.3));
    expect(m.npi, greaterThanOrEqualTo(3.0));
    expect(m.npiCategory, NpiCategory.brisk);
    expect(a.quality.effectiveFps, closeTo(120, 1));
    expect(a.quality.warnings, isEmpty);
  });

  test('abnormal reflex scores below 3', () {
    final frames = _record(PlrModel.abnormal);
    final a = analyzer.analyze(frames, PlrAnalyzer.detectStimulus(frames, commandOnUs: 1000000));
    expect(a.metrics.latencyMs, closeTo(380, 60));
    expect(a.metrics.avgConstrictionVelocityMmS, lessThan(0.8));
    expect(a.metrics.npi, lessThan(3.0));
    expect(a.metrics.npiCategory, NpiCategory.sluggish);
  });

  test('non-reactive pupil → index 0, no latency', () {
    const fixed = PlrModel(
        baselineMm: 6, latencyMs: 300, amplitudeMm: 0.02, tauConstrictMs: 300, tauDilateMs: 1500);
    final frames = _record(fixed);
    final a = analyzer.analyze(frames, PlrAnalyzer.detectStimulus(frames, commandOnUs: 1000000));
    expect(a.metrics.latencyMs, isNull);
    expect(a.metrics.npi, 0);
    expect(a.metrics.npiCategory, NpiCategory.nonReactive);
  });

  test('works at 60 fps and through a blink', () {
    final blink = {for (var i = 70; i < 78; i++) i}; // ~130 ms gap mid-constriction at 60 fps
    final frames = _record(PlrModel.healthy, fps: 60, blinkFrames: blink);
    final a = analyzer.analyze(frames, PlrAnalyzer.detectStimulus(frames, commandOnUs: 1000000));
    expect(a.metrics.latencyMs, closeTo(220, 35));
    expect(a.metrics.npiCategory, NpiCategory.brisk);
    expect(a.quality.blinkCount, 1);
  });

  test('Hampel filter removes glint spikes', () {
    final frames = _record(PlrModel.healthy);
    final spiked = [
      for (var i = 0; i < frames.length; i++)
        i % 37 == 0 && frames[i].diameterMm != null
            ? FrameSample(timestampUs: frames[i].timestampUs, meanLuma: frames[i].meanLuma, diameterMm: frames[i].diameterMm! - 1.2)
            : frames[i],
    ];
    final a = analyzer.analyze(spiked, PlrAnalyzer.detectStimulus(spiked, commandOnUs: 1000000));
    expect(a.metrics.baselineDiameterMm, closeTo(5.0, 0.05));
    expect(a.metrics.latencyMs, closeTo(220, 25));
  });

  test('RTS smoother is zero-lag where the forward filter lags', () {
    final kf = ConstantVelocityKalman(q: 300, r: 0.0025, record: true);
    // Ramp at −3 mm/s starting at t = 0.5 s.
    final ts = [for (var i = 0; i < 240; i++) i / 120];
    double truth(double t) => t < 0.5 ? 5 : 5 - 3 * (t - 0.5);
    final forward = <double>[];
    for (final t in ts) {
      kf.update(t, truth(t));
      forward.add(kf.position);
    }
    final smooth = kf.smooth();
    final i = ts.indexWhere((t) => t >= 0.6);
    expect((smooth[i].x - truth(ts[i])).abs(), lessThan((forward[i] - truth(ts[i])).abs() + 1e-9));
    expect(smooth[i].v, closeTo(-3, 0.3));
  });
}
