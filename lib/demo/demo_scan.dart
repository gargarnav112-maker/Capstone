import 'dart:math' as math;

import '../biomarkers/plr_analyzer.dart';
import '../biomarkers/plr_models.dart';
import '../biomarkers/reference_curves.dart';

/// Synthetic scan used for UI previews, widget tests and the "demo result"
/// fallback when no camera is available (simulator). It runs the real
/// analysis pipeline on a modelled 120 fps recording.
ScanResult buildDemoScan({PlrModel model = PlrModel.healthy, Eye eye = Eye.right, int seed = 3}) {
  final rnd = math.Random(seed);
  double gauss() =>
      math.sqrt(-2 * math.log(rnd.nextDouble() + 1e-12)) * math.cos(2 * math.pi * rnd.nextDouble());
  const onUs = 1000000;
  final frames = <FrameSample>[];
  for (var i = 0;; i++) {
    final ts = (onUs - 500000 + i * 1e6 / 120).round();
    final rel = (ts - onUs - 15000) / 1000;
    if (rel > 3000) break;
    frames.add(FrameSample(
      timestampUs: ts,
      meanLuma: (rel >= 0 && rel < 100 ? 165 : 112) + gauss(),
      diameterMm: model.diameterAt(rel) + 0.025 * gauss(),
    ));
  }
  final analysis = const PlrAnalyzer().analyze(
    frames,
    PlrAnalyzer.detectStimulus(frames, commandOnUs: onUs),
  );
  return ScanResult(
    scanId: '3f6c1a2e-9b4d-4c1e-8a7f-0d5e2b9c4a11',
    eye: eye,
    recordedAt: DateTime(2026, 10, 8, 14, 32, 5),
    metrics: analysis.metrics,
    curve: analysis.curve,
    rawCurve: analysis.rawCurve,
    quality: analysis.quality,
    sensor: const {'iso': 400.0, 'exposureUs': 4000.0, 'focus': 0.82, 'fps': 120.0, 'manual': true},
  );
}
