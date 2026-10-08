// ignore_for_file: avoid_print
// Benchmarks the per-frame vision pipeline. Run on the VM with
// `dart run tool/bench_pipeline.dart`, or compile to JS to measure the web
// build: `dart compile js -O2 tool/bench_pipeline.dart -o /tmp/b.js && node /tmp/b.js`.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:synapse/vision/grayscale.dart';
import 'package:synapse/vision/pupil_tracker.dart';

GrayImage _eye(int n, double r, int seed) {
  final rnd = math.Random(seed);
  final px = Uint8List(n * n);
  final c = n / 2;
  for (var y = 0; y < n; y++) {
    for (var x = 0; x < n; x++) {
      final d = math.sqrt((x - c) * (x - c) + (y - c) * (y - c));
      final v = d < r ? 16.0 : (d < 50 ? 92.0 : 212.0);
      px[y * n + x] = (v + rnd.nextDouble() * 8).round().clamp(0, 255);
    }
  }
  return GrayImage(n, n, px);
}

void main() {
  for (final n in [128, 256]) {
    final frames = [for (var i = 0; i < 20; i++) _eye(n, n / 6, i)];
    final tracker = PupilTracker();
    for (var i = 0; i < 20; i++) {
      tracker.process(frames[i % 20], timestampUs: i * 16667, frameIndex: i);
    }
    final sw = Stopwatch()..start();
    const iters = 200;
    for (var i = 0; i < iters; i++) {
      tracker.process(frames[i % 20], timestampUs: (i + 20) * 16667, frameIndex: i + 20);
    }
    print('ROI ${n}x$n: ${(sw.elapsedMicroseconds / iters / 1000).toStringAsFixed(2)} ms/frame');
  }
}
