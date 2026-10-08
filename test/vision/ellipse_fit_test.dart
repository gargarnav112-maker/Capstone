import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/vision/ellipse_fit.dart';

(Float64List, Float64List) _points(
  double cx,
  double cy,
  double a,
  double b,
  double th, {
  int n = 120,
  double from = 0,
  double to = 2 * math.pi,
  double noise = 0,
  int seed = 7,
}) {
  final rnd = math.Random(seed);
  final xs = Float64List(n), ys = Float64List(n);
  for (var i = 0; i < n; i++) {
    final t = from + (to - from) * i / n;
    final x = a * math.cos(t), y = b * math.sin(t);
    xs[i] = cx + x * math.cos(th) - y * math.sin(th) + noise * (rnd.nextDouble() - 0.5);
    ys[i] = cy + x * math.sin(th) + y * math.cos(th) + noise * (rnd.nextDouble() - 0.5);
  }
  return (xs, ys);
}

void main() {
  test('recovers an exact rotated ellipse', () {
    final (xs, ys) = _points(120.5, 80.25, 40, 25, 0.6);
    final e = EllipseFitter.fit(xs, ys)!;
    expect(e.cx, closeTo(120.5, 1e-6));
    expect(e.cy, closeTo(80.25, 1e-6));
    expect(e.a, closeTo(40, 1e-6));
    expect(e.b, closeTo(25, 1e-6));
    expect(e.theta, closeTo(0.6, 1e-6));
  });

  test('handles axis-aligned and vertical-major ellipses', () {
    var (xs, ys) = _points(0, 0, 30, 10, 0);
    var e = EllipseFitter.fit(xs, ys)!;
    expect(e.a, closeTo(30, 1e-6));
    expect(e.theta.abs(), lessThan(1e-6));

    (xs, ys) = _points(0, 0, 30, 10, math.pi / 2);
    e = EllipseFitter.fit(xs, ys)!;
    expect(e.a, closeTo(30, 1e-6));
    expect(e.theta.abs(), closeTo(math.pi / 2, 1e-6));
  });

  test('is accurate on a partial arc with noise (eyelid occlusion)', () {
    final (xs, ys) = _points(50, 60, 22, 20, -0.3, from: 0.2, to: 4.2, noise: 0.6);
    final e = EllipseFitter.fit(xs, ys)!;
    expect(e.majorDiameter, closeTo(44, 1.5));
    expect(e.cx, closeTo(50, 1));
  });

  test('robust refit rejects glint/eyelash outliers', () {
    final (xs, ys) = _points(100, 100, 30, 29, 0, n: 200, noise: 0.4);
    // Corrupt 12 % of the contour with a notch pulled 8 px inward.
    for (var i = 0; i < 24; i++) {
      xs[i] -= 8;
    }
    final plain = EllipseFitter.fit(xs, ys)!;
    final (robust, inliers) = EllipseFitter.fitRobust(xs, ys)!;
    expect((robust.a - 30).abs(), lessThan((plain.a - 30).abs()));
    expect(robust.a, closeTo(30, 0.6));
    expect(inliers, lessThan(0.95));
  });

  test('returns null for degenerate input', () {
    expect(EllipseFitter.fit(Float64List(3), Float64List(3)), isNull);
    final line = Float64List.fromList(List.generate(20, (i) => i.toDouble()));
    expect(EllipseFitter.fit(line, line), isNull);
  });
}
