import 'dart:math' as math;
import 'dart:typed_data';

/// Geometric ellipse: centre, semi-axes (a ≥ b) and rotation of the major axis.
class Ellipse {
  const Ellipse({
    required this.cx,
    required this.cy,
    required this.a,
    required this.b,
    required this.theta,
  });

  final double cx;
  final double cy;

  /// Semi-major axis (px).
  final double a;

  /// Semi-minor axis (px).
  final double b;

  /// Major-axis angle in radians, (-π/2, π/2].
  final double theta;

  /// Pupil diameter estimate. Off-axis viewing foreshortens the pupil along one
  /// direction only, so the *major* axis is the unforeshortened diameter.
  double get majorDiameter => 2 * a;
  double get minorDiameter => 2 * b;
  double get eccentricity => a == 0 ? 0 : math.sqrt(1 - (b * b) / (a * a));

  /// Approximate normalised distance of (x, y) from the boundary:
  /// 0 on the ellipse, ±1 at the centre / twice the radius.
  double radialResidual(double x, double y) {
    final c = math.cos(theta);
    final s = math.sin(theta);
    final dx = x - cx;
    final dy = y - cy;
    final u = (dx * c + dy * s) / a;
    final v = (-dx * s + dy * c) / b;
    return math.sqrt(u * u + v * v) - 1;
  }

  @override
  String toString() =>
      'Ellipse(c=(${cx.toStringAsFixed(2)}, ${cy.toStringAsFixed(2)}), '
      'a=${a.toStringAsFixed(2)}, b=${b.toStringAsFixed(2)}, θ=${theta.toStringAsFixed(3)})';
}

/// Pipeline step 4: direct least-squares ellipse fitting.
///
/// Implements Fitzgibbon, Pilu & Fisher (1999) in the numerically stable
/// partitioned form of Halíř & Flusser (1998). The ellipse-specific constraint
/// 4AC − B² = 1 guarantees an ellipse (never a hyperbola) even for a partial
/// arc, which matters when the eyelid occludes part of the pupil.
class EllipseFitter {
  const EllipseFitter._();

  /// Fits an ellipse to the points. Returns null for < 6 points or degenerate
  /// geometry.
  static Ellipse? fit(Float64List xs, Float64List ys) {
    final n = xs.length;
    if (n < 6 || ys.length != n) return null;

    // Normalise (centre + isotropic scale) for conditioning.
    var mx = 0.0, my = 0.0;
    for (var i = 0; i < n; i++) {
      mx += xs[i];
      my += ys[i];
    }
    mx /= n;
    my /= n;
    var sc = 0.0;
    for (var i = 0; i < n; i++) {
      sc += math.sqrt((xs[i] - mx) * (xs[i] - mx) + (ys[i] - my) * (ys[i] - my));
    }
    sc /= n;
    if (sc < 1e-9) return null;

    // Scatter blocks S1 = D1ᵀD1, S2 = D1ᵀD2, S3 = D2ᵀD2 with
    // D1 = [x², xy, y²], D2 = [x, y, 1].
    final s1 = List.generate(3, (_) => Float64List(3));
    final s2 = List.generate(3, (_) => Float64List(3));
    final s3 = List.generate(3, (_) => Float64List(3));
    for (var i = 0; i < n; i++) {
      final x = (xs[i] - mx) / sc;
      final y = (ys[i] - my) / sc;
      final d1 = [x * x, x * y, y * y];
      final d2 = [x, y, 1.0];
      for (var r = 0; r < 3; r++) {
        for (var c = 0; c < 3; c++) {
          s1[r][c] += d1[r] * d1[c];
          s2[r][c] += d1[r] * d2[c];
          s3[r][c] += d2[r] * d2[c];
        }
      }
    }

    final s3inv = _inv3(s3);
    if (s3inv == null) return null;
    // T = −S3⁻¹ S2ᵀ
    final t = List.generate(3, (_) => Float64List(3));
    for (var r = 0; r < 3; r++) {
      for (var c = 0; c < 3; c++) {
        var acc = 0.0;
        for (var k = 0; k < 3; k++) {
          acc += s3inv[r][k] * s2[c][k];
        }
        t[r][c] = -acc;
      }
    }
    // M = S1 + S2 T, then premultiply by C1⁻¹ = [[0,0,½],[0,−1,0],[½,0,0]].
    final m = List.generate(3, (_) => Float64List(3));
    for (var r = 0; r < 3; r++) {
      for (var c = 0; c < 3; c++) {
        var acc = s1[r][c];
        for (var k = 0; k < 3; k++) {
          acc += s2[r][k] * t[k][c];
        }
        m[r][c] = acc;
      }
    }
    final mm = [
      Float64List.fromList([m[2][0] / 2, m[2][1] / 2, m[2][2] / 2]),
      Float64List.fromList([-m[1][0], -m[1][1], -m[1][2]]),
      Float64List.fromList([m[0][0] / 2, m[0][1] / 2, m[0][2] / 2]),
    ];

    // The admissible eigenvector is the one satisfying 4AC − B² > 0.
    Float64List? a1;
    var bestCond = 0.0;
    for (final lambda in _realEigenvalues(mm)) {
      final v = _nullVector(mm, lambda);
      if (v == null) continue;
      final cond = 4 * v[0] * v[2] - v[1] * v[1];
      if (cond > bestCond) {
        bestCond = cond;
        a1 = v;
      }
    }
    if (a1 == null) return null;

    final a2 = Float64List(3);
    for (var r = 0; r < 3; r++) {
      a2[r] = t[r][0] * a1[0] + t[r][1] * a1[1] + t[r][2] * a1[2];
    }
    final e = _conicToEllipse(a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]);
    if (e == null) return null;
    return Ellipse(
      cx: e.cx * sc + mx,
      cy: e.cy * sc + my,
      a: e.a * sc,
      b: e.b * sc,
      theta: e.theta,
    );
  }

  /// Fit, drop points whose residual exceeds [k]·MAD-scaled spread (eyelid,
  /// eyelashes, corneal glint notches), refit. Returns the refined ellipse and
  /// the inlier ratio.
  static (Ellipse, double)? fitRobust(
    Float64List xs,
    Float64List ys, {
    int iterations = 2,
    double k = 2.5,
  }) {
    var e = fit(xs, ys);
    if (e == null) return null;
    var cx = xs, cy = ys;
    for (var it = 0; it < iterations; it++) {
      final res = Float64List(cx.length);
      for (var i = 0; i < cx.length; i++) {
        res[i] = e!.radialResidual(cx[i], cy[i]).abs();
      }
      final sorted = Float64List.fromList(res)..sort();
      final med = sorted[sorted.length ~/ 2];
      final cut = math.max(k * 1.4826 * med, 0.5 / math.max(1.0, e!.b));
      final keepX = <double>[];
      final keepY = <double>[];
      for (var i = 0; i < cx.length; i++) {
        if (res[i] <= cut) {
          keepX.add(cx[i]);
          keepY.add(cy[i]);
        }
      }
      if (keepX.length == cx.length || keepX.length < 6) break;
      final refit = fit(Float64List.fromList(keepX), Float64List.fromList(keepY));
      if (refit == null) break;
      e = refit;
      cx = Float64List.fromList(keepX);
      cy = Float64List.fromList(keepY);
    }
    return (e!, cx.length / xs.length);
  }

  static Ellipse? _conicToEllipse(double a, double b, double c, double d, double e, double f) {
    // The conic is defined up to sign; fix A + C > 0 so the closed-form
    // semi-axis expressions below are real.
    if (a + c < 0) {
      a = -a;
      b = -b;
      c = -c;
      d = -d;
      e = -e;
      f = -f;
    }
    final den = b * b - 4 * a * c;
    if (den >= 0) return null; // not an ellipse
    final x0 = (2 * c * d - b * e) / den;
    final y0 = (2 * a * e - b * d) / den;
    final common = 2 * (a * e * e + c * d * d - b * d * e + den * f);
    final root = math.sqrt((a - c) * (a - c) + b * b);
    final p = common * (a + c + root);
    final q = common * (a + c - root);
    if (p < 0 || q < 0) return null;
    final r1 = -math.sqrt(p) / den;
    final r2 = -math.sqrt(q) / den;
    // r1 ≥ r2 always (root ≥ 0, common>0 for a real ellipse with A+C>0) but
    // the overall sign of the conic is arbitrary, so sort explicitly.
    final major = math.max(r1, r2);
    final minor = math.min(r1, r2);
    if (!(minor > 0) || !major.isFinite) return null;

    // Angle of the axis associated with r1 (Wikipedia, general ellipse).
    double theta;
    if (b.abs() < 1e-12) {
      theta = a <= c ? 0 : math.pi / 2;
    } else {
      theta = math.atan((c - a - root) / b);
    }
    // Make theta the direction of the *major* axis.
    if (r1 < r2) theta += math.pi / 2;
    if (theta > math.pi / 2) theta -= math.pi;
    if (theta <= -math.pi / 2) theta += math.pi;
    return Ellipse(cx: x0, cy: y0, a: major, b: minor, theta: theta);
  }

  static List<Float64List>? _inv3(List<Float64List> m) {
    final a = m[0][0], b = m[0][1], c = m[0][2];
    final d = m[1][0], e = m[1][1], f = m[1][2];
    final g = m[2][0], h = m[2][1], i = m[2][2];
    final det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det.abs() < 1e-12) return null;
    final s = 1 / det;
    return [
      Float64List.fromList([(e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s]),
      Float64List.fromList([(f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s]),
      Float64List.fromList([(d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s]),
    ];
  }

  /// Real roots of det(M − λI) = −λ³ + tr·λ² − c1·λ + det.
  static List<double> _realEigenvalues(List<Float64List> m) {
    final tr = m[0][0] + m[1][1] + m[2][2];
    final c1 = m[0][0] * m[1][1] - m[0][1] * m[1][0] +
        m[0][0] * m[2][2] - m[0][2] * m[2][0] +
        m[1][1] * m[2][2] - m[1][2] * m[2][1];
    final det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // λ³ − tr λ² + c1 λ − det = 0
    return _solveCubic(-tr, c1, -det);
  }

  /// Real roots of x³ + a x² + b x + c = 0.
  static List<double> _solveCubic(double a, double b, double c) {
    final q = (a * a - 3 * b) / 9;
    final r = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
    final q3 = q * q * q;
    if (r * r < q3) {
      final th = math.acos((r / math.sqrt(q3)).clamp(-1.0, 1.0));
      final sq = -2 * math.sqrt(q);
      return [
        sq * math.cos(th / 3) - a / 3,
        sq * math.cos((th + 2 * math.pi) / 3) - a / 3,
        sq * math.cos((th - 2 * math.pi) / 3) - a / 3,
      ];
    }
    final sgn = r < 0 ? 1.0 : -1.0;
    final aa = sgn * math.pow(r.abs() + math.sqrt(r * r - q3), 1 / 3).toDouble();
    final bb = aa == 0 ? 0.0 : q / aa;
    return [(aa + bb) - a / 3];
  }

  /// Null vector of (M − λI): the largest cross product of two of its rows.
  static Float64List? _nullVector(List<Float64List> m, double lambda) {
    final rows = [
      for (var r = 0; r < 3; r++)
        [for (var c = 0; c < 3; c++) m[r][c] - (r == c ? lambda : 0)],
    ];
    Float64List? best;
    var bestNorm = 0.0;
    for (final (i, j) in const [(0, 1), (0, 2), (1, 2)]) {
      final u = rows[i], v = rows[j];
      final x = Float64List.fromList([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
      ]);
      final norm = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
      if (norm > bestNorm) {
        bestNorm = norm;
        best = x;
      }
    }
    if (best == null || bestNorm < 1e-30) return null;
    final s = 1 / math.sqrt(bestNorm);
    return Float64List.fromList([best[0] * s, best[1] * s, best[2] * s]);
  }
}
