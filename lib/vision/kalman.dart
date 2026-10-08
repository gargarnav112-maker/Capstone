import 'dart:math' as math;

/// Constant-velocity Kalman filter for one scalar signal with irregular
/// sampling (frame timestamps are used directly, so dropped frames are
/// handled correctly).
///
/// State x = [position, velocity]; process noise is the continuous white-noise
/// acceleration model with spectral density [q]; measurement noise variance
/// is [r]. Innovations larger than [gateSigma] standard deviations are
/// rejected (blinks, mis-detections) and the step becomes a pure prediction.
///
/// The filter records every step so [smooth] can run a Rauch–Tung–Striebel
/// backward pass: the forward filter is causal (for the live overlay) but lags
/// the signal, which would bias latency late and under-read peak velocity.
/// The RTS smoother is zero-lag and is what the biomarker extraction uses.
class ConstantVelocityKalman {
  ConstantVelocityKalman({
    required this.q,
    required this.r,
    this.gateSigma = 4.0,
    this.record = false,
  });

  final double q;
  final double r;
  final double gateSigma;
  final bool record;

  double _x = 0, _v = 0;
  // Covariance [[p00, p01], [p01, p11]].
  double _p00 = 0, _p01 = 0, _p11 = 0;
  double? _t;
  int _rejectStreak = 0;

  final List<_Step> _steps = [];

  bool get initialized => _t != null;
  double get position => _x;
  double get velocity => _v;
  double get positionVariance => _p00;

  void reset() {
    _t = null;
    _rejectStreak = 0;
    _steps.clear();
  }

  /// Predicted position at time [t] (seconds) without mutating the filter.
  double predictAt(double t) => _t == null ? _x : _x + _v * (t - _t!);

  /// Advances to time [t] (seconds) and, if [z] is non-null, fuses the
  /// measurement. Returns true if the measurement was accepted.
  bool update(double t, double? z) {
    if (_t == null) {
      if (z == null) return false;
      _t = t;
      _x = z;
      _v = 0;
      _p00 = r;
      _p01 = 0;
      _p11 = 1e3 * r + 1;
      if (record) _steps.add(_Step(t, _x, _v, _p00, _p01, _p11, _x, _v, _p00, _p01, _p11));
      return true;
    }
    final dt = math.max(1e-6, t - _t!);
    _t = t;

    // Predict: F = [[1, dt], [0, 1]], Q = q·[[dt³/3, dt²/2], [dt²/2, dt]].
    final xp = _x + _v * dt;
    final vp = _v;
    final p00 = _p00 + 2 * dt * _p01 + dt * dt * _p11 + q * dt * dt * dt / 3;
    final p01 = _p01 + dt * _p11 + q * dt * dt / 2;
    final p11 = _p11 + q * dt;

    var accepted = false;
    if (z != null) {
      final s = p00 + r;
      final innov = z - xp;
      // After several consecutive rejections the target has genuinely moved
      // (or the track was lost): re-acquire instead of rejecting forever.
      if (innov * innov <= gateSigma * gateSigma * s || _rejectStreak >= 5) {
        final k0 = p00 / s;
        final k1 = p01 / s;
        _x = xp + k0 * innov;
        _v = vp + k1 * innov;
        _p00 = (1 - k0) * p00;
        _p01 = (1 - k0) * p01;
        _p11 = p11 - k1 * p01;
        _rejectStreak = 0;
        accepted = true;
      } else {
        _rejectStreak++;
      }
    }
    if (!accepted) {
      _x = xp;
      _v = vp;
      _p00 = p00;
      _p01 = p01;
      _p11 = p11;
    }
    if (record) _steps.add(_Step(t, xp, vp, p00, p01, p11, _x, _v, _p00, _p01, _p11));
    return accepted;
  }

  /// Rauch–Tung–Striebel fixed-interval smoother over the recorded steps.
  /// Returns (t, position, velocity) triples.
  List<({double t, double x, double v})> smooth() {
    final n = _steps.length;
    if (n == 0) return const [];
    final xs = List<double>.filled(n, 0);
    final vs = List<double>.filled(n, 0);
    var sp00 = _steps.last.f00, sp01 = _steps.last.f01, sp11 = _steps.last.f11;
    xs[n - 1] = _steps.last.fx;
    vs[n - 1] = _steps.last.fv;
    for (var k = n - 2; k >= 0; k--) {
      final cur = _steps[k];
      final nxt = _steps[k + 1];
      final dt = nxt.t - cur.t;
      // C = P_f F^T P_p^{-1}
      final pf00 = cur.f00, pf01 = cur.f01, pf11 = cur.f11;
      // P_f F^T = [[pf00 + dt·pf01, pf01], [pf01 + dt·pf11, pf11]]
      final a00 = pf00 + dt * pf01, a01 = pf01;
      final a10 = pf01 + dt * pf11, a11 = pf11;
      final det = nxt.p00 * nxt.p11 - nxt.p01 * nxt.p01;
      if (det.abs() < 1e-18) {
        xs[k] = cur.fx;
        vs[k] = cur.fv;
        sp00 = pf00;
        sp01 = pf01;
        sp11 = pf11;
        continue;
      }
      final i00 = nxt.p11 / det, i01 = -nxt.p01 / det, i11 = nxt.p00 / det;
      final c00 = a00 * i00 + a01 * i01, c01 = a00 * i01 + a01 * i11;
      final c10 = a10 * i00 + a11 * i01, c11 = a10 * i01 + a11 * i11;
      final dx = xs[k + 1] - nxt.px;
      final dv = vs[k + 1] - nxt.pv;
      xs[k] = cur.fx + c00 * dx + c01 * dv;
      vs[k] = cur.fv + c10 * dx + c11 * dv;
      // P_s = P_f + C (P_s,next − P_p,next) Cᵀ
      final d00 = sp00 - nxt.p00, d01 = sp01 - nxt.p01, d11 = sp11 - nxt.p11;
      final e00 = c00 * d00 + c01 * d01, e01 = c00 * d01 + c01 * d11;
      final e10 = c10 * d00 + c11 * d01, e11 = c10 * d01 + c11 * d11;
      sp00 = pf00 + e00 * c00 + e01 * c01;
      sp01 = pf01 + e00 * c10 + e01 * c11;
      sp11 = pf11 + e10 * c10 + e11 * c11;
    }
    return [
      for (var k = 0; k < n; k++) (t: _steps[k].t, x: xs[k], v: vs[k]),
    ];
  }
}

class _Step {
  _Step(this.t, this.px, this.pv, this.p00, this.p01, this.p11, this.fx, this.fv, this.f00,
      this.f01, this.f11);

  final double t;
  // Predicted (prior) state & covariance.
  final double px, pv, p00, p01, p11;
  // Filtered (posterior) state & covariance.
  final double fx, fv, f00, f01, f11;
}

/// 2-D tremor stabiliser: independent constant-velocity filters on x and y
/// (equivalent to a block-diagonal 4-state filter with isotropic noise).
class CenterKalman {
  CenterKalman({double q = 2e5, double r = 1.0})
      : _x = ConstantVelocityKalman(q: q, r: r, gateSigma: 6),
        _y = ConstantVelocityKalman(q: q, r: r, gateSigma: 6);

  final ConstantVelocityKalman _x;
  final ConstantVelocityKalman _y;

  bool get initialized => _x.initialized;
  double get x => _x.position;
  double get y => _y.position;

  (double, double) predictAt(double t) => (_x.predictAt(t), _y.predictAt(t));

  void update(double t, double? mx, double? my) {
    _x.update(t, mx);
    _y.update(t, my);
  }

  void reset() {
    _x.reset();
    _y.reset();
  }
}
