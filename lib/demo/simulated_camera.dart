import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:synapse_camera/synapse_camera.dart';

import '../biomarkers/reference_curves.dart';

/// Patient profiles the simulator can present.
enum SimulatedPatient {
  healthy('Healthy', PlrModel.healthy),
  sluggish('Sluggish', PlrModel.abnormal),
  nonReactive(
    'Fixed',
    PlrModel(baselineMm: 6.2, latencyMs: 300, amplitudeMm: 0.03, tauConstrictMs: 300, tauDilateMs: 1500),
  );

  const SimulatedPatient(this.label, this.model);

  final String label;
  final PlrModel model;
}

/// What the simulated scene looks like right now, for drawing the preview.
class SimulatedScene {
  const SimulatedScene({
    required this.offsetX,
    required this.offsetY,
    required this.pupilRadiusPx,
    required this.irisRadiusPx,
    required this.torchOn,
  });

  /// Eye centre offset from the frame centre (scene px, 256-wide frame).
  final double offsetX;
  final double offsetY;
  final double pupilRadiusPx;
  final double irisRadiusPx;
  final bool torchOn;
}

/// A software stand-in for the native camera: a synthetic eye whose pupil
/// follows a physiological light-reflex model, with hand tremor, sensor
/// noise and a torch that brightens the scene 15 ms after it is commanded.
///
/// The frames go through exactly the same vision pipeline and biomarker
/// analysis as real camera frames; only the photons are simulated. Used by
/// the web build (no camera or torch control in a browser) and on devices
/// without a usable camera.
class SimulatedCamera extends SynapseCamera {
  SimulatedCamera({this.fps = 60, this.patient = SimulatedPatient.healthy});

  final double fps;
  SimulatedPatient patient;

  @override
  bool get isHardware => false;

  static const frameSize = 256;
  static const roiSize = 128;
  static const irisRadiusPx = 50.0;

  /// Iris diameter 100 px ⇔ 11.7 mm (HVID).
  static const pxPerMm = 2 * irisRadiusPx / 11.7;
  static const _torchLagUs = 15000;

  final scene = ValueNotifier<SimulatedScene?>(null);
  final _frames = StreamController<CameraLumaFrame>.broadcast();
  final _rnd = math.Random(7);
  Timer? _timer;
  int _clockUs = 0;
  int _index = 0;
  int? _torchOnUs;
  int? _torchOffUs;

  // Tremor: slow drift + physiological 8–12 Hz component.
  double _driftX = 0, _driftY = 0;

  @override
  Future<CameraSession> initialize({int targetFps = 120, int minFps = 60, int roiSize = 256}) async {
    _timer?.cancel();
    _wall
      ..reset()
      ..start();
    _emitted = 0;
    _timer = Timer.periodic(Duration(microseconds: (1e6 / fps).round()), (_) => _tick());
    return CameraSession(
      textureId: -1,
      previewWidth: frameSize.toDouble(),
      previewHeight: frameSize.toDouble(),
      fps: fps,
      quarterTurns: 0,
      supportsManualSensor: true,
    );
  }

  double _pupilDiameterMm() {
    final on = _torchOnUs;
    if (on == null) return patient.model.baselineMm;
    return patient.model.diameterAt((_clockUs - on - _torchLagUs) / 1000);
  }

  bool get _lit {
    final on = _torchOnUs;
    if (on == null || _clockUs < on + _torchLagUs) return false;
    final off = _torchOffUs;
    return off == null || _clockUs < off + _torchLagUs;
  }

  /// Emits every frame the sensor "would have" captured since the last tick,
  /// so the simulated clock keeps real time even when the UI thread is busy
  /// drawing (slow GPUs, software WebGL). Only the last frame of a burst
  /// updates the preview.
  void _tick() {
    final due = (_wall.elapsedMicroseconds * fps / 1e6).floor() - _emitted;
    final n = due.clamp(0, 12);
    for (var i = 0; i < n; i++) {
      _emit(updateScene: i == n - 1);
    }
    _emitted += due; // drop (not queue) anything beyond the burst cap
  }

  final _wall = Stopwatch();
  int _emitted = 0;

  void _emit({bool updateScene = true}) {
    // Timestamps advance by exactly one frame period, like a sensor clock,
    // even if the browser timer jitters.
    _clockUs += (1e6 / fps).round();
    final t = _clockUs / 1e6;
    _driftX = (_driftX + (_rnd.nextDouble() - 0.5) * 0.6).clamp(-7.0, 7.0);
    _driftY = (_driftY + (_rnd.nextDouble() - 0.5) * 0.6).clamp(-7.0, 7.0);
    final ox = _driftX + 1.2 * math.sin(2 * math.pi * 9.5 * t);
    final oy = _driftY + 1.0 * math.cos(2 * math.pi * 10.5 * t);
    final r = _pupilDiameterMm() / 2 * pxPerMm;
    final lit = _lit;
    if (updateScene) {
      scene.value = SimulatedScene(
        offsetX: ox,
        offsetY: oy,
        pupilRadiusPx: r,
        irisRadiusPx: irisRadiusPx,
        torchOn: lit,
      );
    }
    _frames.add(
      CameraLumaFrame(
        width: roiSize,
        height: roiSize,
        bytes: _render(roiSize / 2 + ox, roiSize / 2 + oy, r, lit),
        timestampUs: _clockUs,
        index: _index++,
        torchOn: lit,
      ),
    );
  }

  /// Static eye (lids, sclera, textured iris, no pupil) rendered once on a
  /// canvas larger than the ROI so tremor can shift it; plus a noise table.
  late final Float32List _base = _renderBase();
  late final Float32List _noise = Float32List.fromList(
    List.generate(_noiseLen, (_) => (_rnd.nextDouble() + _rnd.nextDouble() - 1) * 5),
  );
  static const _pad = 16;
  static const _baseSize = roiSize + 2 * _pad;
  static const _noiseLen = 1 << 16;

  Float32List _renderBase() {
    const n = _baseSize;
    const c = n / 2;
    const irisR = irisRadiusPx;
    final out = Float32List(n * n);
    for (var y = 0; y < n; y++) {
      final dy = y - c;
      for (var x = 0; x < n; x++) {
        final dx = x - c;
        final rr = dx * dx + dy * dy;
        final lidY = 0.9 * irisR * math.sqrt(math.max(0, 1 - (dx * dx) / (4.84 * irisR * irisR)));
        double v;
        if (dy.abs() > lidY) {
          v = 150;
        } else if (rr > irisR * irisR) {
          v = 212;
        } else {
          final ang = math.atan2(dy, dx);
          v = 92 + 9 * math.sin(ang * 23) * math.cos(math.sqrt(rr) / 5);
        }
        out[y * n + x] = v;
      }
    }
    return out;
  }

  /// Renders the centre ROI as 8-bit luma: the static eye shifted by the
  /// tremor, an anti-aliased pupil at the exact sub-pixel size, the corneal
  /// glint, torch gain and sensor noise.
  Uint8List _render(double cx, double cy, double pupilR, bool lit) {
    const n = roiSize;
    final px = Uint8List(n * n);
    final gain = lit ? 1.18 : 1.0;
    final sx = (cx - n / 2).round(), sy = (cy - n / 2).round();
    final noiseAt = _rnd.nextInt(_noiseLen - n * n);
    final r2out = (pupilR + 1) * (pupilR + 1);
    final gx0 = cx + 0.4 * pupilR, gy0 = cy - 0.4 * pupilR;
    for (var y = 0; y < n; y++) {
      final by = y - sy + _pad;
      final dy = y - cy;
      for (var x = 0; x < n; x++) {
        final bx = x - sx + _pad;
        var v = (bx >= 0 && by >= 0 && bx < _baseSize && by < _baseSize) ? _base[by * _baseSize + bx] : 150.0;
        final dx = x - cx;
        final rr = dx * dx + dy * dy;
        if (rr < r2out) {
          // Coverage of the pixel by the pupil disc (≈ 1 px AA band).
          final cov = (pupilR - math.sqrt(rr) + 0.5).clamp(0.0, 1.0);
          v = v + (16 - v) * cov;
        }
        final gx = x - gx0, gy = y - gy0;
        if (gx * gx + gy * gy < 9) v = 250;
        v = v * gain + _noise[noiseAt + y * n + x];
        px[y * n + x] = v < 0 ? 0 : (v > 255 ? 255 : v.round());
      }
    }
    return px;
  }

  @override
  Future<SensorLock> lock({double? iso, double? exposureDurationUs, double? focus}) async {
    // Each scan starts from a fully dark-adapted, resting pupil.
    resetStimulus();
    return SensorLock(iso: 400, exposureDurationUs: 1e6 / fps / 2, focus: 0.8, fps: fps, manual: true);
  }

  @override
  Future<void> unlock() async {}

  @override
  Future<TorchPulse> flashPulse({int durationMs = 100, double level = 1.0}) async {
    _torchOnUs = _clockUs;
    _torchOffUs = null;
    await Future<void>.delayed(Duration(milliseconds: durationMs));
    _torchOffUs = _torchOnUs! + durationMs * 1000;
    return TorchPulse(onUs: _torchOnUs!, offUs: _torchOffUs!);
  }

  /// Returns the scene to the resting (pre-flash) pupil.
  void resetStimulus() {
    _torchOnUs = null;
    _torchOffUs = null;
  }

  @override
  Future<int> nowUs() async => _clockUs;

  @override
  Stream<CameraLumaFrame> frames() => _frames.stream;

  @override
  Future<void> dispose() async {
    _timer?.cancel();
    await _frames.close();
    scene.dispose();
  }
}
