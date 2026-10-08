import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/biomarkers/plr_models.dart';
import 'package:synapse/biomarkers/reference_curves.dart';
import 'package:synapse/hardware/hardware_sync_controller.dart';
import 'package:synapse_camera/synapse_camera.dart';

import '../support/synthetic_eye.dart';

/// Simulated sensor: 120 fps synthetic eye whose pupil follows [model]
/// relative to the *optical* flash onset (torch lag 15 ms). Records the
/// order of hardware calls so the 4-step protocol can be asserted.
class FakeCamera extends SynapseCamera {
  FakeCamera({this.model = PlrModel.healthy, this.fps = 120});

  final PlrModel model;
  final double fps;

  @override
  bool get isHardware => false;
  final calls = <String>[];
  final _frames = StreamController<CameraLumaFrame>.broadcast();
  Timer? _timer;
  int _clockUs = 10000000;
  int _index = 0;
  int? _torchOnUs;
  int? _torchOffUs;
  bool locked = false;

  static const pxPerMm = 50 / 11.7 * 2; // iris radius 50 px ⇔ HVID 11.7 mm

  @override
  Future<CameraSession> initialize({int targetFps = 120, int minFps = 60, int roiSize = 256}) async {
    calls.add('initialize');
    _timer = Timer.periodic(Duration(microseconds: (1e6 / fps).round()), (_) => _emit());
    return CameraSession(
      textureId: 1,
      previewWidth: 720,
      previewHeight: 1280,
      fps: fps,
      quarterTurns: 0,
      supportsManualSensor: true,
    );
  }

  void _emit() {
    _clockUs += (1e6 / fps).round();
    final opticalOn = _torchOnUs == null ? null : _torchOnUs! + 15000;
    final lit = opticalOn != null && _clockUs >= opticalOn && (_torchOffUs == null || _clockUs < _torchOffUs! + 15000);
    final rel = opticalOn == null ? -1000.0 : (_clockUs - opticalOn) / 1000;
    final r = model.diameterAt(rel) / 2 * pxPerMm;
    final img = renderEye(
      size: 128,
      cx: 64,
      cy: 64,
      pupilA: r,
      pupilB: r * 0.96,
      irisR: 50,
      brightness: lit ? 1.15 : 1.0,
      seed: _index,
    );
    _frames.add(CameraLumaFrame(
      width: 128,
      height: 128,
      bytes: img.pixels,
      timestampUs: _clockUs,
      index: _index++,
      torchOn: lit,
    ));
  }

  @override
  Future<SensorLock> lock({double? iso, double? exposureDurationUs, double? focus}) async {
    calls.add('lock');
    locked = true;
    return const SensorLock(iso: 400, exposureDurationUs: 4000, focus: 0.8, fps: 120, manual: true);
  }

  @override
  Future<void> unlock() async {
    calls.add('unlock');
    locked = false;
  }

  @override
  Future<TorchPulse> flashPulse({int durationMs = 100, double level = 1.0}) async {
    calls.add('flash:$durationMs');
    expect(locked, isTrue, reason: 'flash must fire with the sensor locked');
    _torchOnUs = _clockUs;
    await Future<void>.delayed(Duration(milliseconds: durationMs));
    _torchOffUs = _torchOnUs! + durationMs * 1000;
    return TorchPulse(onUs: _torchOnUs!, offUs: _torchOffUs!);
  }

  @override
  Stream<CameraLumaFrame> frames() => _frames.stream;

  @override
  Future<void> dispose() async {
    _timer?.cancel();
    await _frames.close();
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('one-tap scan runs lock → baseline → 100 ms flash → response → analysis', () async {
    final cam = FakeCamera();
    final c = HardwareSyncController(
      camera: cam,
      config: const HardwareSyncConfig(roiSize: 128, responseMs: 2500, hvidMm: 11.7),
    );
    final phases = <ScanPhase>[];
    c.addListener(() {
      if (phases.isEmpty || phases.last != c.phase) phases.add(c.phase);
    });

    await c.initialize();
    expect(c.phase, anyOf(ScanPhase.searching, ScanPhase.aligned), reason: c.error);
    await c.startScan();

    expect(c.error, isNull);
    expect(c.phase, ScanPhase.complete);
    expect(cam.calls, ['initialize', 'lock', 'flash:100', 'unlock']);
    expect(
      phases.where((p) => p.index >= ScanPhase.lockingSensor.index).toList(),
      [
        ScanPhase.lockingSensor,
        ScanPhase.settling,
        ScanPhase.baseline,
        ScanPhase.stimulus,
        ScanPhase.response,
        ScanPhase.analyzing,
        ScanPhase.complete,
      ],
    );

    final r = c.result!;
    // Baseline window ≈ 500 ms of pre-flash samples.
    expect(r.curve.first.tMs, closeTo(-500, 40));
    expect(r.quality.stimulusFromOptics, isTrue);
    expect(r.quality.detectedFraction, greaterThan(0.95));
    expect(r.metrics.baselineDiameterMm, closeTo(5.0, 0.25));
    expect(r.metrics.latencyMs, closeTo(220, 40));
    expect(r.metrics.npiCategory, NpiCategory.brisk);
    expect(c.liveCurve.value, isNotEmpty);
    c.dispose();
  }, timeout: const Timeout(Duration(seconds: 60)));

  test('scan fails cleanly and unlocks if the flash errors', () async {
    final cam = _FailingFlashCamera();
    final c = HardwareSyncController(camera: cam, config: const HardwareSyncConfig(roiSize: 128));
    await c.initialize();
    await c.startScan();
    expect(c.error, contains('Flash failed'));
    expect(cam.calls.last, 'unlock');
    expect(c.phase, ScanPhase.searching);
    c.dispose();
  }, timeout: const Timeout(Duration(seconds: 30)));
}

class _FailingFlashCamera extends FakeCamera {
  @override
  Future<TorchPulse> flashPulse({int durationMs = 100, double level = 1.0}) async {
    calls.add('flash');
    throw PlatformException(code: 'torch', message: 'Torch busy');
  }
}
