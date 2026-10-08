import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:synapse_camera/synapse_camera.dart';

import '../biomarkers/plr_analyzer.dart';
import '../biomarkers/plr_models.dart';
import '../vision/kalman.dart';
import '../vision/pupil_tracker.dart';
import '../vision/vision_worker.dart';

enum ScanPhase {
  idle('Ready'),
  initializing('Starting camera'),
  searching('Searching for pupil'),
  aligned('Pupil locked'),
  lockingSensor('Locking ISO · focus · exposure'),
  settling('Stabilising sensor'),
  baseline('Resting baseline'),
  stimulus('Flash'),
  response('Recording response'),
  analyzing('Analysing'),
  complete('Complete'),
  error('Error');

  const ScanPhase(this.label);

  final String label;

  bool get isRecording => this == baseline || this == stimulus || this == response;
  bool get isBusy => index >= lockingSensor.index && index <= analyzing.index;
}

class HardwareSyncConfig {
  const HardwareSyncConfig({
    this.targetFps = 120,
    this.minFps = 60,
    this.roiSize = 256,
    this.settleMs = 120,
    this.baselineMs = 500,
    this.flashMs = 100,
    this.responseMs = 3000,
    this.torchLevel = 1.0,
    this.lockFrames = 6,
    this.lockConfidence = 0.4,
    this.lockTimeout = const Duration(seconds: 4),
    this.hvidMm = 11.7,
  }) : assert(baselineMs >= 0 && baselineMs <= 500),
       assert(targetFps >= minFps && minFps >= 60);

  /// Requested sensor rate; 120 fps gives a new geometric sample every 8.3 ms,
  /// the 60 fps floor every 16.7 ms.
  final int targetFps;
  final int minFps;
  final int roiSize;

  /// Frames discarded after locking while the new sensor settings propagate
  /// through the ISP pipeline (typically 3–5 frames).
  final int settleMs;

  /// Step 2: resting baseline in ambient light (0–500 ms).
  final int baselineMs;

  /// Step 3: stimulus width.
  final int flashMs;

  /// Step 4: recording continues this long after the flash onset.
  final int responseMs;

  /// Torch drive level (0–1); fixed per device during calibration so every
  /// scan delivers the same retinal illuminance.
  final double torchLevel;

  /// Consecutive confident detections required for visual "lock".
  final int lockFrames;
  final double lockConfidence;
  final Duration lockTimeout;

  /// Horizontal visible iris diameter used as the in-frame ruler.
  final double hvidMm;
}

/// Orchestrates camera hardware, the stimulus timeline and the vision
/// pipeline, all on one hardware clock:
///
///   1. Lock the camera (ISO, exposure duration, focus, white balance).
///   2. Record a resting baseline (0–500 ms) in ambient light.
///   3. Fire a 100 ms torch pulse (native high-priority timer).
///   4. Keep recording the response until the end of the capture window.
///
/// The timeline is advanced by **frame timestamps**, not wall-clock timers,
/// so Dart event-loop jitter cannot shift the stimulus relative to the data.
/// Frames are processed in memory on a background isolate and discarded;
/// nothing is written to disk or sent over a network.
PlrAnalysis _runAnalysis((PlrAnalyzer, List<FrameSample>, int) job) {
  final (analyzer, frames, onUs) = job;
  return analyzer.analyze(frames, PlrAnalyzer.detectStimulus(frames, commandOnUs: onUs));
}

class HardwareSyncController extends ChangeNotifier {
  HardwareSyncController({
    SynapseCamera? camera,
    this.config = const HardwareSyncConfig(),
    this.analyzer = const PlrAnalyzer(),
  }) : _camera = camera ?? SynapseCamera();

  final SynapseCamera _camera;
  SynapseCamera get camera => _camera;
  final HardwareSyncConfig config;
  final PlrAnalyzer analyzer;

  VisionWorker? _worker;
  StreamSubscription<CameraLumaFrame>? _frameSub;
  StreamSubscription<PupilObservation>? _obsSub;

  ScanPhase _phase = ScanPhase.idle;
  ScanPhase get phase => _phase;

  CameraSession? _session;
  CameraSession? get session => _session;

  SensorLock? _sensorLock;
  SensorLock? get sensorLock => _sensorLock;

  String? _error;
  String? get error => _error;

  PupilObservation? _latest;
  PupilObservation? get latest => _latest;

  bool _pupilLocked = false;
  bool get pupilLocked => _pupilLocked;
  int _confidentStreak = 0;
  Completer<void>? _lockWaiter;

  Eye eye = Eye.right;

  ScanResult? _result;
  ScanResult? get result => _result;

  /// Live, causally-filtered curve for the real-time chart. t = 0 is the
  /// (planned, then measured) flash onset.
  final ValueNotifier<List<KinematicPoint>> liveCurve = ValueNotifier(const []);
  ConstantVelocityKalman? _liveFilter;
  final List<KinematicPoint> _livePoints = [];
  DateTime _lastUi = DateTime.fromMillisecondsSinceEpoch(0);

  // ---- Timeline state (all in native µs).
  int _inFlight = 0;
  int? _settleUntilUs;
  int? _recordStartUs;
  int? _plannedFlashUs;
  bool _flashRequested = false;
  TorchPulse? _pulse;
  int? _recordEndUs;
  final List<PupilObservation> _recorded = [];
  Completer<void>? _drained;
  DateTime _lastProgress = DateTime.now();

  // ------------------------------------------------------------------ setup

  Future<void> initialize() async {
    if (_phase != ScanPhase.idle && _phase != ScanPhase.error) return;
    _setPhase(ScanPhase.initializing);
    try {
      if (_camera.isHardware && !kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
        final status = await Permission.camera.request();
        if (!status.isGranted) throw StateError('Camera permission is required.');
      }
      _worker = await VisionWorker.spawn(hvidMm: config.hvidMm);
      _obsSub = _worker!.observations.listen(_onObservation);
      _session = await _camera.initialize(
        targetFps: config.targetFps,
        minFps: config.minFps,
        roiSize: config.roiSize,
      );
      _frameSub = _camera.frames().listen(_onFrame, onError: (Object e) => _fail('$e'));
      _setPhase(ScanPhase.searching);
    } catch (e) {
      _fail(_describe(e));
    }
  }

  // ------------------------------------------------------------- one tap

  /// Runs the full 4-step acquisition and analysis. Completes when the
  /// result (or an error) is available.
  Future<void> startScan() async {
    if (_phase != ScanPhase.searching && _phase != ScanPhase.aligned && _phase != ScanPhase.complete) {
      return;
    }
    _result = null;
    _error = null;
    try {
      if (!_pupilLocked) {
        _lockWaiter = Completer<void>();
        await _lockWaiter!.future.timeout(
          config.lockTimeout,
          onTimeout: () => throw StateError('Pupil not found – centre the eye in the ring.'),
        );
      }

      // Step 1 – freeze the sensor.
      _setPhase(ScanPhase.lockingSensor);
      _sensorLock = await _camera.lock();

      _resetTimeline();
      _worker!.reset();
      _setPhase(ScanPhase.settling);
      // Steps 2–4 are driven from _onFrame by frame timestamps.
      _drained = Completer<void>();
      // The timeline runs on frame timestamps, so a slow device simply takes
      // longer. Fail only if frames or results stop arriving altogether.
      _lastProgress = DateTime.now();
      final watchdog = Timer.periodic(const Duration(milliseconds: 500), (_) {
        if (DateTime.now().difference(_lastProgress) > const Duration(seconds: 3)) {
          _drained?.completeError(StateError('Camera stopped delivering frames.'));
          _drained = null;
        }
      });
      try {
        await _drained!.future;
      } finally {
        watchdog.cancel();
      }

      await _camera.unlock();
      _setPhase(ScanPhase.analyzing);
      _result = await _analyze();
      _setPhase(ScanPhase.complete);
    } catch (e) {
      try {
        await _camera.unlock();
      } catch (_) {}
      _fail(e is PlrAnalysisException ? 'Scan rejected: ${e.message}' : _describe(e));
      // Return to live view so the clinician can simply retry.
      _setPhase(ScanPhase.searching, keepError: true);
    }
  }

  /// Abort an in-progress scan and return to live preview.
  Future<void> cancel() async {
    _resetTimeline();
    _drained?.completeError(StateError('Scan cancelled'));
    _drained = null;
  }

  // -------------------------------------------------------- frame pipeline

  void _onFrame(CameraLumaFrame f) {
    final recording = _phase.isRecording || _phase == ScanPhase.settling;
    // Live preview: drop frames if the worker lags so the reticle stays
    // responsive. Recording: every frame is kept (the worker queues them).
    if (!recording && _inFlight > 2) return;
    _inFlight++;
    _worker?.submit(f.bytes, f.width, f.height, f.timestampUs, f.index);

    final ts = f.timestampUs;
    switch (_phase) {
      case ScanPhase.settling:
        _settleUntilUs ??= ts + config.settleMs * 1000;
        if (ts >= _settleUntilUs!) {
          _recordStartUs = ts;
          _plannedFlashUs = ts + config.baselineMs * 1000;
          _liveFilter = ConstantVelocityKalman(q: 10, r: 0.0025);
          _setPhase(ScanPhase.baseline);
          if (config.baselineMs == 0) _fireFlash();
        }
      case ScanPhase.baseline:
        if (ts >= _plannedFlashUs!) _fireFlash();
      case ScanPhase.response:
        if (_pulse != null && ts >= _pulse!.onUs + config.responseMs * 1000 && _recordEndUs == null) {
          _recordEndUs = ts;
        }
      default:
        break;
    }
  }

  void _fireFlash() {
    if (_flashRequested) return;
    _flashRequested = true;
    _setPhase(ScanPhase.stimulus);
    _camera.flashPulse(durationMs: config.flashMs, level: config.torchLevel).then((pulse) {
      _pulse = pulse;
      _plannedFlashUs = pulse.onUs;
      if (_phase == ScanPhase.stimulus) _setPhase(ScanPhase.response);
    }, onError: (Object e) {
      _drained?.completeError(StateError('Flash failed: $e'));
    });
  }

  void _onObservation(PupilObservation o) {
    _lastProgress = DateTime.now();
    final uiDue = _uiDue();
    _inFlight = math.max(0, _inFlight - 1);
    _latest = o;

    // Visual lock feedback (hysteresis: N good frames to lock, 3 bad to lose).
    if (o.detected && o.confidence >= config.lockConfidence) {
      _confidentStreak = math.max(1, _confidentStreak + 1);
    } else {
      _confidentStreak = math.min(0, _confidentStreak - 1);
    }
    final wasLocked = _pupilLocked;
    if (_confidentStreak >= config.lockFrames) _pupilLocked = true;
    if (_confidentStreak <= -3) _pupilLocked = false;
    if (_pupilLocked && !wasLocked) {
      _lockWaiter?.complete();
      _lockWaiter = null;
    }
    if (!_phase.isBusy) {
      final next = _pupilLocked ? ScanPhase.aligned : ScanPhase.searching;
      if (_phase == ScanPhase.searching || _phase == ScanPhase.aligned) {
        if (_phase != next) _setPhase(next, keepError: true);
      }
    }

    // Recording buffer, defined purely by timestamps.
    final start = _recordStartUs;
    if (start != null && o.timestampUs >= start) {
      final end = _recordEndUs;
      if (end == null || o.timestampUs <= end) {
        _recorded.add(o);
        _appendLive(o, publish: uiDue);
      }
      if (end != null && o.timestampUs >= end) {
        liveCurve.value = List.unmodifiable(_livePoints);
        _drained?.complete();
        _drained = null;
      }
    }
    if (_pupilLocked != wasLocked || uiDue) notifyListeners();
  }

  /// UI refresh is capped at ~30 Hz; data is still recorded at full rate.
  bool _uiDue() {
    final now = DateTime.now();
    if (now.difference(_lastUi) < const Duration(milliseconds: 33)) return false;
    _lastUi = now;
    return true;
  }

  void _appendLive(PupilObservation o, {required bool publish}) {
    final d = o.diameterMm;
    final filter = _liveFilter;
    if (filter == null || _plannedFlashUs == null) return;
    final t = o.timestampUs / 1e6;
    filter.update(t, d);
    if (!filter.initialized) return;
    final rel = (o.timestampUs - _plannedFlashUs!) / 1000;
    _livePoints.add(KinematicPoint(rel, filter.position));
    if (publish) liveCurve.value = List.unmodifiable(_livePoints);
  }

  // -------------------------------------------------------------- analysis

  Future<ScanResult> _analyze() async {
    final pulse = _pulse;
    if (pulse == null) throw StateError('Flash was not delivered.');
    final frames = [
      for (final o in _recorded)
        FrameSample(timestampUs: o.timestampUs, meanLuma: o.meanLuma, diameterMm: o.diameterMm),
    ];
    // Background isolate on device; inline on web.
    final analysis = await compute(_runAnalysis, (analyzer, frames, pulse.onUs));
    return ScanResult(
      scanId: _newScanId(),
      eye: eye,
      recordedAt: DateTime.now(),
      metrics: analysis.metrics,
      curve: analysis.curve,
      rawCurve: analysis.rawCurve,
      quality: analysis.quality,
      sensor: {
        ...?_sensorLock?.toJson(),
        'configuredFps': _session?.fps ?? 0,
        'torchLevel': config.torchLevel,
        'commandedFlashMs': pulse.commandedDurationUs / 1000,
      },
    );
  }

  static String _describe(Object e) => switch (e) {
        PlatformException(:final message, :final code) => message ?? code,
        StateError(:final message) => message,
        _ => '$e',
      };

  static String _newScanId() {
    final r = math.Random.secure();
    String hex(int n) => List.generate(n, (_) => r.nextInt(16).toRadixString(16)).join();
    // RFC 4122 v4 layout so it can double as a FHIR fullUrl uuid.
    return '${hex(8)}-${hex(4)}-4${hex(3)}-${(8 + r.nextInt(4)).toRadixString(16)}${hex(3)}-${hex(12)}';
  }

  // ----------------------------------------------------------------- utils

  void _resetTimeline() {
    _settleUntilUs = null;
    _recordStartUs = null;
    _plannedFlashUs = null;
    _flashRequested = false;
    _pulse = null;
    _recordEndUs = null;
    _recorded.clear();
    liveCurve.value = const [];
    _livePoints.clear();
    _liveFilter = null;
  }

  void _setPhase(ScanPhase p, {bool keepError = false}) {
    _phase = p;
    if (!keepError && p != ScanPhase.error) _error = null;
    notifyListeners();
  }

  void _fail(String message) {
    _error = message;
    _phase = ScanPhase.error;
    notifyListeners();
  }

  @override
  void dispose() {
    _frameSub?.cancel();
    _obsSub?.cancel();
    _worker?.dispose();
    _camera.dispose();
    liveCurve.dispose();
    super.dispose();
  }
}
