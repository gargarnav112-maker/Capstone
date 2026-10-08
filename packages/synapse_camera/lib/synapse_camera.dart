/// Low-level bridge to the native, manually-controlled clinical camera.
///
/// The native side (AVFoundation on iOS, Camera2 on Android) owns the sensor
/// and the torch. It exposes:
///
///  * a Flutter [Texture] for the live preview,
///  * a stream of upright, center-cropped **luma (Y-plane) ROIs** – the
///    grayscale conversion is free because the sensor already delivers YUV,
///  * hardware-clock timestamps for every frame and for the torch edges, so
///    the stimulus and the response share one monotonic time base.
///
/// Nothing in this plugin touches the network or the filesystem.
library;

import 'dart:async';

import 'package:flutter/services.dart';

/// One upright grayscale region-of-interest delivered by the sensor.
class CameraLumaFrame {
  const CameraLumaFrame({
    required this.width,
    required this.height,
    required this.bytes,
    required this.timestampUs,
    required this.index,
    required this.torchOn,
  });

  factory CameraLumaFrame.fromMap(Map<Object?, Object?> m) => CameraLumaFrame(
        width: m['w']! as int,
        height: m['h']! as int,
        bytes: m['bytes']! as Uint8List,
        timestampUs: m['ts']! as int,
        index: m['idx']! as int,
        torchOn: (m['torch'] as bool?) ?? false,
      );

  final int width;
  final int height;

  /// Row-major 8-bit luma, stride == [width].
  final Uint8List bytes;

  /// Sensor exposure timestamp in microseconds on the native monotonic clock
  /// (mach host time on iOS, the Camera2 sensor timestamp base on Android).
  final int timestampUs;
  final int index;

  /// Whether the torch had been commanded on when this frame was delivered.
  /// Only a hint – the true optical onset is detected from luminance.
  final bool torchOn;
}

/// Result of [SynapseCamera.initialize].
class CameraSession {
  const CameraSession({
    required this.textureId,
    required this.previewWidth,
    required this.previewHeight,
    required this.fps,
    required this.quarterTurns,
    required this.supportsManualSensor,
  });

  factory CameraSession.fromMap(Map<Object?, Object?> m) => CameraSession(
        textureId: m['textureId']! as int,
        previewWidth: (m['previewWidth']! as num).toDouble(),
        previewHeight: (m['previewHeight']! as num).toDouble(),
        fps: (m['fps']! as num).toDouble(),
        quarterTurns: (m['quarterTurns'] as int?) ?? 0,
        supportsManualSensor: (m['manualSensor'] as bool?) ?? false,
      );

  final int textureId;
  final double previewWidth;
  final double previewHeight;

  /// Frame rate the sensor was actually configured for.
  final double fps;

  /// Clockwise quarter turns needed to display the preview texture upright.
  final int quarterTurns;

  /// True when ISO / exposure time / focus distance can be pinned to absolute
  /// values (always on iOS; Camera2 MANUAL_SENSOR capability on Android).
  final bool supportsManualSensor;
}

/// The sensor parameters that were frozen by [SynapseCamera.lock].
class SensorLock {
  const SensorLock({
    required this.iso,
    required this.exposureDurationUs,
    required this.focus,
    required this.fps,
    required this.manual,
  });

  factory SensorLock.fromMap(Map<Object?, Object?> m) => SensorLock(
        iso: (m['iso']! as num).toDouble(),
        exposureDurationUs: (m['exposureUs']! as num).toDouble(),
        focus: (m['focus']! as num).toDouble(),
        fps: (m['fps']! as num).toDouble(),
        manual: (m['manual'] as bool?) ?? false,
      );

  final double iso;
  final double exposureDurationUs;

  /// Lens position (iOS, 0..1) or focus distance in diopters (Android).
  final double focus;
  final double fps;

  /// True when values are pinned absolutely; false means AE/AF/AWB "lock"
  /// fallbacks were used (no MANUAL_SENSOR support).
  final bool manual;

  Map<String, Object> toJson() => {
        'iso': iso,
        'exposureUs': exposureDurationUs,
        'focus': focus,
        'fps': fps,
        'manual': manual,
      };
}

/// Hardware timestamps of a torch pulse, on the same clock as frame
/// timestamps.
class TorchPulse {
  const TorchPulse({required this.onUs, required this.offUs});

  factory TorchPulse.fromMap(Map<Object?, Object?> m) =>
      TorchPulse(onUs: m['onUs']! as int, offUs: m['offUs']! as int);

  final int onUs;
  final int offUs;

  int get commandedDurationUs => offUs - onUs;
}

class SynapseCamera {
  SynapseCamera({
    MethodChannel? methodChannel,
    EventChannel? eventChannel,
  })  : _methods = methodChannel ?? const MethodChannel('synapse/camera'),
        _events = eventChannel ?? const EventChannel('synapse/camera/frames');

  final MethodChannel _methods;
  final EventChannel _events;

  /// Opens the rear (torch-equipped) camera at [targetFps] and configures a
  /// [roiSize]×[roiSize] center crop for the luma stream.
  Future<CameraSession> initialize({
    int targetFps = 120,
    int minFps = 60,
    int roiSize = 256,
  }) async {
    final m = await _methods.invokeMapMethod<Object?, Object?>('initialize', {
      'targetFps': targetFps,
      'minFps': minFps,
      'roiSize': roiSize,
    });
    return CameraSession.fromMap(m!);
  }

  /// Freezes ISO, exposure duration, focus and white balance. Pass explicit
  /// values to pin them; omitted values are frozen at their current
  /// auto-converged state. Exposure is always clamped to ≤ one frame period.
  Future<SensorLock> lock({
    double? iso,
    double? exposureDurationUs,
    double? focus,
  }) async {
    final m = await _methods.invokeMapMethod<Object?, Object?>('lock', {
      'iso': ?iso,
      'exposureUs': ?exposureDurationUs,
      'focus': ?focus,
    });
    return SensorLock.fromMap(m!);
  }

  Future<void> unlock() => _methods.invokeMethod<void>('unlock');

  /// Fires the torch for [durationMs] using a native high-priority timer so
  /// the pulse width does not depend on the Dart event loop.
  Future<TorchPulse> flashPulse({int durationMs = 100, double level = 1.0}) async {
    final m = await _methods.invokeMapMethod<Object?, Object?>('flashPulse', {
      'durationMs': durationMs,
      'level': level,
    });
    return TorchPulse.fromMap(m!);
  }

  /// Current native monotonic time in microseconds (same base as frames).
  Future<int> nowUs() async => (await _methods.invokeMethod<int>('nowUs'))!;

  /// Live luma ROI stream. Listening starts frame delivery; cancelling stops it.
  Stream<CameraLumaFrame> frames() => _events
      .receiveBroadcastStream()
      .map((e) => CameraLumaFrame.fromMap(e as Map<Object?, Object?>));

  Future<void> dispose() => _methods.invokeMethod<void>('dispose');
}
