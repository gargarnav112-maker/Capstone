import 'dart:typed_data';

import 'pupil_tracker.dart';
import 'vision_worker_isolate.dart'
    if (dart.library.js_interop) 'vision_worker_inline.dart' as impl;

/// Runs the [PupilTracker] off the UI path.
///
/// On iOS/Android this is a dedicated background isolate fed with zero-copy
/// [TransferableTypedData]. The web build has no isolates, so frames are
/// processed inline on the event loop.
///
/// Everything stays in this process's memory; frames are dropped as soon as
/// they are processed and are never written to disk.
abstract interface class VisionWorker {
  static Future<VisionWorker> spawn({double hvidMm = 11.7, double? fixedMmPerPx}) =>
      impl.createVisionWorker(hvidMm: hvidMm, fixedMmPerPx: fixedMmPerPx);

  Stream<PupilObservation> get observations;

  void submit(Uint8List luma, int width, int height, int timestampUs, int frameIndex);

  /// Clears tracking state (Kalman priors, iris scale) between scans.
  void reset();

  void dispose();
}
