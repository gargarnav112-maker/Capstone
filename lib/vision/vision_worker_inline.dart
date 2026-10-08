import 'dart:async';
import 'dart:typed_data';

import 'grayscale.dart';
import 'pixel_scale.dart';
import 'pupil_tracker.dart';
import 'vision_worker.dart';

Future<VisionWorker> createVisionWorker({required double hvidMm, double? fixedMmPerPx}) async =>
    InlineVisionWorker(
      PupilTracker(
        scale: fixedMmPerPx != null ? FixedScale(fixedMmPerPx) : IrisReferenceScale(hvidMm: hvidMm),
      ),
    );

/// Same contract as the isolate worker, processed on the calling thread.
/// Results are delivered asynchronously so callers see identical ordering.
class InlineVisionWorker implements VisionWorker {
  InlineVisionWorker(this._tracker);

  final PupilTracker _tracker;
  final _out = StreamController<PupilObservation>.broadcast();

  @override
  Stream<PupilObservation> get observations => _out.stream;

  @override
  void submit(Uint8List luma, int width, int height, int timestampUs, int frameIndex) {
    final obs = _tracker.process(
      GrayImage(width, height, luma),
      timestampUs: timestampUs,
      frameIndex: frameIndex,
    );
    scheduleMicrotask(() {
      if (!_out.isClosed) _out.add(obs);
    });
  }

  @override
  void reset() => _tracker.reset();

  @override
  void dispose() => _out.close();
}
