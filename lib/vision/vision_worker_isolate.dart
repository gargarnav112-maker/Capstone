import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'grayscale.dart';
import 'pixel_scale.dart';
import 'pupil_tracker.dart';
import 'vision_worker.dart';

/// Runs the [PupilTracker] on a dedicated background isolate so 120 fps
/// processing never competes with UI rasterisation. Frame bytes are moved
/// with [TransferableTypedData] (zero-copy hand-off).
///
/// Everything stays in this process's memory; frames are dropped as soon as
/// they are processed and are never written to disk.
Future<VisionWorker> createVisionWorker({required double hvidMm, double? fixedMmPerPx}) =>
    IsolateVisionWorker.spawn(hvidMm: hvidMm, fixedMmPerPx: fixedMmPerPx);

class IsolateVisionWorker implements VisionWorker {
  IsolateVisionWorker._(this._isolate, this._toWorker, this._results);

  final Isolate _isolate;
  final SendPort _toWorker;
  final Stream<PupilObservation> _results;

  @override
  Stream<PupilObservation> get observations => _results;

  static Future<IsolateVisionWorker> spawn({double hvidMm = 11.7, double? fixedMmPerPx}) async {
    final fromWorker = ReceivePort();
    final isolate = await Isolate.spawn(
      _main,
      (fromWorker.sendPort, hvidMm, fixedMmPerPx),
      debugName: 'synapse-vision',
    );
    final broadcast = fromWorker.asBroadcastStream();
    final toWorker = await broadcast.first as SendPort;
    final results = broadcast
        .where((m) => m is Map)
        .map((m) => PupilObservation.fromMessage(m as Map<Object?, Object?>));
    return IsolateVisionWorker._(isolate, toWorker, results);
  }

  @override
  void submit(Uint8List luma, int width, int height, int timestampUs, int frameIndex) {
    _toWorker.send(<Object>[
      TransferableTypedData.fromList([luma]),
      width,
      height,
      timestampUs,
      frameIndex,
    ]);
  }

  @override
  void reset() => _toWorker.send('reset');

  @override
  void dispose() {
    _toWorker.send('stop');
    _isolate.kill(priority: Isolate.beforeNextEvent);
  }

  static void _main((SendPort, double, double?) args) {
    final (out, hvid, fixed) = args;
    final inbox = ReceivePort();
    out.send(inbox.sendPort);
    final tracker = PupilTracker(
      scale: fixed != null ? FixedScale(fixed) : IrisReferenceScale(hvidMm: hvid),
    );
    inbox.listen((msg) {
      if (msg == 'reset') {
        tracker.reset();
        return;
      }
      if (msg == 'stop') {
        inbox.close();
        return;
      }
      final m = msg as List<Object?>;
      final bytes = (m[0]! as TransferableTypedData).materialize().asUint8List();
      final obs = tracker.process(
        GrayImage(m[1]! as int, m[2]! as int, bytes),
        timestampUs: m[3]! as int,
        frameIndex: m[4]! as int,
      );
      out.send(obs.toMessage());
    });
  }
}
