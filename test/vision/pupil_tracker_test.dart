import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/vision/grayscale.dart';
import 'package:synapse/vision/pixel_scale.dart';
import 'package:synapse/vision/pupil_segmenter.dart';
import 'package:synapse/vision/pupil_tracker.dart';

import '../support/synthetic_eye.dart';

void main() {
  test('grayscale uses BT.601 luma weights', () {
    final rgba = Uint8List.fromList([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
    final g = grayscaleFromRgba(rgba, 4, 1);
    expect(g.pixels, [76, 149, 28, 255]);
    final b = grayscaleFromRgba(rgba, 4, 1, bgra: true);
    expect(b.pixels[0], 28); // red channel read as blue
  });

  test('segmenter isolates the pupil despite the corneal glint', () {
    final img = renderEye();
    final blob = PupilSegmenter().segment(img)!;
    expect(blob.centroidX, closeTo(128, 2));
    expect(blob.centroidY, closeTo(128, 2));
    // π·30·28 ≈ 2639 px; holes are filled for the contour, not the area.
    expect(blob.area, closeTo(2639, 200));
  });

  test('tracker fits the pupil to sub-pixel accuracy', () {
    final tracker = PupilTracker(scale: const FixedScale(0.1));
    final obs = tracker.process(renderEye(seed: 3), timestampUs: 0, frameIndex: 0);
    expect(obs.detected, isTrue);
    expect(obs.ellipse!.cx, closeTo(128, 0.6));
    expect(obs.ellipse!.cy, closeTo(128, 0.6));
    expect(obs.ellipse!.majorDiameter, closeTo(60, 1.2));
    expect(obs.ellipse!.minorDiameter, closeTo(56, 1.2));
    expect(obs.diameterMm, closeTo(6.0, 0.12));
  });

  test('iris-referenced scale converts px to mm independent of distance', () {
    // Same eye at two working distances (scale 1.0 and 0.8).
    double measure(double k, int seed) {
      final tracker = PupilTracker(scale: const IrisReferenceScale(hvidMm: 11.7));
      final img = renderEye(pupilA: 30 * k, pupilB: 30 * k, irisR: 88 * k, seed: seed);
      return tracker.process(img, timestampUs: 0, frameIndex: 0).diameterMm!;
    }

    final near = measure(1.0, 4);
    final far = measure(0.8, 5);
    // True pupil = 11.7 × 30/88 ≈ 3.99 mm at both distances.
    expect(near, closeTo(3.99, 0.15));
    expect(far, closeTo(3.99, 0.2));
  });

  test('no pupil → no detection', () {
    final img = GrayImage(128, 128, Uint8List(128 * 128)..fillRange(0, 128 * 128, 180));
    final obs = PupilTracker().process(img, timestampUs: 0, frameIndex: 0);
    expect(obs.detected, isFalse);
  });

  test('pupil remains detectable under the flash (locked exposure)', () {
    final obs = PupilTracker(scale: const FixedScale(0.1))
        .process(renderEye(brightness: 1.15, pupil: 30, seed: 9), timestampUs: 0, frameIndex: 0);
    expect(obs.detected, isTrue);
    expect(obs.ellipse!.majorDiameter, closeTo(60, 1.5));
  });

  test('observation survives isolate message round-trip', () {
    final obs = PupilTracker(scale: const FixedScale(0.1))
        .process(renderEye(seed: 2), timestampUs: 123456, frameIndex: 9);
    final back = PupilObservation.fromMessage(obs.toMessage());
    expect(back.timestampUs, 123456);
    expect(back.ellipse!.a, obs.ellipse!.a);
    expect(back.diameterMm, obs.diameterMm);
  });
}
