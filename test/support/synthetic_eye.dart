import 'dart:math' as math;
import 'dart:typed_data';

import 'package:synapse/vision/grayscale.dart';

/// Renders a synthetic eye ROI: skin, sclera, a textured iris, an elliptical
/// pupil and a torch glint, with sensor noise. Units are pixels.
GrayImage renderEye({
  int size = 256,
  double cx = 128,
  double cy = 128,
  double pupilA = 30,
  double pupilB = 28,
  double theta = 0.3,
  double irisR = 88,
  int skin = 150,
  int sclera = 215,
  int iris = 95,
  int pupil = 18,
  double noise = 4,
  bool glint = true,
  int seed = 1,
  double brightness = 1.0,
}) {
  final rnd = math.Random(seed);
  final px = Uint8List(size * size);
  final c = math.cos(theta), s = math.sin(theta);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final dx = x - cx, dy = y - cy;
      final r = math.sqrt(dx * dx + dy * dy);
      double v;
      // Almond-shaped palpebral fissure: skin above/below.
      final lid = (dy.abs() > 0.9 * irisR * math.sqrt(math.max(0, 1 - math.pow(dx / (2.2 * irisR), 2))));
      if (lid) {
        v = skin.toDouble();
      } else if (r > irisR) {
        v = sclera.toDouble();
      } else {
        final u = (dx * c + dy * s) / pupilA;
        final w = (-dx * s + dy * c) / pupilB;
        if (u * u + w * w <= 1) {
          v = pupil.toDouble();
        } else {
          // Radial iris striations.
          final ang = math.atan2(dy, dx);
          v = iris + 10 * math.sin(ang * 23) * math.cos(r / 5);
        }
      }
      if (glint && (x - (cx + 0.45 * pupilA)) * (x - (cx + 0.45 * pupilA)) + (y - (cy - 0.4 * pupilB)) * (y - (cy - 0.4 * pupilB)) < 16) {
        v = 250;
      }
      v = v * brightness + noise * (rnd.nextDouble() + rnd.nextDouble() + rnd.nextDouble() - 1.5);
      px[y * size + x] = v.round().clamp(0, 255);
    }
  }
  return GrayImage(size, size, px);
}
