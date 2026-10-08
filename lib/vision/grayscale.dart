import 'dart:typed_data';

/// 8-bit single-channel image, row-major, stride == [width].
class GrayImage {
  GrayImage(this.width, this.height, this.pixels)
      : assert(pixels.length >= width * height);

  final int width;
  final int height;
  final Uint8List pixels;

  int at(int x, int y) => pixels[y * width + x];

  /// Bilinear sample; coordinates are clamped to the image.
  double sample(double x, double y) {
    final cx = x.clamp(0.0, width - 1.0);
    final cy = y.clamp(0.0, height - 1.0);
    final x0 = cx.floor();
    final y0 = cy.floor();
    final x1 = x0 + 1 < width ? x0 + 1 : x0;
    final y1 = y0 + 1 < height ? y0 + 1 : y0;
    final fx = cx - x0;
    final fy = cy - y0;
    final top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
    final bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
    return top * (1 - fy) + bottom * fy;
  }
}

/// Pipeline step 2: grayscale conversion.
///
/// The native camera already streams the Y plane of full-range YUV, which *is*
/// BT.601 luma, so on-device frames arrive grayscale. This converter exists for
/// RGBA/BGRA sources (simulator, recorded test fixtures) and uses the same
/// BT.601 weights in fixed-point (77·R + 150·G + 29·B) >> 8.
GrayImage grayscaleFromRgba(
  Uint8List rgba,
  int width,
  int height, {
  bool bgra = false,
}) {
  final out = Uint8List(width * height);
  final rIdx = bgra ? 2 : 0;
  final bIdx = bgra ? 0 : 2;
  for (var i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (77 * rgba[p + rIdx] + 150 * rgba[p + 1] + 29 * rgba[p + bIdx]) >> 8;
  }
  return GrayImage(width, height, out);
}
