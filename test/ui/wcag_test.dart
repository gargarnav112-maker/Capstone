import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/ui/theme.dart';

void main() {
  test('contrast formula matches WCAG reference values', () {
    expect(Wcag.contrast(Colors.white, Colors.black), closeTo(21, 0.01));
    expect(Wcag.contrast(const Color(0xFF777777), Colors.white), closeTo(4.48, 0.02));
  });

  group('AAA (≥ 7:1)', () {
    const bg = SynapseColors.background;
    // Worst case glass: top fill composited over the background.
    final glass = Color.alphaBlend(SynapseColors.glassFillTop, bg);

    for (final (name, c) in [
      ('primary text', SynapseColors.textPrimary),
      ('secondary text', SynapseColors.textSecondary),
      ('neon green', SynapseColors.neonGreen),
    ]) {
      test('$name on matte black', () => expect(Wcag.contrast(c, bg), greaterThanOrEqualTo(7)));
      test('$name on glass', () => expect(Wcag.contrast(c, glass), greaterThanOrEqualTo(7)));
    }

    test('neon reticle vs its halo holds AAA even over a sunlit (white) image', () {
      final haloOverWhite = Color.alphaBlend(SynapseColors.halo, Colors.white);
      expect(Wcag.contrast(SynapseColors.neonGreen, haloOverWhite), greaterThanOrEqualTo(7));
    });

    test('reference curve colours are distinguishable on black (≥ 4.5)', () {
      expect(Wcag.contrast(SynapseColors.healthyBlue, bg), greaterThanOrEqualTo(4.5));
      expect(Wcag.contrast(SynapseColors.abnormalRed, bg), greaterThanOrEqualTo(4.5));
    });
  });
}
