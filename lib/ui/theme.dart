import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Synapse palette. The background is a *matte* near-black: an emissive
/// white UI would act as an uncontrolled light stimulus and pre-constrict
/// the pupil (and wreck the operator's dark adaptation in a dim ICU bay).
abstract final class SynapseColors {
  static const background = Color(0xFF050505);
  static const backgroundRaised = Color(0xFF0B0B0C);

  static const textPrimary = Color(0xFFF5F5F7);

  /// 10.1:1 on [background] and 7.8:1 on the brightest glass fill – WCAG
  /// AAA for body text everywhere it is used.
  static const textSecondary = Color(0xFFB6B6BB);

  /// Targeting / current-scan neon. 15:1 on black; always drawn with a
  /// black halo so it also holds ≥ 7:1 against a sunlit camera image.
  static const neonGreen = Color(0xFF39FF14);
  static const halo = Color(0xE6000000);

  /// Reference overlays.
  static const healthyBlue = Color(0xFF64B5FF);
  static const abnormalRed = Color(0xFFFF6B61);
  static const amber = Color(0xFFFFC94D);

  static const glassFillTop = Color(0x1FFFFFFF);
  static const glassFillBottom = Color(0x08FFFFFF);
  static const glassStrokeTop = Color(0x66FFFFFF);
  static const glassStrokeBottom = Color(0x0FFFFFFF);
}

abstract final class SynapseType {
  /// Didone serif (Bodoni Moda) for headers – the authoritative voice.
  static TextStyle display(double size, {FontWeight weight = FontWeight.w500, Color? color}) =>
      TextStyle(
        fontFamily: 'BodoniModa',
        fontSize: size,
        fontWeight: weight,
        fontVariations: [
          FontVariation('wght', _wght(weight)),
          // Optical size tracks the rendered size for proper Didone contrast.
          FontVariation('opsz', size.clamp(6, 96).toDouble()),
        ],
        height: 1.05,
        letterSpacing: -0.4,
        color: color ?? SynapseColors.textPrimary,
      );

  /// Clean sans (Inter) with tabular figures so digits don't jitter while
  /// values update in real time.
  static TextStyle data(double size, {FontWeight weight = FontWeight.w500, Color? color}) =>
      TextStyle(
        fontFamily: 'Inter',
        fontSize: size,
        fontWeight: weight,
        fontVariations: [FontVariation('wght', _wght(weight))],
        fontFeatures: const [FontFeature.tabularFigures()],
        height: 1.1,
        letterSpacing: size > 28 ? -1.0 : 0,
        color: color ?? SynapseColors.textPrimary,
      );

  /// Small-caps style label above a value.
  static TextStyle label({Color? color}) => TextStyle(
        fontFamily: 'Inter',
        fontSize: 11.5,
        fontWeight: FontWeight.w600,
        fontVariations: const [FontVariation('wght', 600)],
        letterSpacing: 1.4,
        color: color ?? SynapseColors.textSecondary,
      );

  static double _wght(FontWeight w) => w.value.toDouble();
}

ThemeData buildSynapseTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: SynapseColors.background,
    fontFamily: 'Inter',
    colorScheme: const ColorScheme.dark(
      surface: SynapseColors.background,
      primary: SynapseColors.neonGreen,
      secondary: SynapseColors.healthyBlue,
      error: SynapseColors.abnormalRed,
      onPrimary: Colors.black,
      onSurface: SynapseColors.textPrimary,
    ),
    splashFactory: NoSplash.splashFactory,
  );
  return base.copyWith(
    textTheme: base.textTheme.apply(
      bodyColor: SynapseColors.textPrimary,
      displayColor: SynapseColors.textPrimary,
    ),
  );
}

/// WCAG 2.x relative luminance and contrast ratio.
abstract final class Wcag {
  static double _channel(double c) =>
      c <= 0.04045 ? c / 12.92 : math.pow((c + 0.055) / 1.055, 2.4).toDouble();

  static double luminance(Color c) =>
      0.2126 * _channel(c.r) + 0.7152 * _channel(c.g) + 0.0722 * _channel(c.b);

  /// Contrast of [fg] over [bg]; a translucent [fg] is composited first.
  static double contrast(Color fg, Color bg) {
    final f = Color.alphaBlend(fg, bg);
    final l1 = luminance(f), l2 = luminance(bg);
    return (math.max(l1, l2) + 0.05) / (math.min(l1, l2) + 0.05);
  }

  static bool passesAAA(Color fg, Color bg, {bool largeText = false}) =>
      contrast(fg, bg) >= (largeText ? 4.5 : 7.0);
}
