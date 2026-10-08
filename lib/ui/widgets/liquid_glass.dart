import 'dart:ui';

import 'package:flutter/material.dart';

import '../theme.dart';

/// Apple-style "liquid glass" surface: a backdrop blur, a faint vertical
/// fill gradient, a specular rim that is brighter along the top edge, and a
/// two-layer soft shadow so the panel appears to float above the matte black.
class LiquidGlass extends StatelessWidget {
  const LiquidGlass({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(18),
    this.radius = 28,
    this.blur = 28,
    this.tint,
    this.elevation = 1,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  final double blur;

  /// Optional colour wash (e.g. status colour at very low alpha).
  final Color? tint;

  /// Scales the shadow depth (0 = flush, 1 = floating).
  final double elevation;

  @override
  Widget build(BuildContext context) {
    final r = BorderRadius.circular(radius);
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: r,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.6 * elevation),
            blurRadius: 48 * elevation,
            spreadRadius: -6,
            offset: Offset(0, 22 * elevation),
          ),
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.45 * elevation),
            blurRadius: 10 * elevation,
            offset: Offset(0, 3 * elevation),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: r,
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
          child: CustomPaint(
            painter: _GlassPainter(radius: radius, tint: tint),
            child: Padding(padding: padding, child: child),
          ),
        ),
      ),
    );
  }
}

class _GlassPainter extends CustomPainter {
  _GlassPainter({required this.radius, this.tint});

  final double radius;
  final Color? tint;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final rrect = RRect.fromRectAndRadius(rect, Radius.circular(radius));

    // Body.
    canvas.drawRRect(
      rrect,
      Paint()
        ..shader = LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Color.alphaBlend(tint ?? Colors.transparent, SynapseColors.glassFillTop),
            SynapseColors.glassFillBottom,
          ],
        ).createShader(rect),
    );

    // Caustic: soft light pooling in the upper-left, like refraction.
    canvas.drawRRect(
      rrect,
      Paint()
        ..shader = RadialGradient(
          center: const Alignment(-0.8, -1.1),
          radius: 1.1,
          colors: [Colors.white.withValues(alpha: 0.07), Colors.transparent],
        ).createShader(rect),
    );

    // Specular rim.
    canvas.drawRRect(
      rrect.deflate(0.5),
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1
        ..shader = const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [SynapseColors.glassStrokeTop, SynapseColors.glassStrokeBottom],
        ).createShader(rect),
    );

    // Hairline highlight hugging the top edge.
    final highlight = Path()
      ..moveTo(radius, 1.2)
      ..lineTo(size.width - radius, 1.2);
    canvas.drawPath(
      highlight,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1
        ..shader = LinearGradient(
          colors: [
            Colors.white.withValues(alpha: 0),
            Colors.white.withValues(alpha: 0.5),
            Colors.white.withValues(alpha: 0),
          ],
        ).createShader(Rect.fromLTWH(radius, 0, size.width - 2 * radius, 2)),
    );
  }

  @override
  bool shouldRepaint(_GlassPainter old) => old.radius != radius || old.tint != tint;
}

/// Matte black stage with two very dim chromatic glows. The glass panels
/// need *something* behind them to refract; the glows stay below ~6 % so
/// overall screen emission remains negligible.
class AmbientBackdrop extends StatelessWidget {
  const AmbientBackdrop({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        const ColoredBox(color: SynapseColors.background),
        Positioned.fill(
          child: CustomPaint(painter: _GlowPainter()),
        ),
        child,
      ],
    );
  }
}

class _GlowPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    void glow(Offset c, double r, Color color) {
      canvas.drawCircle(
        c,
        r,
        Paint()
          ..shader = RadialGradient(colors: [color, color.withValues(alpha: 0)])
              .createShader(Rect.fromCircle(center: c, radius: r)),
      );
    }

    glow(Offset(size.width * 0.15, size.height * 0.18), size.width * 0.9,
        SynapseColors.neonGreen.withValues(alpha: 0.055));
    glow(Offset(size.width * 0.95, size.height * 0.62), size.width * 0.95,
        SynapseColors.healthyBlue.withValues(alpha: 0.06));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
