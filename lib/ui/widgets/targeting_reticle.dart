import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

/// Neon-green targeting circle drawn over the live camera feed.
///
/// * Searching: thin dashed ring slowly rotating.
/// * Locked: solid thick ring with glow, corner brackets and a dot that
///   follows the Kalman-stabilised pupil centre.
///
/// Every neon stroke is drawn on top of a wider black halo so it keeps
/// ≥ 7:1 (WCAG AAA) contrast against whatever is behind it – including a
/// sunlit, overexposed camera image.
class TargetingReticle extends StatefulWidget {
  const TargetingReticle({
    super.key,
    required this.radius,
    required this.locked,
    this.pupilOffset,
    this.pupilRadius,
    this.label,
  });

  final double radius;
  final bool locked;

  /// Pupil centre relative to the reticle centre (logical px).
  final Offset? pupilOffset;
  final double? pupilRadius;
  final String? label;

  @override
  State<TargetingReticle> createState() => _TargetingReticleState();
}

class _TargetingReticleState extends State<TargetingReticle> with SingleTickerProviderStateMixin {
  late final AnimationController _spin =
      AnimationController(vsync: this, duration: const Duration(seconds: 6))..repeat();

  @override
  void dispose() {
    _spin.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: widget.locked ? 'Pupil locked' : 'Searching for pupil',
      child: AnimatedBuilder(
        animation: _spin,
        builder: (context, _) => TweenAnimationBuilder<double>(
          tween: Tween(end: widget.locked ? 1 : 0),
          duration: const Duration(milliseconds: 160),
          curve: Curves.easeOutCubic,
          builder: (context, lock, _) => CustomPaint(
            size: Size.square(widget.radius * 2 + 40),
            painter: _ReticlePainter(
              radius: widget.radius,
              lock: lock,
              phase: _spin.value,
              pupilOffset: widget.pupilOffset,
              pupilRadius: widget.pupilRadius,
              label: widget.label,
            ),
          ),
        ),
      ),
    );
  }
}

class _ReticlePainter extends CustomPainter {
  _ReticlePainter({
    required this.radius,
    required this.lock,
    required this.phase,
    this.pupilOffset,
    this.pupilRadius,
    this.label,
  });

  final double radius;

  /// 0 = searching, 1 = locked (animated).
  final double lock;
  final double phase;
  final Offset? pupilOffset;
  final double? pupilRadius;
  final String? label;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final neon = SynapseColors.neonGreen;
    final stroke = 1.5 + 2.0 * lock;

    void haloed(void Function(Paint p) draw, double width) {
      draw(Paint()
        ..color = SynapseColors.halo
        ..style = PaintingStyle.stroke
        ..strokeWidth = width + 3
        ..strokeCap = StrokeCap.round);
      draw(Paint()
        ..color = neon
        ..style = PaintingStyle.stroke
        ..strokeWidth = width
        ..strokeCap = StrokeCap.round);
    }

    // Lock glow.
    if (lock > 0) {
      canvas.drawCircle(
        c,
        radius,
        Paint()
          ..color = neon.withValues(alpha: 0.45 * lock)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 10
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 9),
      );
    }

    // Ring: 24 dashes that close into a solid ring as lock → 1.
    final rect = Rect.fromCircle(center: c, radius: radius);
    const segments = 24;
    final sweep = 2 * math.pi / segments;
    final fill = 0.45 + 0.55 * lock;
    final rot = (1 - lock) * phase * 2 * math.pi;
    haloed((p) {
      for (var i = 0; i < segments; i++) {
        canvas.drawArc(rect, rot + i * sweep, sweep * fill, false, p);
      }
    }, stroke);

    // Corner brackets snap in on lock.
    if (lock > 0.01) {
      final r = radius + 14 - 6 * lock;
      haloed((p) {
        for (var q = 0; q < 4; q++) {
          final a = math.pi / 4 + q * math.pi / 2;
          canvas.drawArc(Rect.fromCircle(center: c, radius: r), a - 0.18, 0.36, false, p);
        }
      }, 2.5);
    }

    // Cross-hair ticks.
    haloed((p) {
      for (var q = 0; q < 4; q++) {
        final a = q * math.pi / 2;
        final d = Offset(math.cos(a), math.sin(a));
        canvas.drawLine(c + d * (radius - 10), c + d * (radius + 6), p);
      }
    }, 2);

    // Pupil marker.
    if (pupilOffset != null && lock > 0.5) {
      final pc = c + pupilOffset!;
      final pr = (pupilRadius ?? 6).clamp(4.0, radius).toDouble();
      haloed((p) => canvas.drawCircle(pc, pr, p), 1.5);
      canvas.drawCircle(pc, 2.5, Paint()..color = neon);
    }

    if (label != null) {
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: SynapseType.label(color: neon).copyWith(
            shadows: const [Shadow(color: Colors.black, blurRadius: 4)],
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final at = Offset(c.dx - tp.width / 2, c.dy + radius + 12);
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(at.dx - 8, at.dy - 4, tp.width + 16, tp.height + 8),
          const Radius.circular(8),
        ),
        Paint()..color = SynapseColors.halo,
      );
      tp.paint(canvas, at);
    }
  }

  @override
  bool shouldRepaint(_ReticlePainter old) =>
      old.lock != lock ||
      old.phase != phase ||
      old.pupilOffset != pupilOffset ||
      old.radius != radius ||
      old.label != label;
}
