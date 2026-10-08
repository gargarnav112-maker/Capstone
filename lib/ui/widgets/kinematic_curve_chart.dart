import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../../biomarkers/plr_models.dart';
import '../../biomarkers/reference_curves.dart';
import '../theme.dart';

/// Pupil diameter (mm) vs time (ms) relative to the flash.
///
/// The current scan (neon green) is overlaid on a healthy (blue) and an
/// abnormal (red) reference. References are rescaled to the patient's own
/// resting diameter so the comparison is of reflex *shape* (latency, depth,
/// speed), not of absolute pupil size, which varies with age and ambient
/// light.
class KinematicCurveChart extends StatelessWidget {
  const KinematicCurveChart({
    super.key,
    required this.scan,
    this.baselineMm,
    this.healthy = PlrModel.healthy,
    this.abnormal = PlrModel.abnormal,
    this.fromMs = -500,
    this.toMs = 3000,
    this.flashMs = 100,
    this.showLegend = true,
    this.height = 220,
  });

  final List<KinematicPoint> scan;

  /// Patient resting diameter used to normalise the references. Defaults to
  /// the mean of the scan's pre-flash points.
  final double? baselineMm;
  final PlrModel healthy;
  final PlrModel abnormal;
  final double fromMs;
  final double toMs;
  final double flashMs;
  final bool showLegend;
  final double height;

  @override
  Widget build(BuildContext context) {
    final pre = scan.where((p) => p.tMs < 0).map((p) => p.diameterMm).toList();
    final base = baselineMm ??
        (pre.isEmpty ? healthy.baselineMm : pre.reduce((a, b) => a + b) / pre.length);
    List<Offset> ref(PlrModel m) => [
          for (final p in m.sample(fromMs: fromMs, toMs: toMs, stepMs: 10))
            Offset(p.tMs, p.diameterMm * base / m.baselineMm),
        ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          height: height,
          child: CustomPaint(
            size: Size.infinite,
            painter: _CurvePainter(
              scan: [for (final p in scan) Offset(p.tMs, p.diameterMm)],
              healthy: ref(healthy),
              abnormal: ref(abnormal),
              fromMs: fromMs,
              toMs: toMs,
              flashMs: flashMs,
            ),
          ),
        ),
        if (showLegend) ...[
          const SizedBox(height: 12),
          const Wrap(
            spacing: 18,
            runSpacing: 8,
            children: [
              _LegendSwatch(color: SynapseColors.neonGreen, label: 'This scan'),
              _LegendSwatch(color: SynapseColors.healthyBlue, label: 'Healthy reference', dashed: true),
              _LegendSwatch(color: SynapseColors.abnormalRed, label: 'Abnormal reference', dashed: true),
            ],
          ),
        ],
      ],
    );
  }
}

class _LegendSwatch extends StatelessWidget {
  const _LegendSwatch({required this.color, required this.label, this.dashed = false});

  final Color color;
  final String label;
  final bool dashed;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        CustomPaint(
          size: const Size(18, 8),
          painter: _SwatchPainter(color, dashed),
        ),
        const SizedBox(width: 6),
        Text(label, style: SynapseType.data(12, color: SynapseColors.textSecondary)),
      ],
    );
  }
}

class _SwatchPainter extends CustomPainter {
  _SwatchPainter(this.color, this.dashed);

  final Color color;
  final bool dashed;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    final y = size.height / 2;
    if (dashed) {
      canvas.drawLine(Offset(0, y), Offset(6, y), paint);
      canvas.drawLine(Offset(11, y), Offset(size.width, y), paint);
    } else {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(_SwatchPainter old) => old.color != color || old.dashed != dashed;
}

class _CurvePainter extends CustomPainter {
  _CurvePainter({
    required this.scan,
    required this.healthy,
    required this.abnormal,
    required this.fromMs,
    required this.toMs,
    required this.flashMs,
  });

  final List<Offset> scan;
  final List<Offset> healthy;
  final List<Offset> abnormal;
  final double fromMs;
  final double toMs;
  final double flashMs;

  static const _left = 34.0;
  static const _bottom = 22.0;
  static const _top = 6.0;
  static const _right = 6.0;

  @override
  void paint(Canvas canvas, Size size) {
    final all = [...scan, ...healthy, ...abnormal].map((o) => o.dy);
    var lo = all.isEmpty ? 2.0 : all.reduce(math.min);
    var hi = all.isEmpty ? 6.0 : all.reduce(math.max);
    final pad = math.max(0.15, (hi - lo) * 0.12);
    lo = ((lo - pad) * 2).floorToDouble() / 2;
    hi = ((hi + pad) * 2).ceilToDouble() / 2;

    final plot = Rect.fromLTRB(_left, _top, size.width - _right, size.height - _bottom);
    Offset map(Offset p) => Offset(
          plot.left + (p.dx - fromMs) / (toMs - fromMs) * plot.width,
          plot.bottom - (p.dy - lo) / (hi - lo) * plot.height,
        );

    // Flash window.
    final f0 = map(Offset(0, hi)).dx, f1 = map(Offset(flashMs, hi)).dx;
    canvas.drawRect(
      Rect.fromLTRB(f0, plot.top, f1, plot.bottom),
      Paint()..color = Colors.white.withValues(alpha: 0.07),
    );
    _text(canvas, 'FLASH', Offset(f1 + 4, plot.top + 2), SynapseColors.textSecondary, 9.5);

    // Grid + axis labels.
    final grid = Paint()
      ..color = Colors.white.withValues(alpha: 0.07)
      ..strokeWidth = 1;
    final yStep = (hi - lo) > 3 ? 1.0 : 0.5;
    for (var v = lo; v <= hi + 1e-9; v += yStep) {
      final y = map(Offset(fromMs, v)).dy;
      canvas.drawLine(Offset(plot.left, y), Offset(plot.right, y), grid);
      _text(canvas, v.toStringAsFixed(1), Offset(0, y - 7), SynapseColors.textSecondary, 10);
    }
    for (var t = 0.0; t <= toMs; t += 500) {
      final x = map(Offset(t, lo)).dx;
      canvas.drawLine(Offset(x, plot.top), Offset(x, plot.bottom), grid);
      _text(canvas, t == 0 ? '0' : '${(t / 1000).toStringAsFixed(1)}s', Offset(x - 8, plot.bottom + 6),
          SynapseColors.textSecondary, 10);
    }

    canvas.save();
    canvas.clipRect(plot);
    _series(canvas, abnormal.map(map).toList(), SynapseColors.abnormalRed, 1.8, dashed: true);
    _series(canvas, healthy.map(map).toList(), SynapseColors.healthyBlue, 1.8, dashed: true);
    final pts = scan.map(map).toList();
    // Glow underlay, then the crisp line.
    _series(canvas, pts, SynapseColors.neonGreen.withValues(alpha: 0.35), 7,
        blur: const ui.MaskFilter.blur(BlurStyle.normal, 5));
    _series(canvas, pts, SynapseColors.neonGreen, 2.6);
    if (pts.isNotEmpty) {
      canvas.drawCircle(pts.last, 4, Paint()..color = SynapseColors.neonGreen);
    }
    canvas.restore();
  }

  void _series(Canvas canvas, List<Offset> pts, Color color, double width,
      {bool dashed = false, ui.MaskFilter? blur}) {
    if (pts.length < 2) return;
    final path = Path()..moveTo(pts.first.dx, pts.first.dy);
    for (final p in pts.skip(1)) {
      path.lineTo(p.dx, p.dy);
    }
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = width
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..maskFilter = blur;
    if (!dashed) {
      canvas.drawPath(path, paint);
      return;
    }
    for (final metric in path.computeMetrics()) {
      for (var d = 0.0; d < metric.length; d += 9) {
        canvas.drawPath(metric.extractPath(d, math.min(d + 5, metric.length)), paint);
      }
    }
  }

  void _text(Canvas canvas, String s, Offset at, Color color, double size) {
    final tp = TextPainter(
      text: TextSpan(text: s, style: SynapseType.data(size, color: color)),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, at);
  }

  @override
  bool shouldRepaint(_CurvePainter old) =>
      !identical(old.scan, scan) || old.scan.length != scan.length || old.fromMs != fromMs;
}
