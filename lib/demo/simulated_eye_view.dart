import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'simulated_camera.dart';

/// Draws the simulator's scene in place of the camera texture. Geometry is
/// identical to the frames fed to the vision pipeline, so the reticle and
/// pupil marker line up exactly.
class SimulatedEyeView extends StatelessWidget {
  const SimulatedEyeView({super.key, required this.camera});

  final SimulatedCamera camera;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<SimulatedScene?>(
      valueListenable: camera.scene,
      builder: (_, scene, _) => CustomPaint(painter: _EyePainter(scene)),
    );
  }
}

class _EyePainter extends CustomPainter {
  _EyePainter(this.scene);

  final SimulatedScene? scene;

  @override
  void paint(Canvas canvas, Size size) {
    final frame = SimulatedCamera.frameSize.toDouble();
    final s = math.max(size.width, size.height) / frame;
    final origin = Offset((size.width - frame * s) / 2, (size.height - frame * s) / 2);
    final sc = scene;
    final lit = sc?.torchOn ?? false;
    final c = origin + Offset(frame / 2 + (sc?.offsetX ?? 0), frame / 2 + (sc?.offsetY ?? 0)) * s;
    final irisR = (sc?.irisRadiusPx ?? SimulatedCamera.irisRadiusPx) * s;
    final pupilR = (sc?.pupilRadiusPx ?? 0) * s;
    double g(double v) => (v * (lit ? 1.18 : 1)).clamp(0, 255);
    Color grey(double v, {int r = 0, int gr = 0, int b = 0}) =>
        Color.fromARGB(255, (g(v) + r).clamp(0, 255).round(), (g(v) + gr).clamp(0, 255).round(),
            (g(v) + b).clamp(0, 255).round());

    // Skin.
    canvas.drawRect(Offset.zero & size, Paint()..color = grey(150, r: 22, gr: 4, b: -10));

    // Palpebral fissure (almond) – same curve as the rendered frames.
    final eye = Path();
    const steps = 64;
    for (var i = 0; i <= steps; i++) {
      final dx = -2.2 * irisR + 4.4 * irisR * i / steps;
      final h = 0.9 * irisR * math.sqrt(math.max(0, 1 - dx * dx / (4.84 * irisR * irisR)));
      final p = c + Offset(dx, -h);
      i == 0 ? eye.moveTo(p.dx, p.dy) : eye.lineTo(p.dx, p.dy);
    }
    for (var i = steps; i >= 0; i--) {
      final dx = -2.2 * irisR + 4.4 * irisR * i / steps;
      final h = 0.9 * irisR * math.sqrt(math.max(0, 1 - dx * dx / (4.84 * irisR * irisR)));
      eye.lineTo(c.dx + dx, c.dy + h);
    }
    eye.close();
    canvas.save();
    canvas.clipPath(eye);
    canvas.drawRect(Offset.zero & size, Paint()..color = grey(212, b: 4));

    // Iris with radial striations.
    canvas.drawCircle(
      c,
      irisR,
      Paint()
        ..shader = RadialGradient(colors: [grey(78, r: 18, gr: 6), grey(98, r: 20, gr: 8), grey(70, r: 10)],
                stops: const [0.3, 0.8, 1])
            .createShader(Rect.fromCircle(center: c, radius: irisR)),
    );
    final fiber = Paint()
      ..color = Colors.black.withValues(alpha: 0.12)
      ..strokeWidth = 1.2 * s;
    for (var i = 0; i < 46; i++) {
      final a = i * 2 * math.pi / 46;
      final d = Offset(math.cos(a), math.sin(a));
      canvas.drawLine(c + d * (pupilR + 2 * s), c + d * irisR * 0.95, fiber);
    }

    // Pupil + corneal glint.
    canvas.drawCircle(c, pupilR, Paint()..color = grey(16));
    canvas.drawCircle(c + Offset(0.4 * pupilR, -0.4 * pupilR), 3 * s, Paint()..color = Colors.white);
    canvas.restore();

    if (lit) {
      canvas.drawRect(Offset.zero & size, Paint()..color = Colors.white.withValues(alpha: 0.12));
    }
  }

  @override
  bool shouldRepaint(_EyePainter old) => !identical(old.scene, scene);
}
