import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../biomarkers/plr_models.dart';
import '../demo/demo_scan.dart';
import '../demo/simulated_camera.dart';
import '../demo/simulated_eye_view.dart';
import '../hardware/hardware_sync_controller.dart';
import 'liquid_results_dashboard.dart';
import 'theme.dart';
import 'widgets/kinematic_curve_chart.dart';
import 'widgets/liquid_glass.dart';
import 'widgets/targeting_reticle.dart';

/// One-tap scanner: live feed, neon targeting ring with lock feedback, eye
/// selector, real-time kinematic curve while recording.
class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key, this.controller});

  /// Injected for tests; otherwise created and owned by the screen.
  final HardwareSyncController? controller;

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  late final HardwareSyncController _c = widget.controller ?? HardwareSyncController();
  bool _wasLocked = false;

  @override
  void initState() {
    super.initState();
    _c.addListener(_onChange);
    _c.initialize();
  }

  @override
  void dispose() {
    _c.removeListener(_onChange);
    if (widget.controller == null) _c.dispose();
    super.dispose();
  }

  void _onChange() {
    // Instant tactile confirmation of pupil lock.
    if (_c.pupilLocked && !_wasLocked) HapticFeedback.mediumImpact();
    _wasLocked = _c.pupilLocked;
    if (mounted) setState(() {});
  }

  Future<void> _scan() async {
    HapticFeedback.selectionClick();
    await _c.startScan();
    final r = _c.result;
    if (r != null && mounted) _openResult(r);
  }

  void _openResult(ScanResult r) {
    Navigator.of(context).push(PageRouteBuilder<void>(
      transitionDuration: const Duration(milliseconds: 380),
      pageBuilder: (_, a, _) => FadeTransition(
        opacity: a,
        child: LiquidResultsDashboard(result: r, onRescan: () => Navigator.of(context).pop()),
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.paddingOf(context);
    final busy = _c.phase.isBusy;
    return Scaffold(
      backgroundColor: SynapseColors.background,
      body: AmbientBackdrop(
        child: Padding(
          padding: EdgeInsets.fromLTRB(16, pad.top + 12, 16, pad.bottom + 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Text('Synapse', style: SynapseType.display(36)),
                  const Spacer(),
                  _EyeSelector(
                    value: _c.eye,
                    enabled: !busy,
                    onChanged: (e) => setState(() => _c.eye = e),
                  ),
                ],
              ),
              const SizedBox(height: 2),
              Text('Quantitative pupillometry · on-device',
                  style: SynapseType.data(12.5, color: SynapseColors.textSecondary)),
              if (_c.camera case final SimulatedCamera sim) ...[
                const SizedBox(height: 12),
                _PatientSelector(
                  value: sim.patient,
                  enabled: !busy,
                  onChanged: (p) => setState(() => sim.patient = p),
                ),
              ],
              const SizedBox(height: 14),
              Expanded(child: RepaintBoundary(child: _Viewfinder(controller: _c))),
              const SizedBox(height: 14),
              _StatusLine(controller: _c, onDemo: () => _openResult(buildDemoScan())),
              const SizedBox(height: 12),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 250),
                child: _c.phase.isRecording || _c.phase == ScanPhase.analyzing
                    ? LiquidGlass(
                        key: const ValueKey('live'),
                        blur: 0,
                        padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
                        child: ValueListenableBuilder<List<KinematicPoint>>(
                          valueListenable: _c.liveCurve,
                          builder: (_, pts, _) =>
                              KinematicCurveChart(scan: pts, height: 120, showLegend: false),
                        ),
                      )
                    : _ScanButton(
                        key: const ValueKey('button'),
                        enabled: _c.phase == ScanPhase.searching ||
                            _c.phase == ScanPhase.aligned ||
                            _c.phase == ScanPhase.complete,
                        locked: _c.pupilLocked,
                        onTap: _scan,
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Viewfinder extends StatelessWidget {
  const _Viewfinder({required this.controller});

  final HardwareSyncController controller;

  @override
  Widget build(BuildContext context) {
    final session = controller.session;
    return ClipRRect(
      borderRadius: BorderRadius.circular(32),
      child: DecoratedBox(
        decoration: const BoxDecoration(color: Colors.black),
        child: LayoutBuilder(builder: (context, box) {
          if (session == null) {
            return Center(
              child: TargetingReticle(radius: box.maxWidth * 0.3, locked: false),
            );
          }
          final turns = session.quarterTurns % 2;
          final pw = turns == 1 ? session.previewHeight : session.previewWidth;
          final ph = turns == 1 ? session.previewWidth : session.previewHeight;
          // BoxFit.cover scale → the native ROI (centre crop) on screen.
          final scale = [box.maxWidth / pw, box.maxHeight / ph].reduce((a, b) => a > b ? a : b);
          final o = controller.latest;
          final roi = (o?.roiWidth ?? controller.config.roiSize) * scale;
          final locked = controller.pupilLocked;
          Offset? offset;
          double? pr;
          if (o != null && o.stabilizedX != null && locked) {
            offset = Offset(o.stabilizedX! - o.roiWidth / 2, o.stabilizedY! - o.roiHeight / 2) * scale;
            pr = (o.ellipse?.a ?? 0) * scale;
          }
          final d = o?.diameterMm;
          return Stack(
            fit: StackFit.expand,
            children: [
              if (controller.camera case final SimulatedCamera sim)
                SimulatedEyeView(camera: sim)
              else
                FittedBox(
                  fit: BoxFit.cover,
                  clipBehavior: Clip.hardEdge,
                  child: SizedBox(
                    width: pw,
                    height: ph,
                    child: RotatedBox(
                      quarterTurns: session.quarterTurns,
                      child: Texture(textureId: session.textureId, filterQuality: FilterQuality.low),
                    ),
                  ),
                ),
              Center(
                child: TargetingReticle(
                  radius: roi / 2,
                  locked: locked,
                  pupilOffset: offset,
                  pupilRadius: pr,
                  label: locked
                      ? (d == null ? 'LOCKED' : 'LOCKED · ${d.toStringAsFixed(2)} MM')
                      : 'ALIGN PUPIL',
                ),
              ),
              Positioned(
                left: 14,
                top: 14,
                child: _Chip(
                  text: controller.camera is SimulatedCamera
                      ? 'SIMULATED EYE · ${session.fps.toStringAsFixed(0)} FPS'
                      : '${session.fps.toStringAsFixed(0)} FPS',
                ),
              ),
              if (controller.sensorLock != null && controller.phase.isBusy)
                const Positioned(
                  left: 14,
                  top: 54,
                  child: _Chip(text: 'ISO · AF · AE LOCKED', color: SynapseColors.neonGreen),
                ),
            ],
          );
        }),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip({required this.text, this.color = SynapseColors.textPrimary});

  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: SynapseColors.halo,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(text, style: SynapseType.label(color: color)),
    );
  }
}

class _StatusLine extends StatelessWidget {
  const _StatusLine({required this.controller, required this.onDemo});

  final HardwareSyncController controller;
  final VoidCallback onDemo;

  @override
  Widget build(BuildContext context) {
    final error = controller.error;
    final color = error != null
        ? SynapseColors.abnormalRed
        : controller.phase.isRecording
            ? SynapseColors.neonGreen
            : SynapseColors.textPrimary;
    return Semantics(
      liveRegion: true,
      child: Row(
        children: [
          Expanded(
            child: Text(
              error ?? controller.phase.label,
              style: SynapseType.data(15, weight: FontWeight.w600, color: color),
            ),
          ),
          if (controller.phase == ScanPhase.error)
            TextButton(
              onPressed: onDemo,
              child: Text('View demo result',
                  style: SynapseType.data(13, color: SynapseColors.healthyBlue)),
            ),
        ],
      ),
    );
  }
}

class _EyeSelector extends StatelessWidget {
  const _EyeSelector({required this.value, required this.onChanged, required this.enabled});

  final Eye value;
  final ValueChanged<Eye> onChanged;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return LiquidGlass(
      radius: 18,
      blur: 0,
      padding: const EdgeInsets.all(4),
      elevation: 0.5,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          for (final e in Eye.values)
            Semantics(
              button: true,
              selected: e == value,
              label: e.label,
              child: GestureDetector(
                onTap: enabled ? () => onChanged(e) : null,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 180),
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: e == value ? SynapseColors.textPrimary : Colors.transparent,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Text(
                    e.code,
                    style: SynapseType.data(13,
                        weight: FontWeight.w700,
                        color: e == value ? Colors.black : SynapseColors.textPrimary),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// Simulator only: which reflex the synthetic patient will show.
class _PatientSelector extends StatelessWidget {
  const _PatientSelector({required this.value, required this.onChanged, required this.enabled});

  final SimulatedPatient value;
  final ValueChanged<SimulatedPatient> onChanged;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text('SIMULATED PATIENT', style: SynapseType.label()),
        const SizedBox(width: 12),
        Expanded(
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final p in SimulatedPatient.values)
                ChoiceChip(
                  label: Text(p.label),
                  selected: p == value,
                  onSelected: enabled ? (_) => onChanged(p) : null,
                  showCheckmark: false,
                  labelStyle: SynapseType.data(13,
                      weight: FontWeight.w600,
                      color: p == value ? Colors.black : SynapseColors.textPrimary),
                  selectedColor: SynapseColors.textPrimary,
                  backgroundColor: Colors.white.withValues(alpha: 0.05),
                  side: const BorderSide(color: Colors.white24),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ScanButton extends StatelessWidget {
  const _ScanButton({super.key, required this.enabled, required this.locked, required this.onTap});

  final bool enabled;
  final bool locked;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      enabled: enabled,
      label: 'Start pupillary light reflex scan',
      child: GestureDetector(
        onTap: enabled ? onTap : null,
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 200),
          opacity: enabled ? 1 : 0.4,
          child: Container(
            height: 68,
            decoration: BoxDecoration(
              color: locked ? SynapseColors.neonGreen : Colors.white.withValues(alpha: 0.06),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: locked ? SynapseColors.neonGreen : Colors.white24),
              boxShadow: locked
                  ? [BoxShadow(color: SynapseColors.neonGreen.withValues(alpha: 0.35), blurRadius: 24)]
                  : null,
            ),
            alignment: Alignment.center,
            child: Text(
              locked ? 'Scan' : 'Scan when locked',
              style: SynapseType.data(19,
                  weight: FontWeight.w700, color: locked ? Colors.black : SynapseColors.textPrimary),
            ),
          ),
        ),
      ),
    );
  }
}
