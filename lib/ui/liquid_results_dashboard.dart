import 'package:flutter/material.dart';

import '../biomarkers/npi.dart';
import '../biomarkers/plr_models.dart';
import 'hl7_export_sheet.dart';
import 'theme.dart';
import 'widgets/glass_metric_card.dart';
import 'widgets/kinematic_curve_chart.dart';
import 'widgets/liquid_glass.dart';

/// Results screen: NPi hero, floating glass metric widgets, the kinematic
/// curve against healthy/abnormal references, acquisition QA, and HL7 export.
class LiquidResultsDashboard extends StatelessWidget {
  const LiquidResultsDashboard({
    super.key,
    required this.result,
    this.onRescan,
    this.norms = NpiCalculator.defaultNorms,
  });

  final ScanResult result;
  final VoidCallback? onRescan;
  final Map<String, Norm> norms;

  /// Normal when within 2 SD in the pathological direction – the same
  /// norms that feed the composite index, so cards and index never disagree.
  MetricStatus _status(String key, double? v) {
    final n = norms[key];
    if (n == null || v == null) return MetricStatus.abnormal;
    return n.penalty(v) <= 2 ? MetricStatus.normal : MetricStatus.abnormal;
  }

  String _limit(String key, String unit, {int digits = 1}) {
    final n = norms[key]!;
    final lim = n.higherIsWorse ? n.mean + 2 * n.sd : n.mean - 2 * n.sd;
    return 'ref ${n.higherIsWorse ? '≤' : '≥'} ${lim.toStringAsFixed(digits)} $unit';
  }

  @override
  Widget build(BuildContext context) {
    final m = result.metrics;
    final q = result.quality;
    final padding = MediaQuery.paddingOf(context);

    return Scaffold(
      backgroundColor: SynapseColors.background,
      body: AmbientBackdrop(
        child: CustomScrollView(
          physics: const BouncingScrollPhysics(),
          slivers: [
            SliverPadding(
              padding: EdgeInsets.fromLTRB(20, padding.top + 16, 20, padding.bottom + 32),
              sliver: SliverList.list(children: [
                _Header(result: result, onBack: Navigator.of(context).canPop() ? () => Navigator.pop(context) : null),
                const SizedBox(height: 22),
                _NpiHero(metrics: m),
                const SizedBox(height: 14),
                _MetricGrid(children: [
                  GlassMetricCard(
                    label: 'Latency',
                    value: m.latencyMs?.toStringAsFixed(0) ?? '—',
                    unit: 'ms',
                    reference: _limit('latencyMs', 'ms', digits: 0),
                    secondary: 'healthy ≈ 200 ms',
                    status: _status('latencyMs', m.latencyMs),
                  ),
                  GlassMetricCard(
                    label: 'Constriction vel.',
                    value: m.maxConstrictionVelocityMmS.toStringAsFixed(2),
                    unit: 'mm/s',
                    secondary: 'avg ${m.avgConstrictionVelocityMmS.toStringAsFixed(2)} mm/s',
                    reference: _limit('avgConstrictionVelocityMmS', 'mm/s'),
                    status: _status('avgConstrictionVelocityMmS', m.avgConstrictionVelocityMmS),
                  ),
                  GlassMetricCard(
                    label: 'Dilation vel.',
                    value: m.dilationVelocityMmS.toStringAsFixed(2),
                    unit: 'mm/s',
                    secondary: 'peak ${m.peakDilationVelocityMmS.toStringAsFixed(2)} mm/s'
                        '${m.t75Ms == null ? '' : ' · T75 ${m.t75Ms!.toStringAsFixed(0)} ms'}',
                    reference: _limit('dilationVelocityMmS', 'mm/s'),
                    status: _status('dilationVelocityMmS', m.dilationVelocityMmS),
                  ),
                  GlassMetricCard(
                    label: 'Constriction',
                    value: m.constrictionPercent.toStringAsFixed(0),
                    unit: '%',
                    secondary:
                        '${m.baselineDiameterMm.toStringAsFixed(2)} → ${m.minDiameterMm.toStringAsFixed(2)} mm',
                    reference: _limit('constrictionPercent', '%', digits: 0),
                    status: _status('constrictionPercent', m.constrictionPercent),
                  ),
                ]),
                const SizedBox(height: 14),
                LiquidGlass(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Kinematic Curve', style: SynapseType.display(22)),
                      const SizedBox(height: 2),
                      Text('Pupil diameter (mm) over time (ms) from flash onset',
                          style: SynapseType.data(12.5, color: SynapseColors.textSecondary)),
                      const SizedBox(height: 14),
                      KinematicCurveChart(scan: result.curve, baselineMm: m.baselineDiameterMm),
                    ],
                  ),
                ),
                const SizedBox(height: 14),
                _QualityPanel(quality: q, sensor: result.sensor),
                const SizedBox(height: 20),
                _ExportButton(onPressed: () => Hl7ExportSheet.show(context, result)),
                if (onRescan != null) ...[
                  const SizedBox(height: 10),
                  TextButton(
                    onPressed: onRescan,
                    style: TextButton.styleFrom(
                      foregroundColor: SynapseColors.textPrimary,
                      minimumSize: const Size.fromHeight(48),
                    ),
                    child: Text('New scan', style: SynapseType.data(15)),
                  ),
                ],
                const SizedBox(height: 18),
                Text(
                  'Synapse index is a device-calculated composite on the NPi scale (0–5); '
                  'it is not the proprietary NeurOptics NPi. Investigational device – not '
                  'validated for diagnosis. All processing occurred on this device.',
                  textAlign: TextAlign.center,
                  style: SynapseType.data(11.5, color: SynapseColors.textSecondary),
                ),
              ]),
            ),
          ],
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.result, this.onBack});

  final ScanResult result;
  final VoidCallback? onBack;

  @override
  Widget build(BuildContext context) {
    final t = result.recordedAt;
    String two(int v) => v.toString().padLeft(2, '0');
    return Row(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Synapse', style: SynapseType.display(44, weight: FontWeight.w500)),
              const SizedBox(height: 6),
              Text(
                'Pupillary light reflex · ${result.eye.label} (${result.eye.code}) · '
                '${two(t.hour)}:${two(t.minute)}:${two(t.second)}',
                style: SynapseType.data(13, color: SynapseColors.textSecondary),
              ),
            ],
          ),
        ),
        if (onBack != null)
          IconButton(
            tooltip: 'Back to scanner',
            onPressed: onBack,
            icon: const Icon(Icons.close_rounded, color: SynapseColors.textPrimary),
          ),
      ],
    );
  }
}

class _NpiHero extends StatelessWidget {
  const _NpiHero({required this.metrics});

  final PlrMetrics metrics;

  @override
  Widget build(BuildContext context) {
    final cat = metrics.npiCategory;
    final color = cat == NpiCategory.brisk ? SynapseColors.neonGreen : SynapseColors.abnormalRed;
    return Semantics(
      label: 'Pupil index ${metrics.npi.toStringAsFixed(1)} of 5, ${cat.label}',
      child: LiquidGlass(
        tint: color.withValues(alpha: 0.06),
        padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('NPi · SYNAPSE INDEX', style: SynapseType.label()),
                  const SizedBox(height: 6),
                  Text.rich(TextSpan(children: [
                    TextSpan(
                      text: metrics.npi.toStringAsFixed(1),
                      style: SynapseType.data(64, weight: FontWeight.w400),
                    ),
                    TextSpan(
                      text: ' / 5.0',
                      style: SynapseType.data(18, color: SynapseColors.textSecondary),
                    ),
                  ])),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.14),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: color.withValues(alpha: 0.6)),
                    ),
                    child: Text('${cat.label} · ${cat.description}',
                        style: SynapseType.data(12.5, weight: FontWeight.w600, color: color)),
                  ),
                ],
              ),
            ),
            _NpiGauge(value: metrics.npi, color: color),
          ],
        ),
      ),
    );
  }
}

class _NpiGauge extends StatelessWidget {
  const _NpiGauge({required this.value, required this.color});

  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 96,
      height: 96,
      child: TweenAnimationBuilder<double>(
        tween: Tween(begin: 0, end: value / 5),
        duration: const Duration(milliseconds: 900),
        curve: Curves.easeOutCubic,
        builder: (context, v, _) => CircularProgressIndicator(
          value: v,
          strokeWidth: 7,
          strokeCap: StrokeCap.round,
          color: color,
          backgroundColor: Colors.white.withValues(alpha: 0.08),
        ),
      ),
    );
  }
}

class _MetricGrid extends StatelessWidget {
  const _MetricGrid({required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      const gap = 12.0;
      final cols = c.maxWidth > 560 ? 4 : 2;
      final w = (c.maxWidth - gap * (cols - 1)) / cols;
      return Wrap(
        spacing: gap,
        runSpacing: gap,
        children: [for (final child in children) SizedBox(width: w, child: child)],
      );
    });
  }
}

class _QualityPanel extends StatelessWidget {
  const _QualityPanel({required this.quality, required this.sensor});

  final ScanQuality quality;
  final Map<String, Object> sensor;

  @override
  Widget build(BuildContext context) {
    String fmt(Object? v, {int d = 0}) => v is num ? v.toStringAsFixed(d) : '—';
    final chips = <(String, String)>[
      ('Frame rate', '${fmt(quality.effectiveFps)} fps'),
      ('Tracked', '${(quality.detectedFraction * 100).toStringAsFixed(0)} %'),
      ('Max gap', '${fmt(quality.maxGapMs, d: 1)} ms'),
      ('Blinks', '${quality.blinkCount}'),
      ('Flash', quality.flashOpticalDurationMs == null ? 'command' : '${fmt(quality.flashOpticalDurationMs)} ms'),
      ('ISO', fmt(sensor['iso'])),
      ('Exposure', '${fmt(sensor['exposureUs'])} µs'),
      ('Sensor lock', sensor['manual'] == true ? 'manual' : 'AE/AF lock'),
    ];
    return LiquidGlass(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('ACQUISITION QUALITY', style: SynapseType.label()),
              const Spacer(),
              Icon(
                quality.acceptable ? Icons.verified_rounded : Icons.warning_amber_rounded,
                size: 18,
                color: quality.acceptable ? SynapseColors.neonGreen : SynapseColors.amber,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 18,
            runSpacing: 10,
            children: [
              for (final (k, v) in chips)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(k, style: SynapseType.data(11, color: SynapseColors.textSecondary)),
                    Text(v, style: SynapseType.data(14)),
                  ],
                ),
            ],
          ),
          for (final w in quality.warnings) ...[
            const SizedBox(height: 10),
            Text('⚠ $w', style: SynapseType.data(12.5, color: SynapseColors.amber)),
          ],
        ],
      ),
    );
  }
}

class _ExportButton extends StatelessWidget {
  const _ExportButton({required this.onPressed});

  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Export HL7 payload to EHR',
      child: GestureDetector(
        onTap: onPressed,
        child: LiquidGlass(
          radius: 22,
          tint: SynapseColors.neonGreen.withValues(alpha: 0.08),
          padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 20),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.local_hospital_outlined, color: SynapseColors.neonGreen, size: 20),
              const SizedBox(width: 10),
              Text('Export HL7 to EHR',
                  style: SynapseType.data(16, weight: FontWeight.w600, color: SynapseColors.neonGreen)),
            ],
          ),
        ),
      ),
    );
  }
}
