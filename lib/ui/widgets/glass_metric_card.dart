import 'package:flutter/material.dart';

import '../theme.dart';
import 'liquid_glass.dart';

enum MetricStatus { normal, abnormal, neutral }

/// A floating frosted-glass widget housing one clinical metric.
class GlassMetricCard extends StatelessWidget {
  const GlassMetricCard({
    super.key,
    required this.label,
    required this.value,
    this.unit,
    this.secondary,
    this.reference,
    this.status = MetricStatus.neutral,
  });

  final String label;
  final String value;
  final String? unit;

  /// Smaller companion reading (e.g. average CV under peak MCV).
  final String? secondary;

  /// Reference range caption, e.g. "ref ≤ 300 ms".
  final String? reference;
  final MetricStatus status;

  Color get _statusColor => switch (status) {
        MetricStatus.normal => SynapseColors.neonGreen,
        MetricStatus.abnormal => SynapseColors.abnormalRed,
        MetricStatus.neutral => SynapseColors.textSecondary,
      };

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      label: '$label $value ${unit ?? ''}. ${switch (status) {
        MetricStatus.normal => 'Within reference.',
        MetricStatus.abnormal => 'Outside reference.',
        MetricStatus.neutral => '',
      }}',
      child: ExcludeSemantics(
        child: LiquidGlass(
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
          radius: 24,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Expanded(child: Text(label.toUpperCase(), style: SynapseType.label())),
                  if (status != MetricStatus.neutral)
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: _statusColor,
                        shape: BoxShape.circle,
                        boxShadow: [BoxShadow(color: _statusColor.withValues(alpha: 0.7), blurRadius: 8)],
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 10),
              FittedBox(
                fit: BoxFit.scaleDown,
                alignment: Alignment.centerLeft,
                child: Text.rich(
                  TextSpan(children: [
                    TextSpan(text: value, style: SynapseType.data(34, weight: FontWeight.w500)),
                    if (unit != null)
                      TextSpan(
                        text: ' $unit',
                        style: SynapseType.data(14, color: SynapseColors.textSecondary),
                      ),
                  ]),
                  maxLines: 1,
                ),
              ),
              if (secondary != null) ...[
                const SizedBox(height: 4),
                Text(secondary!, style: SynapseType.data(12.5, color: SynapseColors.textSecondary)),
              ],
              if (reference != null) ...[
                const SizedBox(height: 8),
                Text(reference!, style: SynapseType.data(11.5, color: SynapseColors.textSecondary)),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
