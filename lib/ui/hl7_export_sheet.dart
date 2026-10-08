import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:share_plus/share_plus.dart';

import '../biomarkers/plr_models.dart';
import '../export/hl7_exporter.dart';
import 'theme.dart';
import 'widgets/liquid_glass.dart';

enum _Format { v2, fhir }

/// Bottom sheet that formats the kinetic profile as an HL7 payload and hands
/// it to the clipboard or the OS share sheet. The app never transmits it
/// itself; the clinician picks the destination (e.g. the EHR's mobile app or
/// an MDM-managed interface inbox).
class Hl7ExportSheet extends StatefulWidget {
  const Hl7ExportSheet({super.key, required this.result, this.exporter = const Hl7Exporter()});

  final ScanResult result;
  final Hl7Exporter exporter;

  static Future<void> show(BuildContext context, ScanResult result) => showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        backgroundColor: Colors.transparent,
        barrierColor: Colors.black.withValues(alpha: 0.6),
        builder: (_) => Hl7ExportSheet(result: result),
      );

  @override
  State<Hl7ExportSheet> createState() => _Hl7ExportSheetState();
}

class _Hl7ExportSheetState extends State<Hl7ExportSheet> {
  _Format _format = _Format.v2;
  final _patientId = TextEditingController();

  @override
  void dispose() {
    _patientId.dispose();
    super.dispose();
  }

  String get _payload {
    final id = _patientId.text.trim();
    return switch (_format) {
      _Format.v2 => widget.exporter.toOruR01(widget.result, patientId: id.isEmpty ? 'UNKNOWN' : id),
      _Format.fhir => widget.exporter.toFhirJson(
          widget.result,
          patientReference: id.isEmpty ? null : 'Patient/$id',
        ),
    };
  }

  /// HL7 v2 uses CR segment terminators; show them as line breaks.
  String get _display => _payload.replaceAll('\r', '\n').trimRight();

  @override
  Widget build(BuildContext context) {
    final bottom = MediaQuery.viewInsetsOf(context).bottom;
    return Padding(
      padding: EdgeInsets.fromLTRB(12, 0, 12, 12 + bottom),
      child: SafeArea(
        top: false,
        child: LiquidGlass(
          radius: 32,
          blur: 40,
          padding: const EdgeInsets.fromLTRB(20, 14, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 38,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Text('Export to EHR', style: SynapseType.display(28)),
              const SizedBox(height: 4),
              Text(
                'HL7-formatted kinetic profile. Nothing leaves the device until you share it.',
                style: SynapseType.data(13, color: SynapseColors.textSecondary),
              ),
              const SizedBox(height: 16),
              SegmentedButton<_Format>(
                showSelectedIcon: false,
                style: SegmentedButton.styleFrom(
                  backgroundColor: Colors.white.withValues(alpha: 0.04),
                  selectedBackgroundColor: SynapseColors.neonGreen,
                  selectedForegroundColor: Colors.black,
                  foregroundColor: SynapseColors.textPrimary,
                  side: const BorderSide(color: Colors.white24),
                ),
                segments: const [
                  ButtonSegment(value: _Format.v2, label: Text('HL7 v2.5.1 ORU^R01')),
                  ButtonSegment(value: _Format.fhir, label: Text('FHIR R4')),
                ],
                selected: {_format},
                onSelectionChanged: (s) => setState(() => _format = s.first),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _patientId,
                onChanged: (_) => setState(() {}),
                style: SynapseType.data(15),
                cursorColor: SynapseColors.neonGreen,
                decoration: InputDecoration(
                  labelText: 'Patient MRN (optional)',
                  labelStyle: SynapseType.data(13, color: SynapseColors.textSecondary),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: const BorderSide(color: Colors.white24),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: const BorderSide(color: SynapseColors.neonGreen),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              ConstrainedBox(
                constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.32),
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.55),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(12),
                    scrollDirection: Axis.vertical,
                    child: SelectableText(
                      _display,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 11,
                        height: 1.35,
                        color: SynapseColors.textPrimary,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        foregroundColor: SynapseColors.textPrimary,
                        side: const BorderSide(color: Colors.white30),
                        minimumSize: const Size.fromHeight(52),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      ),
                      icon: const Icon(Icons.copy_rounded, size: 18),
                      label: const Text('Copy'),
                      onPressed: () async {
                        await Clipboard.setData(ClipboardData(text: _payload));
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Payload copied')),
                          );
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton.icon(
                      style: FilledButton.styleFrom(
                        backgroundColor: SynapseColors.neonGreen,
                        foregroundColor: Colors.black,
                        minimumSize: const Size.fromHeight(52),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      ),
                      icon: const Icon(Icons.ios_share_rounded, size: 18),
                      label: const Text('Share'),
                      onPressed: () => SharePlus.instance.share(ShareParams(
                        text: _payload,
                        subject: 'Synapse PLR ${widget.result.eye.code} ${widget.result.scanId}',
                      )),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
