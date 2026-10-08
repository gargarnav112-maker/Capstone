import 'dart:convert';

import '../biomarkers/plr_models.dart';

/// Formats a [ScanResult] for EHR integration.
///
/// Two encodings are produced from the same data:
///
///  * **HL7 v2.5.1 ORU^R01** – the unsolicited-observation message most
///    hospital interface engines (Epic Bridges, Cerner, Rhapsody, Mirth)
///    accept today. One NM OBX per biomarker, plus NA (numeric array) OBXs for
///    the time and diameter vectors of the kinematic curve.
///  * **HL7 FHIR R4** – a `Bundle` (type `collection`) containing one
///    `Observation` with a `component` per biomarker and the kinematic curve as
///    `SampledData` (period in ms, data in mm).
///
/// Coding: no LOINC codes exist for most of these PLR kinematics, so they are
/// sent under a local code system (`99SYN` in v2, [codeSystem] in FHIR) – the
/// standard practice until the receiving site maps them. Units are UCUM.
///
/// Patient identity is supplied by the caller (e.g. scanned from a wristband);
/// the device never stores it.
class Hl7Exporter {
  const Hl7Exporter({
    this.sendingApplication = 'SYNAPSE',
    this.sendingFacility = 'SYNAPSE_DEVICE',
    this.receivingApplication = 'EHR',
    this.receivingFacility = 'HOSPITAL',
    this.codeSystem = 'urn:synapse:pupillometry',
    this.curveSampleMs = 10,
  });

  final String sendingApplication;
  final String sendingFacility;
  final String receivingApplication;
  final String receivingFacility;
  final String codeSystem;

  /// The curve is resampled to this period to keep messages compact.
  final double curveSampleMs;

  static const _localSystem = '99SYN';

  // ------------------------------------------------------------ HL7 v2.5.1

  String toOruR01(
    ScanResult r, {
    String patientId = 'UNKNOWN',
    String? patientName,
    String? messageControlId,
    DateTime? now,
  }) {
    final ts = _hl7Ts(now ?? DateTime.now());
    final obsTs = _hl7Ts(r.recordedAt);
    final m = r.metrics;
    final control = messageControlId ?? r.scanId;
    final segments = <String>[
      // MSH-1 is the field separator itself, so MSH fields are offset by one.
      'MSH|^~\\&|${_e(sendingApplication)}|${_e(sendingFacility)}|${_e(receivingApplication)}|'
          '${_e(receivingFacility)}|$ts||ORU^R01^ORU_R01|${_e(control)}|P|2.5.1|||NE|AL',
      _segment('PID', {
        1: '1',
        3: '${_e(patientId)}^^^${_e(sendingFacility)}^MR',
        5: _e(patientName ?? ''),
      }),
      _segment('OBR', {
        1: '1',
        3: '${_e(r.scanId)}^${_e(sendingApplication)}',
        4: 'SYN-PLR^Quantitative pupillometry (pupillary light reflex)^$_localSystem',
        7: obsTs,
        25: 'F',
      }),
    ];

    var set = 0;
    void obx(String type, String code, String text, String value, {String units = '', String flag = ''}) {
      set++;
      segments.add(_segment('OBX', {
        1: '$set',
        2: type,
        3: '$code^${_e(text)}^$_localSystem',
        4: r.eye.code, // observation sub-ID: laterality
        5: value,
        6: units,
        8: flag,
        11: 'F',
        14: obsTs,
      }));
    }

    void nm(String code, String text, double? value, String unit, String unitText,
        {int digits = 2, String flag = ''}) {
      if (value == null) return;
      obx('NM', code, text, value.toStringAsFixed(digits),
          units: '${_e(unit)}^${_e(unitText)}^UCUM', flag: flag);
    }

    final abn = m.npiCategory == NpiCategory.brisk ? 'N' : 'A';
    nm('SYN-NPI', 'Synapse pupil index (0-5)', m.npi, '{score}', 'score', digits: 1, flag: abn);
    nm('SYN-LAT', 'Constriction latency', m.latencyMs, 'ms', 'millisecond', digits: 0);
    nm('SYN-MCV', 'Maximum constriction velocity', m.maxConstrictionVelocityMmS, 'mm/s',
        'millimeter per second');
    nm('SYN-CV', 'Average constriction velocity', m.avgConstrictionVelocityMmS, 'mm/s',
        'millimeter per second');
    nm('SYN-DV', 'Average dilation velocity', m.dilationVelocityMmS, 'mm/s',
        'millimeter per second');
    nm('SYN-SIZE', 'Resting pupil diameter', m.baselineDiameterMm, 'mm', 'millimeter');
    nm('SYN-MIN', 'Minimum pupil diameter', m.minDiameterMm, 'mm', 'millimeter');
    nm('SYN-CH', 'Percent constriction', m.constrictionPercent, '%', 'percent', digits: 1);
    nm('SYN-T75', 'Time to 75% re-dilation', m.t75Ms, 'ms', 'millisecond', digits: 0);

    // NA = numeric array; elements are component-separated.
    final curve = _resample(r.curve);
    obx('NA', 'SYN-CURVE-T', 'Kinematic curve time axis',
        curve.map((p) => p.tMs.toStringAsFixed(0)).join('^'),
        units: 'ms^millisecond^UCUM');
    obx('NA', 'SYN-CURVE-D', 'Kinematic curve pupil diameter',
        curve.map((p) => p.diameterMm.toStringAsFixed(3)).join('^'),
        units: 'mm^millimeter^UCUM');
    for (final w in r.quality.warnings) {
      obx('ST', 'SYN-QA', 'Acquisition quality note', _e(w));
    }
    segments.add(_segment('NTE', {
      1: '1',
      2: 'L',
      3: _e('Device-calculated research index; not the NeurOptics NPi. '
          'Not validated for diagnostic use.'),
    }));
    // HL7 v2 segment terminator is a carriage return.
    return '${segments.join('\r')}\r';
  }

  /// Builds a segment from 1-based field positions (gaps become empty fields).
  static String _segment(String id, Map<int, String> fields) {
    final last = fields.keys.reduce((a, b) => a > b ? a : b);
    return [id, for (var i = 1; i <= last; i++) fields[i] ?? ''].join('|');
  }

  /// HL7 v2 escape sequences for the encoding characters.
  static String _e(String s) => s
      .replaceAll('\\', r'\E\')
      .replaceAll('|', r'\F\')
      .replaceAll('^', r'\S\')
      .replaceAll('&', r'\T\')
      .replaceAll('~', r'\R\')
      .replaceAll('\r', ' ')
      .replaceAll('\n', ' ');

  static String _hl7Ts(DateTime t) {
    final u = t.toUtc();
    String two(int v) => v.toString().padLeft(2, '0');
    return '${u.year}${two(u.month)}${two(u.day)}${two(u.hour)}${two(u.minute)}${two(u.second)}+0000';
  }

  // ------------------------------------------------------------ FHIR R4

  Map<String, Object?> toFhirBundle(ScanResult r, {String? patientReference}) {
    final m = r.metrics;
    final curve = _resample(r.curve);
    Map<String, Object?> qty(String code, String display, double? v, String unit) => {
          'code': {
            'coding': [
              {'system': codeSystem, 'code': code, 'display': display},
            ],
          },
          if (v != null)
            'valueQuantity': {
              'value': double.parse(v.toStringAsFixed(3)),
              'unit': unit,
              'system': 'http://unitsofmeasure.org',
              'code': unit,
            }
          else
            'dataAbsentReason': {
              'coding': [
                {
                  'system': 'http://terminology.hl7.org/CodeSystem/data-absent-reason',
                  'code': 'not-applicable',
                },
              ],
            },
        };

    final observation = <String, Object?>{
      'resourceType': 'Observation',
      'id': r.scanId,
      'status': 'final',
      'category': [
        {
          'coding': [
            {
              'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
              'code': 'exam',
              'display': 'Exam',
            },
          ],
        },
      ],
      'code': {
        'coding': [
          {'system': codeSystem, 'code': 'SYN-PLR', 'display': 'Quantitative pupillometry (PLR)'},
        ],
        'text': 'Pupillary light reflex',
      },
      'subject': ?(patientReference == null ? null : {'reference': patientReference}),
      'effectiveDateTime': r.recordedAt.toUtc().toIso8601String(),
      'bodySite': {
        'coding': [
          {
            'system': 'http://snomed.info/sct',
            'code': r.eye == Eye.right ? '18944008' : '8966001',
            'display': r.eye == Eye.right ? 'Right eye structure' : 'Left eye structure',
          },
        ],
      },
      'interpretation': [
        {
          'coding': [
            {
              'system': 'http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation',
              'code': m.npiCategory == NpiCategory.brisk ? 'N' : 'A',
            },
          ],
          'text': m.npiCategory.description,
        },
      ],
      'device': {'display': 'Synapse smartphone pupillometer (on-device processing)'},
      'note': [
        {'text': 'Synapse index is a device-calculated research surrogate, not the NeurOptics NPi.'},
        for (final w in r.quality.warnings) {'text': 'QA: $w'},
      ],
      'component': [
        qty('SYN-NPI', 'Synapse pupil index (0-5)', m.npi, '{score}'),
        qty('SYN-LAT', 'Constriction latency', m.latencyMs, 'ms'),
        qty('SYN-MCV', 'Maximum constriction velocity', m.maxConstrictionVelocityMmS, 'mm/s'),
        qty('SYN-CV', 'Average constriction velocity', m.avgConstrictionVelocityMmS, 'mm/s'),
        qty('SYN-DV', 'Average dilation velocity', m.dilationVelocityMmS, 'mm/s'),
        qty('SYN-SIZE', 'Resting pupil diameter', m.baselineDiameterMm, 'mm'),
        qty('SYN-MIN', 'Minimum pupil diameter', m.minDiameterMm, 'mm'),
        qty('SYN-CH', 'Percent constriction', m.constrictionPercent, '%'),
        {
          'code': {
            'coding': [
              {'system': codeSystem, 'code': 'SYN-CURVE', 'display': 'Pupil diameter kinematic curve'},
            ],
          },
          'valueSampledData': {
            'origin': {
              'value': 0,
              'unit': 'mm',
              'system': 'http://unitsofmeasure.org',
              'code': 'mm',
            },
            'period': curveSampleMs,
            'dimensions': 1,
            'data': curve.map((p) => p.diameterMm.toStringAsFixed(3)).join(' '),
          },
        },
        {
          'code': {
            'coding': [
              {'system': codeSystem, 'code': 'SYN-CURVE-T0', 'display': 'Curve start relative to flash'},
            ],
          },
          'valueQuantity': {
            'value': curve.isEmpty ? 0 : curve.first.tMs,
            'unit': 'ms',
            'system': 'http://unitsofmeasure.org',
            'code': 'ms',
          },
        },
      ],
    };

    return {
      'resourceType': 'Bundle',
      'type': 'collection',
      'timestamp': DateTime.now().toUtc().toIso8601String(),
      'entry': [
        {'fullUrl': 'urn:uuid:${r.scanId}', 'resource': observation},
      ],
    };
  }

  String toFhirJson(ScanResult r, {String? patientReference}) =>
      const JsonEncoder.withIndent('  ').convert(toFhirBundle(r, patientReference: patientReference));

  /// Linear resampling onto a uniform grid (SampledData needs a fixed period).
  List<KinematicPoint> _resample(List<KinematicPoint> c) {
    if (c.length < 2) return c;
    final out = <KinematicPoint>[];
    var j = 0;
    for (var t = c.first.tMs; t <= c.last.tMs; t += curveSampleMs) {
      while (j < c.length - 2 && c[j + 1].tMs < t) {
        j++;
      }
      final a = c[j], b = c[j + 1];
      final f = b.tMs == a.tMs ? 0.0 : ((t - a.tMs) / (b.tMs - a.tMs)).clamp(0.0, 1.0);
      out.add(KinematicPoint(t, a.diameterMm + f * (b.diameterMm - a.diameterMm)));
    }
    return out;
  }
}
