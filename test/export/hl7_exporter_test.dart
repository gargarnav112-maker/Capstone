import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/demo/demo_scan.dart';
import 'package:synapse/export/hl7_exporter.dart';

void main() {
  final scan = buildDemoScan();
  const exporter = Hl7Exporter();

  test('ORU^R01 is well-formed HL7 v2.5.1', () {
    final msg = exporter.toOruR01(scan, patientId: 'MRN|42^X', now: DateTime.utc(2026, 10, 8, 12));
    final segs = msg.split('\r')..removeLast();
    expect(msg.endsWith('\r'), isTrue);
    expect(segs.first, startsWith(r'MSH|^~\&|SYNAPSE|'));
    final msh = segs.first.split('|');
    expect(msh[8], 'ORU^R01^ORU_R01'); // MSH-9 (MSH-1 is the separator)
    expect(msh[11], '2.5.1'); // MSH-12
    expect(segs.map((s) => s.substring(0, 3)).toSet(), containsAll(['MSH', 'PID', 'OBR', 'OBX', 'NTE']));

    // Encoding characters in data are escaped.
    expect(segs[1], contains(r'MRN\F\42\S\X'));

    final obx = segs.where((s) => s.startsWith('OBX')).map((s) => s.split('|')).toList();
    // Set IDs are sequential.
    expect([for (final f in obx) f[1]], [for (var i = 1; i <= obx.length; i++) '$i']);
    final npi = obx.firstWhere((f) => f[3].startsWith('SYN-NPI'));
    expect(npi[2], 'NM');
    expect(npi[4], 'OD');
    expect(double.parse(npi[5]), scan.metrics.npi);
    expect(npi[8], 'N'); // OBX-8 abnormal flag
    expect(npi[11], 'F'); // OBX-11 result status
    final lat = obx.firstWhere((f) => f[3].startsWith('SYN-LAT'));
    expect(lat[6], startsWith('ms^'));

    final t = obx.firstWhere((f) => f[3].startsWith('SYN-CURVE-T'));
    final d = obx.firstWhere((f) => f[3].startsWith('SYN-CURVE-D'));
    expect(t[2], 'NA');
    expect(t[5].split('^').length, d[5].split('^').length);
    expect(t[5].split('^').length, greaterThan(300));
  });

  test('FHIR R4 bundle carries components and SampledData', () {
    final json = jsonDecode(exporter.toFhirJson(scan, patientReference: 'Patient/123')) as Map;
    expect(json['resourceType'], 'Bundle');
    final obs = (json['entry'] as List).single['resource'] as Map;
    expect(obs['resourceType'], 'Observation');
    expect(obs['status'], 'final');
    expect(obs['subject'], {'reference': 'Patient/123'});
    final comps = obs['component'] as List;
    final codes = comps.map((c) => c['code']['coding'][0]['code']).toList();
    expect(codes, containsAll(['SYN-NPI', 'SYN-LAT', 'SYN-MCV', 'SYN-CV', 'SYN-DV', 'SYN-CURVE']));
    final curve = comps.firstWhere((c) => c['code']['coding'][0]['code'] == 'SYN-CURVE');
    final sd = curve['valueSampledData'] as Map;
    expect(sd['period'], 10);
    expect((sd['data'] as String).split(' ').length, greaterThan(300));
  });

  test('FHIR omits subject when no patient is given', () {
    final json = exporter.toFhirBundle(scan);
    final obs = (json['entry'] as List).single['resource'] as Map;
    expect(obs.containsKey('subject'), isFalse);
  });
}
