import 'package:flutter_test/flutter_test.dart';
import 'package:synapse/biomarkers/plr_models.dart';
import 'package:synapse/demo/simulated_camera.dart';
import 'package:synapse/hardware/hardware_sync_controller.dart';

/// The web build's simulator, run through the real controller + pipeline.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  Future<ScanResult> scan(SimulatedPatient p) async {
    final c = HardwareSyncController(
      camera: SimulatedCamera(fps: 60, patient: p),
      config: const HardwareSyncConfig(targetFps: 60, roiSize: SimulatedCamera.roiSize),
    );
    await c.initialize();
    await c.startScan();
    expect(c.error, isNull);
    final r = c.result!;
    c.dispose();
    return r;
  }

  test('healthy simulated patient → brisk, ~5 mm', () async {
    final r = await scan(SimulatedPatient.healthy);
    expect(r.metrics.baselineDiameterMm, closeTo(5.0, 0.2));
    // Model latency is 220 ms after the light actually reaches the eye.
    expect(r.metrics.latencyMs, closeTo(220, 40));
    expect(r.metrics.npiCategory, NpiCategory.brisk);
    expect(r.quality.detectedFraction, greaterThan(0.95));
  }, timeout: const Timeout(Duration(seconds: 60)));

  test('sluggish simulated patient → abnormal', () async {
    final r = await scan(SimulatedPatient.sluggish);
    expect(r.metrics.npiCategory, NpiCategory.sluggish);
  }, timeout: const Timeout(Duration(seconds: 60)));

  test('fixed pupil → non-reactive', () async {
    final r = await scan(SimulatedPatient.nonReactive);
    expect(r.metrics.npiCategory, NpiCategory.nonReactive);
  }, timeout: const Timeout(Duration(seconds: 60)));
}
