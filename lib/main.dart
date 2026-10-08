import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'demo/simulated_camera.dart';
import 'hardware/hardware_sync_controller.dart';
import 'ui/scanner_screen.dart';
import 'ui/theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.light.copyWith(
    systemNavigationBarColor: SynapseColors.background,
    statusBarColor: Colors.transparent,
  ));
  runApp(const SynapseApp());
}

class SynapseApp extends StatelessWidget {
  const SynapseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Synapse',
      debugShowCheckedModeBanner: false,
      theme: buildSynapseTheme(),
      darkTheme: buildSynapseTheme(),
      themeMode: ThemeMode.dark,
      home: kIsWeb
          // Browsers can't lock ISO/focus or fire the torch, so the web build
          // scans a simulated eye through the real pipeline.
          ? ScannerScreen(
              controller: HardwareSyncController(
                camera: SimulatedCamera(fps: 60),
                config: const HardwareSyncConfig(targetFps: 60, roiSize: SimulatedCamera.roiSize),
              ),
            )
          : const ScannerScreen(),
    );
  }
}
