import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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
      home: const ScannerScreen(),
    );
  }
}
