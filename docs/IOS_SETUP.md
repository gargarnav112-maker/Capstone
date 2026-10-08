# Running Synapse on an iPhone

To build, you need a Mac. Total time is about 30–45 minutes the first time,
most of it waiting on downloads.

## What you need

| | |
|---|---|
| Mac | Apple Silicon or Intel, on a recent macOS that runs the current Xcode |
| Xcode | Latest version from the Mac App Store (≈ 10 GB) |
| iPhone | iOS 15 or later, with a rear camera and flash (any iPhone from the last ~8 years) |
| Cable | USB-C or Lightning cable that connects the iPhone to the Mac |
| Apple ID | A free one works. A paid developer account ($99/yr) is only needed for TestFlight or builds that don't expire after 7 days |

> **Use a physical iPhone.** The iOS Simulator has no camera and no flash.
> It still opens, shows "Rear camera with torch is required", and offers
> **View demo result** so you can see the dashboard.

---

## 1. Install the tools (one time)

Open **Terminal** (⌘-Space → "Terminal") and run each block.

**1a. Xcode.** Install it from the App Store, open it once, accept the
licence, and let it install its components. Then:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadPlatform iOS   # iOS device support, if Xcode didn't already
```

**1b. Homebrew** (package manager). Skip this if `brew --version` already works.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

When it finishes, run the two `echo … >> ~/.zprofile` / `eval …` lines it
prints at the end.

**1c. Flutter and CocoaPods.**

```bash
brew install --cask flutter
brew install cocoapods
flutter --version      # needs Flutter 3.47 or newer (Dart 3.13+)
```

If your Flutter is older, run `flutter upgrade`.

**1d. Health check.**

```bash
flutter doctor
```

The **Flutter** and **Xcode** lines must show ✓. Android-related ✗ lines
can be ignored.

## 2. Get the code

```bash
cd ~/Desktop
git clone https://github.com/gargarnav112-maker/Capstone.git
cd Capstone
git checkout claude/synapse-pupillometer-mvp-4h6q2l
flutter pub get
cd ios && pod install && cd ..
```

(If `pod install` says the project uses Swift Package Manager, that's fine.
Flutter will handle the plugin either way.)

## 3. Set up code signing (one time)

1. Open the workspace (not the `.xcodeproj`):
   ```bash
   open ios/Runner.xcworkspace
   ```
2. In Xcode, select **Runner** at the top of the left sidebar, then select
   the **Runner** target, then the **Signing & Capabilities** tab.
3. Tick **Automatically manage signing**.
4. **Team** → *Add an Account…* → sign in with your Apple ID → select
   "*Your Name (Personal Team)*".
5. **Bundle Identifier**: change `com.synapse.pupillometry.synapse` to
   something unique to you, e.g. `com.yourname.synapse`. Personal teams
   can't reuse an identifier someone else has registered.
6. Make sure the red signing errors are gone. Then close Xcode.

## 4. Prepare the iPhone (one time)

1. Connect the iPhone with the cable, unlock it, and tap **Trust This
   Computer**.
2. Turn on **Developer Mode**: *Settings → Privacy & Security → Developer
   Mode → On*. The phone restarts; confirm when asked.
   (The option appears only after the phone has been connected to Xcode
   once. If it's missing, open Xcode with the phone plugged in, then check
   again.)

## 5. Build and install

```bash
flutter devices                 # your iPhone should be listed
flutter run --release           # pick the iPhone if asked
```

Always use **`--release`**. Debug builds are too slow to process 120 fps,
and they stop working once the cable is unplugged.

The first build takes 5–10 minutes. If the app won't open and says
"Untrusted Developer", go to *Settings → General → VPN & Device Management*,
tap your Apple ID, then tap **Trust**. Then run `flutter run --release`
again or just tap the app icon.

The app stays installed after you unplug the phone. With a free Apple ID it
expires after **7 days**; repeat step 5 to reinstall.

## 6. Do a scan

1. Allow camera access when asked.
2. Work in a dim room (bright ambient light pre-constricts the pupil and
   leaves little reflex to measure).
3. Hold the phone **5–10 cm** from the subject's eye. The rear camera faces
   the eye, and you look at the screen.
4. Centre the pupil in the ring. Wait until the ring turns **solid neon
   green** (you'll feel a tap). The label shows the live diameter in mm.
5. Choose **OD** (right eye) or **OS** (left eye) at the top, then tap
   **Scan**. Hold still for about 4 seconds: lock → ½ s baseline → flash →
   3 s recording.
6. The results dashboard opens. **Export HL7 to EHR** lets you copy or share
   the HL7 v2 / FHIR payload.

If the scan is rejected (for example "Pupil tracked in only 60 % of
frames"), check the lighting, distance and blinking, then tap **Scan**
again.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `flutter devices` doesn't list the iPhone | Unlock the phone, reconnect the cable, accept **Trust**. Open Xcode → *Window → Devices and Simulators* and wait for "Preparing device…" to finish. |
| "Signing for Runner requires a development team" | Repeat step 3. Also set the same Team on the **RunnerTests** target if Xcode complains about it. |
| "Failed to register bundle identifier" | Pick a more unique Bundle Identifier (step 3.5). |
| `pod install` fails | `sudo gem install cocoapods` or `brew upgrade cocoapods`, then `cd ios && pod repo update && pod install`. |
| "Camera permission denied" | *Settings → Synapse → Camera → On*. |
| Ring never turns green | Get closer (the whole iris should fit inside the ring), use less glare, and keep the eyelid open. |
| Swift compile error in `SynapseCameraPlugin.swift` | The native iOS code has not been compiled on a Mac yet. Please send me the exact error text and I'll fix it. |

## What's verified and what isn't

- **Verified (in CI-style tests):** the Dart side, including the vision
  pipeline, biomarker maths, HL7/FHIR export and the full scan state
  machine, against a simulated camera.
- **Not yet verified:** the Swift camera plugin has not been compiled or run
  on a device. Expect possible small fixes on the first build. Please report
  any build errors or odd behaviour (wrong frame rate, the flash not firing,
  the ring never locking).

> Investigational software, not a medical device. Don't use it for clinical
> decisions.
