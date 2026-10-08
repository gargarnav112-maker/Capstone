# Synapse — smartphone quantitative pupillometer (MVP)

Synapse uses a phone's rear camera and torch to measure the **pupillary light
reflex (PLR)** for neurological triage. **Everything runs on the device.** The
release build has no `INTERNET` permission. Frames stay in memory and are never
written to disk or uploaded.

> **Investigational software. It is not a medical device and is not validated
> for diagnosis.** The "Synapse index" uses the same 0–5 scale as the NPi, but
> it is a transparent stand-in for NeurOptics' proprietary NPi®, not a copy of
> it. Its normative values (`lib/biomarkers/npi.dart`) are placeholders. Replace
> them with values from a validation study run against a reference pupillometer.

## Architecture

```
packages/synapse_camera/        Native plugin (no network, no disk)
  ios/…/SynapseCameraPlugin.swift   AVFoundation: 420f format at 120 fps (60 floor),
                                     custom exposure (fixed ISO + duration), locked
                                     focus/WB, HDR/low-light boost/stabilisation off,
                                     torch pulse on a strict GCD timer, host-clock stamps
  android/…/SynapseCameraPlugin.kt  Camera2: AE_MODE_OFF + SENSOR_SENSITIVITY/EXPOSURE,
                                     AF_MODE_OFF + LENS_FOCUS_DISTANCE (MANUAL_SENSOR),
                                     FLASH_MODE_TORCH on the repeating request
  lib/synapse_camera.dart           Dart bridge: texture, luma ROI stream, lock, flashPulse

lib/
  hardware/hardware_sync_controller.dart   4-step protocol state machine (see below)
  vision/            Edge-AI pipeline (runs on a background isolate)
    grayscale.dart        Y-plane = BT.601 luma (RGBA converter for fixtures)
    pupil_segmenter.dart  Bradley–Roth adaptive threshold + two-level Otsu cap,
                          8-connected components, pupil-likeness scoring,
                          hole-filled contour (removes the corneal glint)
    ellipse_fit.dart      Fitzgibbon direct least-squares ellipse fit
                          (Halíř–Flusser stable form) + robust MAD refit
    pupil_tracker.dart    sub-pixel edge refinement, Kalman tremor stabiliser,
                          iris-referenced px→mm
    pixel_scale.dart      limbus detection; HVID (11.7 mm) used as an in-frame ruler
    kalman.dart           constant-velocity Kalman + Rauch–Tung–Striebel smoother
  biomarkers/
    plr_analyzer.dart     optical flash-onset detection, Hampel filter, latency,
                          MCV/CV, DV, %CH, T75, quality checks
    npi.dart              composite index on the 0–5 scale
    reference_curves.dart parametric healthy/abnormal PLR waveforms
  export/hl7_exporter.dart  HL7 v2.5.1 ORU^R01 and FHIR R4 Bundle (SampledData curve)
  ui/
    liquid_results_dashboard.dart   LiquidResultsDashboard
    scanner_screen.dart             one-tap scanner, neon reticle, live curve
    widgets/                        LiquidGlass, GlassMetricCard,
                                    KinematicCurveChart, TargetingReticle
```

### The stimulus sequence (`HardwareSyncController`)

1. **Lock.** ISO, exposure duration (capped at one frame period), focus and
   white balance are frozen. About 120 ms of frames are then discarded while
   the image signal processor (ISP) pipeline settles.
2. **Baseline.** 0–500 ms (default 500) of resting pupil in ambient light.
3. **Flash.** A 100 ms torch pulse, timed by a native high-priority timer.
4. **Response.** Recording continues for 3 s after flash onset. The
   recording is then analysed.

The timeline advances on **sensor frame timestamps**, not Dart timers.
Stimulus onset is measured from the brightness step in the frames themselves
(with sub-frame interpolation), so torch rise time does not bias latency.
Biomarkers come from a **zero-lag RTS smoother**. A causal filter would
report latency late and peak velocity low.

### Metrics

| Metric | Definition |
|---|---|
| Latency | Flash onset → constriction speed first exceeds 15 % of its peak |
| Constriction velocity (MCV) | Peak \|dD/dt\| during constriction, mm/s; average CV is also reported (> 0.8 mm/s is normal) |
| Dilation velocity (DV) | Mean re-dilation speed from minimum to 75 % recovery |
| %CH | (baseline − min) / baseline |
| Synapse index | 5 − 1.25 · z̄, where z̄ is the weighted mean of one-sided z-scores; 0 if non-reactive |

## Running

```bash
flutter pub get
flutter test          # 33 tests: CV accuracy, analyzer, HL7/FHIR, WCAG, end-to-end controller
flutter run --release # on a physical iPhone / Android (Camera2 MANUAL_SENSOR recommended)
```

Hold the phone 5–10 cm from the eye in a dim room. Centre the pupil in the
ring and wait for it to turn solid green. Then tap **Scan** and keep still
for about 4 s. If no camera is available (for example on a simulator), the
error state offers **View demo result**, which runs the real analysis on a
modelled recording.

## Validation still required before clinical use

- Per-device torch calibration (lux at the cornea for a fixed `torchLevel`).
- Accuracy of the iris-diameter (HVID) scale against a fixed calibration target.
- Agreement study for the index and its norms against a reference pupillometer
  (e.g. NPi-200).
- Testing on dark irides, small pupils, and in direct sunlight.

Fonts: Bodoni Moda and Inter, both under the SIL Open Font License (see `assets/fonts/`).
