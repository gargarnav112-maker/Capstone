# ASHKE — lyric edit (Karan Aujla)

A procedural, audio-reactive 9:16 lyric video. Every sung word is force-aligned to the vocal, and each
line has its own animation based on what the line means. The engine is pure Python
(numpy / OpenCV / Pillow) piped into ffmpeg. All graphics are drawn in code; no artwork is downloaded.

## Deliverables (`out/`)
| File | Spec |
|---|---|
| `Ashke_LyricEdit_1080x1920.mp4` | 1080×1920, 30 fps, H.264 High 20 Mbps (2-pass), AAC 320k 48 kHz, 23.000 s |
| `Ashke_LyricEdit_under25MB.mp4` | 1080×1920, 30 fps, 7.6 Mbps (2-pass), AAC 192k, 22.5 MB |
| `Ashke_LyricEdit_preview_480p.mp4` | 480×854 preview |

## Decisions you should know about
* **Window = song 2:52.000 → 3:15.000 (23.000 s), not 2:50–3:06.** Forced alignment showed the six lines
  run 2:52.18 → 3:13.56. The 2:50–3:06 window only held the previous line's tail plus L1–L4. You chose
  "all six lines". After "ashke" (3:13.53) the audio crossfades (90 ms) into the separated **instrumental**
  stem, so the hook's repeat ("ni modhe…") doesn't play uncaptioned under the end card. Fade-in 0.15 s, fade-out 0.4 s.
* **30 fps, not 60.** The screen recording is VFR with an average of ~28 fps (480 frames in 16.5 s).
* **Footage sync.** The screen recording is HEVC with rotation=90. Cropdetect gives 2382×1074+0+66; there is also
  an iPhone player band on the left, so the active picture is 2140×1068 (2:1). An RMS/onset cross-correlation
  places footage t=0 at **song 172.304 s** (fine sample-level refinement agrees). The footage covers song
  172.30–188.85, so the lip-synced picture runs to 16.85 s of the edit.
* **B-roll / slow-motion.** L5 is real-time until 15.6 s, then a 0.49× optical-flow (DIS) slow-mo of the same
  close-up so it reaches the line end. L6 (18.1–21.7 s) has no matching footage, so it uses the golden
  polo-field shot (footage 5.0–6.8 s) at ~0.5× with flow interpolation.
* **Zoom stays ≤ 1.5×.** A full-height 9:16 crop of 1068 px would need 1.8×, so every layout is designed around
  ≤ 1.5× footage: a cut-out, card background, sky band, letterbox strip, circle, and in full-bleed scenes a
  darkened, blurred "cover" extension behind the sharp layer. Nothing is stretched.
* **Masks.** MediaPipe selfie-multiclass person masks, smoothed with forward/backward EMA within each shot.
  L1 keeps the connected component under the tracked face (whole people, never half-sliced). The dark
  dining shots use a soft body window under the subject's face. No layout had to fall back to text-in-front.
* **End card:** no handle (per your answer). It shows the outlined ASHKE, an original drawn camera icon
  with an animated gold→saffron→red gradient, and "KARAN AUJLA / ASHKE".
* **No style reference video was attached**, so the style follows the written brief.

## Pipeline (`scripts/`)
1. `separate.py` runs UVR-MDX-NET-Voc_FT (sherpa-onnx) for the vocal and instrumental stems.
2. `align.py` runs CTC Viterbi forced alignment with **Meta Omnilingual-ASR 300M CTC**, using a wildcard
   prefix and suffix. The aligner reads a per-word Gurmukhi transliteration of *your* lyrics (`lyrics.py`).
   Display text is always your spelling, mapped 1:1 by word. Each word was cross-checked with the 1B model:
   max disagreement 60 ms, median 20 ms. No word was low-confidence, so none needed asking about.
3. `analyze_audio.py` runs librosa: 89.6 BPM, beats, downbeats, band split (bass < 150 Hz / mids / highs >
   4 kHz) with envelopes and onsets, written to `work/beats.json`. Word starts snap to the nearest beat or
   (backtracked) onset within 80 ms, written to `work/words.json`.
4. `footage.py` + `subject.py` do face detection, person masks, and per-shot subject tracking.
5. `render.py` renders the frames (`engine/scene_l1..l6.py`, `falcon.py`, `transitions.py`, `core.py`).
   `python3 scripts/render.py out/master.mp4 --crf 12` takes ~4 min on 4 cores. `preview.py <scene> t1 t2 …`
   renders contact sheets.

Audio reactivity: bass drives the zoom punch, the ring behind the subject and the road-dash glow. Snare
(mid onsets) drives the flashes. Hi-hat (high onsets) drives the star glints. Line ends drive the glitch.
A gold waveform ribbon runs along the bottom.

## Scenes
| Line | Time | Idea |
|---|---|---|
| L1 | 0–3.68 | Maroon + moving phulkari. Torn-paper cut-out of the subject; giant words stack **behind** it and punch on beats. VYAHUNA: a marigold/gold bead garland traces the word, then snaps. SHASTAR: a blade-line slash, the halves slide apart, sparks. |
| L2 | 3.68–7.30 | Frosted-glass "now playing" card tilting in 3D and scrolling. Karaoke fill sweeps each word exactly over its sung start→end. BAAZAN: an original falcon silhouette launches and sheds feathers… |
| L3 | 7.30–10.86 | …which fly into RAHAN. The footage becomes the sky over a perspective road through phulkari fields; words race toward camera. YAARAN: ink outlines draw themselves around the people. KASS KE: tracking squeezes, the word thickens, shakes, rubber-band snap on the beat. |
| L4 | 10.86–14.62 | Venom grade. The line slithers along a moving S-path; the head word glows green-black and drips venom. ZEHRI NAAG ripples with a scale-skin fill. DASS KE: glitch + RGB split. A maroon dupatta sweep reveals L5. |
| L5 | 14.62–18.20 | Footage strip between gold-trimmed red bands with rotating corner sunbursts. Gold-foil words with light sweeps; star glints on hi-hats. BAABE: calm fade with a breathing halo. LASHKE: lens flare + shimmer wave. |
| L6 | 18.20–21.72 | An iris opens into the footage circle hanging on a chain, swaying ±8° with extremes on the beats; ticks and dots orbit it. JHUMMAR rocks with the same pendulum and leaves an echo per beat. ASHKE lands huge, cracks, and **shatters** on the beat into 1600 gold sparkles with a shockwave. They swirl into the end card. |

## Word timings (edit time; song time = 172.000 + t)
| Line | Word | Start (s) | End (s) | Song time | Snap | 300M vs 1B Δstart |
|---|---|---|---|---|---|---|
| L1 | BEBE | 0.210 | 0.550 | 2:52.210 | +29 ms | +0 ms |
| L1 | KEHNDI | 0.630 | 0.990 | 2:52.630 | +8 ms | +20 ms |
| L1 | TAINU | 1.100 | 1.263 | 2:53.100 | +17 ms | +0 ms |
| L1 | VYAHUNA | 1.280 | 1.700 | 2:53.280 | -23 ms | +20 ms |
| L1 | TE | 1.770 | 1.960 | 2:53.770 | +27 ms | +20 ms |
| L1 | MERA | 2.000 | 2.190 | 2:54.000 | +36 ms | +20 ms |
| L1 | SHASTAR | 2.220 | 2.610 | 2:54.220 | +36 ms | +40 ms |
| L1 | VI | 2.690 | 2.840 | 2:54.690 | +25 ms | +0 ms |
| L1 | NA | 2.870 | 3.020 | 2:54.870 | +5 ms | +20 ms |
| L1 | THAKA | 3.090 | 3.560 | 2:55.090 | +5 ms | +20 ms |
| L2 | DASS | 3.760 | 3.940 | 2:55.760 | +14 ms | +0 ms |
| L2 | KI | 3.980 | 4.140 | 2:55.980 | -6 ms | +20 ms |
| L2 | KAR | 4.220 | 4.400 | 2:56.220 | -7 ms | +20 ms |
| L2 | LAINA | 4.440 | 4.830 | 2:56.440 | -27 ms | +20 ms |
| L2 | KAVAN | 4.880 | 5.310 | 2:56.880 | -7 ms | +20 ms |
| L2 | NI | 5.350 | 5.520 | 2:57.350 | +2 ms | +20 ms |
| L2 | MERA | 5.560 | 5.730 | 2:57.560 | +12 ms | +20 ms |
| L2 | BAAZAN | 5.780 | 6.290 | 2:57.780 | -8 ms | +21 ms |
| L2 | AALA | 6.329 | 6.660 | 2:58.329 | — | +60 ms |
| L2 | RAAKHA | 6.700 | 7.160 | 2:58.700 | +10 ms | +40 ms |
| L3 | RAHAN | 7.360 | 7.751 | 2:59.360 | -10 ms | +20 ms |
| L3 | VICH | 7.790 | 8.040 | 2:59.790 | -21 ms | +0 ms |
| L3 | RODE | 8.071 | 8.460 | 3:00.071 | — | +0 ms |
| L3 | BEHNA | 8.710 | 9.060 | 3:00.710 | -2 ms | +0 ms |
| L3 | JE | 9.100 | 9.260 | 3:01.100 | -13 ms | +0 ms |
| L3 | YAARAN | 9.293 | 9.560 | 3:01.293 | — | +0 ms |
| L3 | NAL | 9.600 | 9.773 | 3:01.600 | +7 ms | +20 ms |
| L3 | PAINA | 9.840 | 10.180 | 3:01.840 | +26 ms | +20 ms |
| L3 | KASS | 10.280 | 10.560 | 3:02.280 | +6 ms | +20 ms |
| L3 | KE | 10.650 | 10.880 | 3:02.650 | +15 ms | +0 ms |
| L4 | JATT | 10.920 | 11.075 | 3:02.920 | -15 ms | +0 ms |
| L4 | MUDD | 11.130 | 11.300 | 3:03.130 | -25 ms | +0 ms |
| L4 | TON | 11.340 | 11.580 | 3:03.340 | -55 ms | +21 ms |
| L4 | ZEHRI | 11.616 | 12.030 | 3:03.616 | — | +20 ms |
| L4 | NAAG | 12.320 | 12.630 | 3:04.320 | +23 ms | +20 ms |
| L4 | NI | 12.670 | 12.810 | 3:04.670 | +13 ms | +20 ms |
| L4 | REHA | 12.850 | 13.090 | 3:04.850 | -7 ms | +20 ms |
| L4 | KOI | 13.190 | 13.430 | 3:05.190 | +32 ms | +20 ms |
| L4 | SANU | 13.460 | 13.770 | 3:05.460 | +62 ms | +20 ms |
| L4 | DASS | 13.840 | 14.150 | 3:05.840 | -19 ms | +0 ms |
| L4 | KE | 14.210 | 14.470 | 3:06.210 | -9 ms | +0 ms |
| L5 | BAABE | 14.530 | 14.940 | 3:06.530 | +7 ms | +20 ms |
| L5 | DI | 14.970 | 15.190 | 3:06.970 | +6 ms | +20 ms |
| L5 | SANU | 15.220 | 15.525 | 3:07.220 | +36 ms | +20 ms |
| L5 | BAKSH | 15.650 | 16.040 | 3:07.650 | +25 ms | +20 ms |
| L5 | AE | 16.080 | 16.160 | 3:08.080 | -6 ms | +20 ms |
| L5 | KANN | 16.320 | 16.466 | 3:08.320 | -6 ms | +20 ms |
| L5 | NATTIYAN | 16.550 | 16.940 | 3:08.550 | +24 ms | +0 ms |
| L5 | GAANI | 16.980 | 17.370 | 3:08.980 | -7 ms | +40 ms |
| L5 | LASHKE | 17.410 | 18.020 | 3:09.410 | -18 ms | +20 ms |
| L6 | NI | 18.120 | 18.280 | 3:10.120 | -9 ms | -20 ms |
| L6 | MODHE | 18.320 | 18.680 | 3:10.320 | -10 ms | +0 ms |
| L6 | PAUNDI | 18.790 | 19.190 | 3:10.790 | +20 ms | +21 ms |
| L6 | JHUMMAR | 19.450 | 19.940 | 3:11.450 | -22 ms | +0 ms |
| L6 | DUNAALI | 20.010 | 20.470 | 3:12.010 | -3 ms | +20 ms |
| L6 | KEH | 20.570 | 20.700 | 3:12.570 | +16 ms | +20 ms |
| L6 | KE | 20.740 | 21.050 | 3:12.740 | -54 ms | +40 ms |
| L6 | ASHKE | 21.110 | 21.560 | 3:13.110 | +15 ms | +20 ms |