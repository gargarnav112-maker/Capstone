# Banda Bamb — Instagram Reel (1:51–2:20)

`out/final_reel.mp4` is 1080x1920, 30 fps, H.264 High, yuv420p, CRF 17, with AAC 256k audio. It runs exactly 29.000 s and uses +faststart.
`out/cover.png` is the cover image: frame 0, the hero shot with the title (see the note in the cover section).

## Render

```bash
cd /home/claude/banda_bamb_edit
python3 -m venv .venv && . .venv/bin/activate
pip install librosa scipy numpy opencv-python-headless scenedetect mediapipe faster-whisper pillow soundfile
# (MediaPipe needs libegl1/libgles2: apt-get install -y libegl1 libgles2)

scripts/01_ingest.sh                 # probe, cropdetect, crop to the 2140x1204 video rect, 60 fps raw pool, trim audio + fades
python scripts/02_analyze.py         # PySceneDetect shot list + contact sheets, librosa beats/onsets/bass -> data/beats.json
python scripts/04_vocal_map.py       # vocal isolation (HPSS + REPET-SIM) -> syllable onsets / phrases
python scripts/05_track.py           # MediaPipe face + pose tracking, per-shot eased smoothing -> data/track.json
python scripts/06_plan.py            # constant-tempo beat grid, edit decision list, lyric timing -> data/edit_plan.json
python scripts/render.py --scale 0.5 --out qa/preview_540.mp4          # 540p preview (~1 min)
python scripts/render.py --scale 0.5 --debug --out qa/debug_beatgrid_540.mp4   # beat grid / word timing / safe-zone overlay
python scripts/render.py --scale 1.0 --out out/final_reel.mp4          # final (~6 min on 4 cores)
python scripts/07_qa.py out/final_reel.mp4                             # cut-vs-beat table, label timing, ffprobe, clipping check
ffmpeg -y -i out/final_reel.mp4 -frames:v 1 out/cover.png
```

After changing `config.json` (HANDLE, fonts, colours, effect intensities 0–1, grade), re-run only `06_plan.py` and `render.py`.

## Adding the lyrics

Put `lyrics.txt` (18 raw lines) at `data/lyrics.txt` or at the project root, then re-run `06_plan.py` and `render.py`.
The planner merges the raw lines into the 10 lyric moments, using syllable counts to fit each line to its moment's length. It then snaps each word onto a vocal onset and writes `data/lyric_timing.json`. The renderer shows the words in Roman script exactly as written, in each moment's style (slam, ring, burn, whip, bounce, dash, sticker, stomp, mask, map).

## Layout

- `scripts/`: all code.
- `media/`: the derived media (cropped mezzanine, raw frame pool, trimmed audio).
- `data/`: JSON analysis files.
- `qa/`: previews, contact sheets and the debug render.
- `out/`: the deliverables.
- `assets/`: fonts (Anton, Bebas Neue, Oswald, Space Mono, all OFL) and MediaPipe models.

## Notes

- The footage supplied was a **14.1 s** landscape screen recording with **no audio track**, not the 163.6 s portrait recording described in the brief. Its real video rectangle is 2140x1204, with only side bars and no phone UI. Because it has no audio, audio cross-correlation was impossible, so the 14 s of shots serve as a pool cut to the music.
- **Whisper** could not download model weights, because Hugging Face and Azure are blocked by this environment's egress policy. `lyrics.txt` was also not supplied. No lyric words are rendered. On-screen type is limited to the title, the artist, the map regions named in the brief, and HUD labels.
- **Cover:** the frame at 3 s falls on a motion-blurred handheld move, so the cover is frame 0, the hero shot with the title.
