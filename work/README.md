# Karan Aujla beat-synced edit — pipeline

Source: the screen recording `ScreenRecording_10-09-2026_16-34-52_1.mov` (song 2:50–3:06 plus the
music video playing inside it). It is not committed; pass its path to `build_source.py`.

```
ffmpeg -i REC.mov -vn -ac 2 -ar 48000 -c:a pcm_s24le full_audio.wav
python3 analyze.py 0          # segment.wav (0.15 s fade-in / 0.4 s fade-out), beats.json, timeline table
python3 footage_scan.py       # per-shot motion / sharpness / brightness  -> footage.json (needs source_crop.mp4)
python3 build_source.py REC.mov   # subject-centred 9:16 crop + dedupe + minterpolate to 120 fps -> src120.mp4
python3 edl.py                # shotlist.md (all cuts in beats -> frames)
python3 render.py ../output/final_edit.mp4
python3 qc.py                 # ffprobe + cut / sync / black / flicker checks

# CRAZY cut (54 cuts, same audio / grid / footage)
python3 edl_crazy.py          # shotlist_crazy.md
python3 render.py ../output/crazy_edit.mp4 --edl edl_crazy
python3 qc.py ../output/crazy_edit.mp4 edl_crazy
```

To change the end-card handle, edit `END_HANDLE` in `edl.py` and re-run `render.py`. It takes about 2.5 min.
