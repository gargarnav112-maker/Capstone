"""Step 5 QC: container/stream checks, measured cut positions vs beat grid, black-frame
and flicker scan, frame dumps at every cut + 0/4/8/12/15.9 s."""
import json
import subprocess
import sys

import cv2
import numpy as np

import importlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "../output/final_edit.mp4"
edl = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "edl")


def probe(args):
    return subprocess.run(["ffprobe", "-v", "error", *args, OUT], capture_output=True, text=True).stdout


info = json.loads(probe(["-show_entries", "format=duration,bit_rate:stream=codec_type,codec_name,width,height,"
                                         "r_frame_rate,nb_frames,duration,start_time,bit_rate,sample_rate,pix_fmt,profile",
                         "-of", "json"]))
ok = True
print("format duration:", info["format"]["duration"], " total bitrate:", int(info["format"]["bit_rate"]) // 1000, "kb/s")
for s in info["streams"]:
    print(" ", {k: s.get(k) for k in ("codec_type", "codec_name", "profile", "width", "height", "pix_fmt", "r_frame_rate",
                                       "nb_frames", "start_time", "duration", "bit_rate", "sample_rate")})
v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
a = [s for s in info["streams"] if s["codec_type"] == "audio"][0]
checks = {
    "video 1080x1920": (v["width"], v["height"]) == (1080, 1920),
    "30 fps": v["r_frame_rate"] == "30/1",
    "480 frames": int(v["nb_frames"]) == 480,
    "video duration 16.000": abs(float(v["duration"]) - 16.0) < 1e-3,
    "audio duration 16.000 (±1 AAC frame)": abs(float(a["duration"]) - 16.0) <= 1024 / 48000 + 1e-3,
    "A/V start aligned": abs(float(v["start_time"]) - float(a["start_time"])) < 1e-3,
}

# --- audio identical to segment.wav? (correlation of the decoded track vs the source)
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT, "-map", "0:a", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                     capture_output=True).stdout
ref = subprocess.run(["ffmpeg", "-v", "error", "-i", "segment.wav", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                     capture_output=True).stdout
x, y = np.frombuffer(dec, np.float32), np.frombuffer(ref, np.float32)
nn = min(len(x), len(y))
xc = np.correlate(x[:nn][4000:20000], y[:nn][4000 - 400:20000 + 400], mode="valid")
lag = int(np.argmax(xc)) - 400
checks[f"audio lag vs segment.wav = {lag/8:.2f} ms (<1 frame)"] = abs(lag / 8000) < 1 / 60

# --- measured cuts
cap = cv2.VideoCapture(OUT)
frames = []
while True:
    r, f = cap.read()
    if not r:
        break
    frames.append(cv2.resize(f, (135, 240), interpolation=cv2.INTER_AREA))
F = np.stack(frames).astype(np.float32)
diff = np.abs(F[1:] - F[:-1]).mean((1, 2, 3))
luma = F.mean((1, 2, 3))
planned = edl.CUTS
measured = []
print("\ncut  planned_frame  beat_time  frame_time  err_ms  measured_change(frame: diff)")
for c in planned:
    win = range(max(1, c - 3), min(len(F) - 1, c + 4))
    m = max(win, key=lambda j: diff[j - 1])
    measured.append(m)
    beat_t = None
    for s in edl.S:
        if s["f0"] == c:
            beat_t = edl.bt(s["a"])
    err = (c / 30 - beat_t) * 1000
    print(f"     {c:5d}        {beat_t:7.3f}    {c/30:7.3f}   {err:+6.1f}   {m}: {diff[m-1]:.1f}{'  <-- MISMATCH' if m != c else ''}")
# a cut is "on its frame" when the planned frame carries a hard discontinuity; on dense
# cuts (quarter-beats, whips, flashes) the window-argmax above can lock onto a neighbour.
inshot = np.median([diff[i - 1] for i in range(1, len(F)) if i not in set(planned)])
weak = [c for c in planned if diff[c - 1] < 3 * inshot]
print(f"\nmin discontinuity at planned cuts {min(diff[c-1] for c in planned):.1f}, in-shot median {inshot:.1f}")
checks[f"hard cut present on every planned frame (>=3x in-shot median) weak={weak}"] = not weak
checks["all cuts within ½ frame of beat grid"] = all(abs(c / 30 - edl.bt(s["a"])) <= 1 / 60 + 1e-9
                                                    for c, s in zip(planned, edl.S[1:]))

# --- black / flicker scan (exclude intended fade-in, flashes and end-card fade)
intended_dark = set(range(0, edl.bf(1))) | set(range(edl.S[-1]["f0"], edl.NFRAMES))  # fade-up / end card
black = [i for i in range(len(F)) if luma[i] < 8 and i not in intended_dark]
checks["no unintended black frames"] = not black
# flicker: a single frame whose luma departs from both neighbours by > 25 while they agree
flick = [i for i in range(1, len(F) - 1)
         if abs(luma[i] - luma[i - 1]) > 25 and abs(luma[i] - luma[i + 1]) > 25 and abs(luma[i - 1] - luma[i + 1]) < 8]
flash_frames = {s["f0"] + k for s in edl.S for k in range(3) if any(f.startswith("flash") or f == "invert1" for f in s["fx"])}
flick = [i for i in flick if i not in flash_frames]
checks[f"no isolated flicker frames {flick}"] = not flick
checks["first frame black/clean"] = luma[0] < 3
checks["last frame black/clean"] = luma[-1] < 3

print()
for k, val in checks.items():
    print(("PASS " if val else "FAIL ") + k)
    ok &= bool(val)

# --- frame dumps
dump = sorted(set(planned + [0, 120, 240, 360, 477]))
for n in dump:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", OUT, "-vf", f"select=eq(n\\,{n})", "-frames:v", "1",
                    f"qc_frames/{edl.__name__}_f{n:03d}.jpg"])
print("\nOVERALL:", "PASS" if ok else "FAIL")
