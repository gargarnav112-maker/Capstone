"""Step 2: per-frame metrics + shot boundaries for the cropped source."""
import json

import cv2
import numpy as np

cap = cv2.VideoCapture("source_crop.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
prev_h = prev_g = None
rows = []
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    small = cv2.resize(f, (534, 266), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    h = cv2.normalize(h, h).flatten()
    hd = 0.0 if prev_h is None else float(cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA))
    motion = 0.0 if prev_g is None else float(np.mean(cv2.absdiff(g, prev_g)))
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    rows.append(dict(f=i, t=round(i / fps, 3), hist=round(hd, 3), motion=round(motion, 2),
                     sharp=round(sharp, 1), bright=round(float(g.mean()), 1)))
    prev_h, prev_g = h, g
    i += 1

hist = np.array([r["hist"] for r in rows])
cuts = [0] + [r["f"] for r in rows if r["hist"] > 0.35]
# merge cuts closer than 4 frames
merged = []
for c in cuts:
    if not merged or c - merged[-1] > 3:
        merged.append(c)
shots = []
for a, b in zip(merged, merged[1:] + [len(rows)]):
    seg = rows[a:b]
    mot = [r["motion"] for r in seg[1:]] or [0]
    shots.append(dict(id=len(shots), f0=a, f1=b - 1, t0=round(a / fps, 3), t1=round(b / fps, 3),
                      dur=round((b - a) / fps, 3),
                      motion=round(float(np.mean(mot)), 2),
                      sharp=round(float(np.median([r["sharp"] for r in seg])), 1),
                      bright=round(float(np.mean([r["bright"] for r in seg])), 1)))
json.dump(dict(frames=rows, shots=shots), open("footage.json", "w"), indent=1)
print(f"{'id':>2} {'t0':>6} {'t1':>6} {'dur':>5} {'motion':>6} {'sharp':>7} {'bright':>6}")
for s in shots:
    print(f"{s['id']:>2} {s['t0']:6.2f} {s['t1']:6.2f} {s['dur']:5.2f} {s['motion']:6.2f} {s['sharp']:7.1f} {s['bright']:6.1f}")
