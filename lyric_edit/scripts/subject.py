"""Track the main subject (Karan) per shot from raw face detections and write
subject.json: per source frame, face centre/size in ACTIVE-crop pixel coords."""
import json, numpy as np
from scipy.ndimage import gaussian_filter1d

AX, AY, AW, AH = 240, 2, 2140, 1068   # active picture inside the cropdetect crop
d = json.load(open('faces_raw.json'))
pts = np.array(d['pts']); shot = np.array(d['shot']); N = len(pts)
faces = [[(f[0] + f[2] / 2 - AX, f[1] + f[3] / 2 - AY, f[2], f[4]) for f in fr
          if f[4] >= 0.45 and f[2] > 40] for fr in d['faces']]

cx = np.full(N, np.nan); cy = np.full(N, np.nan); fs = np.full(N, np.nan)
for s in sorted(set(shot)):
    idx = np.where(shot == s)[0]
    if s == 6:   # dolly: Karan sits mid-table; lock on once people are visible
        idx = idx[pts[idx] >= 9.0]
        def init(fr): return min(fr, key=lambda f: abs(f[0] - 0.55 * AW) + abs(f[1] - 0.35 * AH))
    else:
        def init(fr): return max(fr, key=lambda f: f[2] * f[2] * f[3])
    cur = None
    for i in idx:
        fr = faces[i]
        if not fr: continue
        if cur is None:
            cur = init(fr)
        else:
            near = min(fr, key=lambda f: np.hypot(f[0] - cur[0], f[1] - cur[1]))
            if np.hypot(near[0] - cur[0], near[1] - cur[1]) < 0.12 * AW and near[2] > 0.5 * cur[2]:
                cur = near
            else:
                continue
        cx[i], cy[i], fs[i] = cur[0], cur[1], cur[2]

# fill gaps (interp within shot, hold at ends), smooth within shot
for s in sorted(set(shot)):
    idx = np.where(shot == s)[0]
    for a in (cx, cy, fs):
        sel = idx if s != 6 else idx[pts[idx] >= 9.0]
        v = a[sel]; ok = ~np.isnan(v)
        if not ok.any():
            continue
        v = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
        a[sel] = gaussian_filter1d(v, 3.0, mode='nearest')
# shot 6 before Karan appears: table, centre framing
nan = np.isnan(cx); cx[nan] = AW * 0.55; cy[nan] = AH * 0.35; fs[nan] = 140
json.dump(dict(active=[AX, AY, AW, AH], pts=pts.tolist(), shot=shot.tolist(),
               cx=np.round(cx, 1).tolist(), cy=np.round(cy, 1).tolist(), fs=np.round(fs, 1).tolist()),
          open('subject.json', 'w'))
for s in sorted(set(shot)):
    idx = np.where(shot == s)[0]
    print(f'shot {s}: t {pts[idx[0]]:.2f}-{pts[idx[-1]]:.2f}  cx {cx[idx].min():.0f}-{cx[idx].max():.0f}  '
          f'cy {cy[idx].min():.0f}-{cy[idx].max():.0f}  face {fs[idx].min():.0f}-{fs[idx].max():.0f}')
