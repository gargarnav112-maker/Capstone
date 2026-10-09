"""Footage access: recording frames, VFR timestamps, per-shot subject centre (source px in the 2140x1068 de-letterboxed frame)."""
import cv2, numpy as np, functools, os
W, H = 2140, 1068
SEG_TO_REC = -2.304  # rec_t = seg_t + SEG_TO_REC (audio xcorr, ncc 0.98)
D = os.path.dirname(os.path.abspath(__file__)) + '/../'
PTS = np.array([float(x.strip().strip(',')) for x in open(D + 'frames/pts.txt') if x.strip()])
CUTS = [0.0, 2.55, 3.5, 3.983, 4.35, 4.983, 6.833, 13.668, 99]
# keyframes (rec_t, subject_x, face_y) per shot, eyeballed from gridded sheet
KEYS = [[(0.0, 1290, 330), (1.5, 1200, 340), (2.55, 1100, 350)],
        [(2.55, 1080, 230), (3.5, 1030, 280)],
        [(3.5, 1560, 230)],
        [(3.983, 1200, 220)],
        [(4.35, 1290, 330)],
        [(4.983, 1750, 260), (6.833, 1620, 260)],
        [(6.833, 1070, 300), (8.6, 1100, 300), (9.2, 1155, 330), (13.668, 1155, 330)],
        [(13.668, 1200, 420), (16.6, 1290, 440)]]

def shot_of(rt):
    return int(np.searchsorted(CUTS, rt, side='right') - 1)

def subject(rt):
    k = KEYS[min(shot_of(rt), len(KEYS) - 1)]
    ts = [a[0] for a in k]
    return float(np.interp(rt, ts, [a[1] for a in k])), float(np.interp(rt, ts, [a[2] for a in k]))

def idx_at(rt):
    """index of frame displayed at rec time rt (sample-and-hold, VFR aware)"""
    return int(np.clip(np.searchsorted(PTS, rt + 1e-4, side='right') - 1, 0, len(PTS) - 1))

@functools.lru_cache(maxsize=48)
def frame(i):
    return cv2.imread(D + f'frames/{i + 1:04d}.jpg')

def frame_at(rt, blend=False):
    """blend=True: linear frame blending between neighbouring source frames (for speed ramps / slow-mo)."""
    i = idx_at(rt)
    if not blend or i + 1 >= len(PTS) or shot_of(PTS[i]) != shot_of(PTS[i + 1]):
        return frame(i)
    a = (rt - PTS[i]) / max(PTS[i + 1] - PTS[i], 1e-3)
    return cv2.addWeighted(frame(i), 1 - a, frame(i + 1), a, 0)

@functools.lru_cache(maxsize=48)
def mask(i):
    p = D + f'masks/{i + 1:04d}.png'
    return cv2.imread(p, 0) if os.path.exists(p) else None

def mask_at(rt):
    return mask(idx_at(rt))
