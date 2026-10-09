"""L5 'Baabe di sanu baksh ae kann nattiyan gaani lashke'
It's a blessing from above — earrings, a glittering necklace.
Letterbox strip of footage between gold-trimmed red bands; slowly rotating
sunbursts in the corners. Gold-foil words; a diagonal light sweep crosses each
word as it lands; 4-point star glints pop on hi-hat onsets. BAABE is calm: slow
fade with a breathing golden halo. LASHKE blazes: lens flare + shimmer wave."""
import math, functools
import numpy as np, cv2
from .core import *

LINE = 4
WS = LINE_WORDS[LINE]
SY0, SY1 = 650, 1270              # strip
ROWS = [(["BAABE"], 300, 270), (["DI", "SANU", "BAKSH", "AE"], 520, 150),
        (["KANN", "NATTIYAN"], 1440, 168), (["GAANI", "LASHKE"], 1680, 230)]


def foot_time(t):
    """real time until 15.6, then slow-motion so the last footage frame lands at the line end."""
    if t <= 15.6: return t - FOOT_OFF, False
    return lerp(15.6 - FOOT_OFF, 16.50, (t - 15.6) / (18.08 - 15.6)), True


@functools.lru_cache(maxsize=1)
def layout():
    out = []; k = 0
    for row, yc, maxs in ROWS:
        txt = ' '.join(row); size = min(maxs, fit_size(txt, 'anton', 960, maxs))
        f = font('anton', size); sp = f.getlength(' ') * 1.1
        widths = [f.getlength(r) for r in row]; total = sum(widths) + sp * (len(row) - 1)
        x = (W - total) / 2
        for r, wd in zip(row, widths):
            assert WS[k]['text'] == r
            out.append(dict(w=WS[k], size=size, cx=x + wd / 2, cy=yc)); x += wd + sp; k += 1
    return out


@functools.lru_cache(maxsize=1)
def bands_static():
    c = np.zeros((H, W, 3), np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    top = BLOOD * (0.55 + 0.45 * (yy / SY0)) ; bot = BLOOD * (0.55 + 0.45 * ((H - yy) / (H - SY1)))
    c[:] = np.where(yy < SY0, top, bot)
    return c


def sunburst(c, cx, cy, t, r=470, n=22, op=0.22):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x0, x1 = int(max(0, cx - r)), int(min(W, cx + r)); y0, y1 = int(max(0, cy - r)), int(min(H, cy + r))
    if x1 <= x0 or y1 <= y0: return
    X = xx[y0:y1, x0:x1] - cx; Y = yy[y0:y1, x0:x1] - cy
    ang = np.arctan2(Y, X) + t * 0.25; d = np.sqrt(X * X + Y * Y)
    rays = (np.cos(ang * n) > 0.35).astype(np.float32)
    rays = cv2.GaussianBlur(rays, (0, 0), 1.0)
    fall = np.clip(1 - d / r, 0, 1) ** 0.8 * np.clip(d / 40, 0, 1)
    comp(c, rays * fall, (x0, y0, x1, y1), GOLD, op)
    ring = np.clip(1 - np.abs(d - r * 0.33) / 3, 0, 1) + np.clip(1 - np.abs(d - r * 0.36) / 1.5, 0, 1)
    comp(c, ring, (x0, y0, x1, y1), PALEGOLD, op * 1.8)


def trim(c, y, flip):
    for dy, th, col, op in ((0, 6, GOLD, 1.0), (14 * flip, 2, PALEGOLD, 0.9), (26 * flip, 1, GOLD, 0.6)):
        yy = int(y + dy); c[max(0, yy - th // 2):yy + (th + 1) // 2] = c[max(0, yy - th // 2):yy + (th + 1) // 2] * (1 - op) + col * op
    yd = int(y + 20 * flip)
    for x in range(12, W, 36):   # small diamond stitches between the trims
        p = np.array([[x, yd - 5], [x + 5, yd], [x, yd + 5], [x - 5, yd]], np.float32)
        draw_polyline(c, p, GOLD, 1, 0.9, closed=True)


def gold_word(c, L, t, fx, calm=False):
    w = L['w']; a, _ = text_alpha(w['text'], 'anton', L['size'])
    h, wd = a.shape
    age = t - w['start']
    if calm:
        s = lerp(0.94, 1.0, ease_out(age / 0.6)); op = ease_io(age / 0.45)
    else:
        s = lerp(1.35, 1.0, ease_back(age / 0.16)); op = clamp01(age / 0.04)
    s *= 1 + 0.025 * pulse(BEATS, t, 0.12)
    ox = h / 2
    src = a
    if w['text'] == 'LASHKE' and age > 0:          # shimmer wave: per-column vertical ripple
        xs = np.arange(wd, dtype=np.float32); ys = np.arange(h, dtype=np.float32)
        amp = 7 * math.exp(-age / 1.2) + 2.5
        mx = np.tile(xs, (h, 1)); my = ys[:, None] + amp * np.sin(xs[None, :] / 34 - age * 14)
        src = cv2.remap(a, mx, my.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    M = M_affine(L['cx'], L['cy'], s, s, 0, wd / 2, ox)
    # drop shadow
    Ms = M.copy(); Ms[0, 2] += 8; Ms[1, 2] += 12
    draw_sprite(c, cv2.GaussianBlur(src, (0, 0), 5), Ms, INK, 0.6 * op)
    sweep = None
    if 0 <= age < 0.35 and not calm: sweep = lerp(-0.2, 1.2, ease_io(age / 0.35))
    if w['text'] == 'LASHKE': sweep = ((age * 1.6) % 1.4) - 0.2 if age > 0 else None
    bright = 1.0 + (0.25 if w['text'] == 'LASHKE' else 0)
    fill = lambda bb, aa: gold_tex(bb[3] - bb[1], bb[2] - bb[0], t, bb[0], bb[1], sweep, 0.12, 0.6, bright)
    ga, gbb = draw_sprite(c, src, M, fill, op)
    edge = np.clip(ga - cv2.erode(ga, np.ones((3, 3), np.uint8)), 0, 1) if ga is not None else None
    if edge is not None: comp(c, edge, gbb, PALEGOLD, 0.7 * op)
    return ga, gbb, M, (wd, h), s


def halo(c, L, t):
    w = L['w']; age = t - w['start']
    if age < 0: return
    a, _ = text_alpha(w['text'], 'anton', L['size'])
    rx, ry = a.shape[1] * 0.85, a.shape[0] * 1.25
    breathe = 0.75 + 0.25 * math.sin(age * 2.6)
    yy, xx = np.mgrid[0:int(ry * 2), 0:int(rx * 2)].astype(np.float32)
    d = ((xx - rx) / rx) ** 2 + ((yy - ry) / ry) ** 2
    g = np.exp(-d * 2.2) * ease_io(age / 0.7) * breathe
    comp(c, g, (int(L['cx'] - rx), int(L['cy'] - ry), int(L['cx'] - rx) + g.shape[1], int(L['cy'] - ry) + g.shape[0]), GOLD, 1.0, 'add')


def lens_flare(c, x, y, t0, t):
    age = t - t0
    if age < 0: return
    k = math.exp(-age / 0.5) * 1.8 + 0.4
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    streak = np.exp(-((yy - y) / 6) ** 2) * np.exp(-np.abs(xx - x) / 700) + 0.25 * np.exp(-((yy - y) / 26) ** 2) * np.exp(-np.abs(xx - x) / 300)
    core = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 38 ** 2))
    c += (streak * 0.9 + core * 1.3)[..., None] * PALEGOLD * k
    # ghosts along the axis through screen centre
    for f_, r_, col in ((-0.4, 60, SAFFRON), (0.35, 30, GOLD), (0.8, 90, RED * 1.5), (1.25, 22, PALEGOLD)):
        gx = W / 2 + (W / 2 - x) * f_; gy = H / 2 + (H / 2 - y) * f_
        d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        ring = np.clip(1 - np.abs(d - r_) / 6, 0, 1) * 0.5 + np.clip(1 - d / r_, 0, 1) * 0.15
        c += ring[..., None] * col * 0.35 * k


def glints(c, t, drawn):
    rng_all = np.random.default_rng(77)
    st = star_sprite(96)
    for j, ot in enumerate(HIGH_ON):
        age = t - ot
        if not (0 <= age < 0.28): continue
        cand = [d for d in drawn if d[0] is not None]
        if not cand: return
        rng = np.random.default_rng(1000 + j)
        for _ in range(2):
            ga, gbb = cand[rng.integers(len(cand))][:2]
            ys, xs = np.nonzero(ga[::4, ::4] > 0.6)
            if not len(xs): continue
            i = rng.integers(len(xs)); x = gbb[0] + xs[i] * 4; y = gbb[1] + ys[i] * 4
            sc = (0.5 + 0.9 * rng.random()) * math.sin(math.pi * min(1, age / 0.28)) ** 0.7
            draw_sprite(c, st, M_affine(x, y, sc, sc, 45 * age, 48, 48), PALEGOLD * 1.4, 1.0, 'add')


def render(t, fx):
    c = bands_static().copy()
    phulkari(c, t, 0.16, speed=(-6, 4), tint=GOLD)
    for cx, cy in ((0, 0), (W, 0), (0, H), (W, H)):
        sunburst(c, cx, cy, t if cx == cy or (cx and cy) else -t)
    # footage strip
    F = footage(); f, slow = foot_time(t)
    img = F.frame(f, interp=slow); cx, cy, fs = F.subject(f)
    sc = 0.86
    vis_w = W / sc; cxx = float(np.clip(cx, vis_w / 2, F.AW - vis_w / 2))
    oy = (SY0 + SY1) / 2 + (cy - (cy + 0.55 * fs)) * sc
    M = foot_M(cxx, cy, sc, W / 2 + 8 * math.sin(t * 0.7), oy)
    strip = grade(warp_full(img, M), 1.1, 1.08, 0.03, GOLD, 0.12, 0.82)
    c[SY0:SY1] = strip[SY0:SY1]
    trim(c, SY0, -1); trim(c, SY1, 1)
    drawn = []
    for L in layout():
        if not visible(L['w'], t): continue
        calm = L['w']['text'] == 'BAABE'
        if calm: halo(c, L, t)
        drawn.append(gold_word(c, L, t, fx, calm))
    glints(c, t, drawn)
    lw = WS[-1]
    if t >= lw['start']:
        L = layout()[-1]; a, _ = text_alpha('LASHKE', 'anton', L['size'])
        lens_flare(c, L['cx'] + a.shape[1] * 0.36, L['cy'] - a.shape[0] * 0.28, lw['start'], t)
        fx['flash'] = max(fx.get('flash', 0), 0.25 * math.exp(-(t - lw['start']) / 0.08))
    return c
