"""L4 'Jatt mudd ton zehri naag ni reha koi sanu dass ke'
Since the beginning the Jatt is a venomous cobra — nobody is left to bite us.
The line is a serpent: glyphs ride an S-curve that slithers (travelling wave),
the head word glows green-black and drips venom. ZEHRI NAAG ripples with a
scale-skin fill. DASS KE: glitch + RGB split, like a whispered secret."""
import math, functools
import numpy as np, cv2
from .core import *

LINE = 3
SIZE = 116; TRACK = 6
WORDS_L = LINE_WORDS[LINE]


@functools.lru_cache(maxsize=1)
def path_table():
    u = np.linspace(0, 1, 4000)
    P0 = np.array([-330.0, 1600.0]); P1 = np.array([1050.0, 520.0])
    ax = (P1 - P0); L = np.hypot(*ax); ax /= L; nrm = np.array([-ax[1], ax[0]])
    off = 120 * np.sin(2 * math.pi * 1.3 * u + 1.2)
    x = P0[0] + ax[0] * L * u + nrm[0] * off
    y = P0[1] + ax[1] * L * u + nrm[1] * off
    d = np.hypot(np.diff(x), np.diff(y)); s = np.concatenate([[0], np.cumsum(d)])
    return s, x, y


def path_at(s, t):
    S, X, Y = path_table()
    s = np.asarray(s, np.float64)
    x = np.interp(s, S, X); y = np.interp(s, S, Y)
    xd = np.interp(s + 2, S, X) - np.interp(s - 2, S, X); yd = np.interp(s + 2, S, Y) - np.interp(s - 2, S, Y)
    n = np.hypot(xd, yd) + 1e-9; tx, ty = xd / n, yd / n
    nx, ny = -ty, tx
    wob = 20 * np.sin(2 * math.pi * s / 430 - 7.0 * t)
    dwob = 20 * 2 * math.pi / 430 * np.cos(2 * math.pi * s / 430 - 7.0 * t)
    ang = np.arctan2(ty + ny * dwob, tx + nx * dwob)
    return x + nx * wob, y + ny * wob, ang, nx, ny


@functools.lru_cache(maxsize=1)
def chars():
    """per char: (char, word index, offset along string, advance)"""
    out = []; o = 0.0
    f = font('anton', SIZE); sp = f.getlength(' ') + 10
    for wi, w in enumerate(WORDS_L):
        for ch in w['text']:
            adv = f.getlength(ch) + TRACK
            out.append((ch, wi, o + adv / 2, adv)); o += adv
        o += sp
    return out, o


@functools.lru_cache(maxsize=64)
def glyph(ch):
    a, base = text_alpha(ch, 'anton', SIZE)
    return a, base


@functools.lru_cache(maxsize=1)
def scale_tex():
    """procedural snake-scale skin (canvas-size), green-gold."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 15.0
    row = np.floor(yy / r); xo = xx + (row % 2) * r
    cx = (np.floor(xo / (2 * r)) + 0.5) * 2 * r; cy = (row + 1.0) * r
    d = np.sqrt((xo - cx) ** 2 + (yy - cy) ** 2) / r
    edge = np.clip((d - 0.82) / 0.18, 0, 1)
    shade = np.clip(1.15 - d * 0.6, 0.3, 1)
    base = VENOM * 0.55 * shade[..., None] + GOLD * 0.35 * (1 - d.clip(0, 1))[..., None]
    return (base * (1 - 0.75 * edge[..., None])).astype(np.float32)


def string_state(t):
    cs, total = chars()
    k = word_at(LINE, t)
    if k < 0: return [], 0, 0, k
    vis = []
    for ch, wi, o, adv in cs:
        if wi < k: vis.append((ch, wi, o, adv, 1.0))
        elif wi == k:
            w = WORDS_L[wi]
            letters = [c for c in cs if c[1] == wi]
            j = letters.index((ch, wi, o, adv))
            appear = w['start'] + min(0.16, (w['end'] - w['start']) * 0.6) * j / max(1, len(letters))
            if t >= appear - 0.5 / FPS: vis.append((ch, wi, o, adv, clamp01((t - appear) / 0.04 + 0.6)))
    if not vis: return [], 0, 0, k
    o_end = vis[-1][2] + vis[-1][3] / 2
    S, _, _ = path_table(); Lp = S[-1]
    lp = prog(t, WORDS_L[0]['start'], WORDS_L[-1]['end'])
    s_head = lerp(Lp * 0.60, Lp * 0.94, lp)
    return vis, o_end, s_head, k


def drips(c, t):
    """venom drops shed from the head as the snake moves."""
    rng = np.random.default_rng(9)
    n = 46
    births = np.sort(rng.uniform(WORDS_L[0]['start'] + 0.1, WORDS_L[-1]['end'], n))
    offs = rng.uniform(10, 160, n); g = rng.uniform(900, 1500, n)
    for i in range(n):
        b = births[i]; age = t - b
        if age < 0 or age > 0.9: continue
        vis, o_end, s_head, k = string_state(b)
        if not vis: continue
        x0, y0, ang, nx, ny = path_at(s_head - offs[i], b)
        x0 = float(x0) + float(nx) * 40; y0 = float(y0) + float(ny) * 40
        y = y0 + 0.5 * g[i] * age * age + 30 * age
        op = clamp01(1 - age / 0.9)
        L = 10 + 40 * min(1, age * 3)
        draw_polyline(c, [(x0, y0), (x0, y - 6)], VENOM * 0.6, 2, 0.45 * op)
        a = np.zeros((int(L + 20), 26), np.float32)
        cv2.ellipse(a, (13, int(L + 6)), (7, 9), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        pts = np.array([[13, 2], [6, int(L + 4)], [20, int(L + 4)]], np.int32)
        cv2.fillPoly(a, [pts], 1.0, cv2.LINE_AA)
        ga, gbb = draw_sprite(c, a, M_affine(x0, y, 1, 1, 0, 13, L + 6), VENOMDK, op)
        glow(c, ga, gbb, VENOM, 6, 0.9 * op)


def render(t, fx):
    F = footage(); f = t - FOOT_OFF
    img = F.frame(f); cx, cy, fs = F.subject(f)
    oy = H * 0.42 + 20 * math.sin(t * 0.5)
    bg, M = fullbleed(img, cx, cy, 1.45, oy)
    bgg = grade(bg, 0.45, 1.15, -0.03, VENOM, 0.22) * 0.78
    c = bgg.copy()
    # bass ring behind subject, then re-paste subject
    hx, hy = M @ np.array([cx, cy, 1.0]); b = env('bass', t)
    th = np.linspace(0, 2 * math.pi, 200); R = fs * 1.45 * 1.25 + 70 * b
    draw_polyline(c, np.stack([hx + R * np.cos(th), hy + R * np.sin(th)], 1), VENOM, 3, 0.2 + 0.5 * b, 'add', True, glow_r=10)
    m = warp_full(F.mask(f, True, 2.2), M)
    c[:] = c * (1 - m[..., None]) + bgg * m[..., None]
    c *= vignette_map(0.6)

    vis, o_end, s_head, k = string_state(t)
    if not vis: return c
    tex = scale_tex()
    for ch, wi, o, adv, op in vis:
        s = s_head - (o_end - o)
        if s < -40: continue
        op *= clamp01((s + 40) / 160)
        x, y, ang, nx, ny = path_at(s, t)
        x, y, ang, nx, ny = float(x), float(y), float(ang), float(nx), float(ny)
        w = WORDS_L[wi]; txt = w['text']
        if txt in ('ZEHRI', 'NAAG') and t >= w['start']:
            amp = 18 * (0.4 + 0.6 * math.exp(-(t - w['start']) / 0.8))
            r = amp * math.sin(2 * math.pi * s / 95 - 11 * t)
            x += nx * r; y += ny * r
        a, base = glyph(ch)
        M = M_affine(x, y, 1, 1, math.degrees(ang), a.shape[1] / 2, base - SIZE * 0.36)
        is_head = wi == k
        if txt in ('DASS', 'KE') and wi >= 9:
            rng = np.random.default_rng(int(t * FPS) * 31 + wi)
            sp = 9 + 6 * rng.random()
            for col, dx in ((np.array([1, 0.1, 0.15], np.float32), -sp), (np.array([0.1, 1, 0.5], np.float32), 0), (np.array([0.2, 0.4, 1], np.float32), sp)):
                Mg = M.copy(); Mg[0, 2] += dx + rng.normal(0, 3); Mg[1, 2] += rng.normal(0, 2)
                draw_sprite(c, a, Mg, col, 0.85 * op, 'add')
            continue
        if txt in ('ZEHRI', 'NAAG'):
            fill = lambda bb, aa: tex[bb[1]:bb[3], bb[0]:bb[2]]
            ga, gbb = draw_sprite(c, a, M, fill, op)
            glow(c, ga, gbb, VENOM, 10, 0.35 * op)
            if ga is not None:
                comp(c, np.clip(ga - cv2.erode(ga, np.ones((5, 5), np.uint8)), 0, 1), gbb, CREAM, 0.85 * op)
        elif is_head:
            ga, gbb = draw_sprite(c, a, M, VENOMDK, op)
            glow(c, ga, gbb, VENOM, 16, 1.1 * op)
            comp(c, np.clip(ga - cv2.erode(ga, np.ones((3, 3), np.uint8)), 0, 1), gbb, VENOM, op)
        else:
            draw_sprite(c, a, M, CREAM, op)
    drips(c, t)
    dk = WORDS_L[9]
    if dk['start'] <= t < WORDS_L[10]['end']:
        fx['glitch'] = max(fx.get('glitch', 0), 0.6 * pulse(np.array([dk['start'], WORDS_L[10]['start']]), t, 0.12) + 0.15)
    return c
