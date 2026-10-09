"""L1 'Bebe kehndi tainu vyahuna te mera shastar vi na thaka'
Mother wants me married; my arms haven't tired.
Giant cream words stack BEHIND a torn-paper cut-out of the subject on maroon.
VYAHUNA: a marigold/gold bead garland traces around the word, then snaps.
SHASTAR: a blade-line slashes the word diagonally; halves slide apart, sparks."""
import math, functools
import numpy as np, cv2
from .core import *

LINE = 0
PAGES = [[["BEBE", "KEHNDI"], ["TAINU"], ["VYAHUNA"]],
         [["TE", "MERA"], ["SHASTAR"], ["VI", "NA", "THAKA"]]]
PAGE_SWITCH = LINE_WORDS[0][4]['start'] - 0.5 / FPS     # 'TE'
TOP = 215; MAXW = 1000; GAP = 26


@functools.lru_cache(maxsize=1)
def layout():
    """per word: page, sprite size, canvas centre (cx, cy)."""
    ws = LINE_WORDS[LINE]; k = 0; out = []
    for p, page in enumerate(PAGES):
        y = TOP
        for row in page:
            txt = ' '.join(row)
            size = fit_size(txt, 'anton', MAXW, 370)
            f = font('anton', size); sp = f.getlength(' ')
            widths = [f.getlength(r) for r in row]
            total = sum(widths) + sp * (len(row) - 1)
            x = (W - total) / 2
            asc = f.getbbox('H')[1]; cap = f.getbbox('H')[3] - asc
            for r, wdt in zip(row, widths):
                assert ws[k]['text'] == r, (ws[k]['text'], r)
                out.append(dict(w=ws[k], page=p, size=size, cx=x + wdt / 2, cy=y + cap / 2, cap=cap))
                x += wdt + sp; k += 1
            y += cap + GAP
    return out


@functools.lru_cache(maxsize=1)
def torn_noise():
    n1 = noise2d(H, W, 9, 11); n2 = noise2d(H, W, 70, 12)
    return (0.6 * n1 + 0.4 * n2).astype(np.float32)


def paper_tex():
    return CREAM * 0.97


def subject_layer(t):
    """returns (rgb canvas-size footage, alpha cut-out, head canvas pos, M)"""
    F = footage(); f = max(0.0, t - FOOT_OFF)
    img = F.frame(f); cx, cy, fs = F.subject(f)
    m = F.mask(f, 'cc')
    scale = float(np.clip(235 / fs, 0.9, 1.5))
    oy = H + 12 - (F.AH - cy) * scale
    M = foot_M(cx, cy, scale, W / 2 + 10, oy)
    # parallax: subject drifts slower than text
    M[0, 2] += 10 * math.sin(t * 0.9); M[1, 2] += 6 * math.sin(t * 0.7)
    rgbl = warp_full(img, M); a = warp_full(m, M)
    return rgbl, a, (W / 2 + 10, oy), M, scale * fs


def draw_word(canvas, L, t, alpha_override=None, offset=(0, 0), extra_scale=1.0):
    w = L['w']; a, _ = text_alpha(w['text'], 'anton', L['size'])
    age = t - w['start']
    s = lerp(1.32, 1.0, ease_out(age / 0.14)) * extra_scale
    bp = pulse(BEATS, t, 0.12)
    s *= 1 + 0.035 * bp
    op = clamp01(age / 0.05) if alpha_override is None else alpha_override
    h, wd = a.shape
    M = M_affine(L['cx'] + offset[0], L['cy'] + offset[1], s, s, 0, wd / 2, h / 2)
    col = lerp(PALEGOLD * 1.1, CREAM, ease_out(age / 0.2))
    return draw_sprite(canvas, a, M, col, op)


def bead(canvas, x, y, r, col, op=1.0):
    x0, y0 = int(x - r - 2), int(y - r - 2); x1, y1 = int(x + r + 3), int(y + r + 3)
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H: return
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    a = np.clip(r - d + 0.5, 0, 1)
    shade = np.clip(1.15 - 0.6 * ((xx - x + r * 0.3) ** 2 + (yy - y + r * 0.3) ** 2) / (r * r + 1e-3), 0.45, 1.2)
    comp(canvas, a, (X0, Y0, X1, Y1), col * shade[..., None], op)


def garland(canvas, L, t):
    """VYAHUNA: beads trace a stadium path round the word, then break and fall."""
    w = L['w']; a, _ = text_alpha(w['text'], 'anton', L['size'])
    hw, hh = a.shape[1] / 2 + 34, a.shape[0] / 2 + 22
    cx, cy = L['cx'], L['cy']
    # stadium path param (start at top centre, clockwise)
    N = 64; pts = []
    r = hh; straight = max(hw - r, 1)
    per = 4 * straight + 2 * math.pi * r
    for i in range(N):
        s = per * i / N
        if s < straight: p = (cx + s, cy - r)
        elif s < straight + math.pi * r:
            ang = -math.pi / 2 + (s - straight) / r; p = (cx + straight + r * math.cos(ang), cy + r * math.sin(ang))
        elif s < 3 * straight + math.pi * r:
            p = (cx + straight - (s - straight - math.pi * r), cy + r)
        elif s < 3 * straight + 2 * math.pi * r:
            ang = math.pi / 2 + (s - 3 * straight - math.pi * r) / r; p = (cx - straight + r * math.cos(ang), cy + r * math.sin(ang))
        else: p = (cx - straight + (s - 3 * straight - 2 * math.pi * r), cy - r)
        pts.append(p)
    t_draw0, t_draw1 = w['start'] + 0.02, w['start'] + 0.30
    t_break = w['start'] + 0.33
    shown = int(N * ease_io(prog(t, t_draw0, t_draw1)))
    rng = np.random.default_rng(5)
    vx = rng.uniform(-260, 260, N); vy = rng.uniform(-420, -60, N)
    for i in range(N):
        if i >= shown: break
        x, y = pts[i]
        if t > t_break:
            dt = t - t_break
            x += vx[i] * dt; y += vy[i] * dt + 0.5 * 2600 * dt * dt
            op = clamp01(1 - dt / 0.55)
        else: op = 1.0
        if op <= 0: continue
        kind = i % 3
        if kind == 0: bead(canvas, x, y, 11, SAFFRON, op); bead(canvas, x - 2, y - 2, 4, rgb('#b8560b'), op * 0.8)
        elif kind == 1: bead(canvas, x, y, 6.5, GOLD * 1.15, op)
        else: bead(canvas, x, y, 4.5, RED, op)
    # glint travelling along the string while it draws
    if t_draw0 < t < t_break and shown > 0:
        x, y = pts[shown - 1]
        st = star_sprite(80)
        draw_sprite(canvas, st, M_affine(x, y, 1.2, 1.2, 25 * t * 10, 40, 40), PALEGOLD, 0.9, 'add')


@functools.lru_cache(maxsize=4)
def split_halves(text, size, ang_deg):
    a, _ = text_alpha(text, 'anton', size)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = -math.sin(math.radians(ang_deg)), math.cos(math.radians(ang_deg))
    d = (xx - w / 2) * nx + (yy - h / 2) * ny
    upper = np.clip(-d + 0.5, 0, 1); lower = 1 - upper
    return a * upper, a * lower


def shastar(canvas, L, t, fx):
    """slash + split + sparks. fx collects flash amount."""
    w = L['w']; a, _ = text_alpha(w['text'], 'anton', L['size'])
    h, wd = a.shape; ANG = -17.0
    t_blade0 = w['start'] + 0.07; t_cut = w['start'] + 0.16
    age = t - w['start']
    s = lerp(1.32, 1.0, ease_out(age / 0.14)) * (1 + 0.035 * pulse(BEATS, t, 0.12))
    nx, ny = -math.sin(math.radians(ANG)), math.cos(math.radians(ANG))
    tx, ty = math.cos(math.radians(ANG)), math.sin(math.radians(ANG))
    if t < t_cut:
        M = M_affine(L['cx'], L['cy'], s, s, 0, wd / 2, h / 2)
        draw_sprite(canvas, a, M, CREAM, clamp01(age / 0.05))
    else:
        u = ease_out(prog(t, t_cut, t_cut + 0.3), 4)
        up, lo = split_halves(w['text'], L['size'], ANG)
        sep = 30 * u; slide = 26 * u
        for half, sg in ((up, -1), (lo, 1)):
            M = M_affine(L['cx'] + sg * (nx * sep + tx * slide), L['cy'] + sg * (ny * sep + ty * slide), s, s, sg * 1.5 * u, wd / 2, h / 2)
            draw_sprite(canvas, half, M, CREAM, 1.0)
    # blade line
    if t_blade0 <= t < t_cut + 0.16:
        u = prog(t, t_blade0, t_cut)
        L_len = wd * 0.75 + 140
        x0 = L['cx'] - tx * L_len; y0 = L['cy'] - ty * L_len
        x1 = x0 + tx * 2 * L_len * u; y1 = y0 + ty * 2 * L_len * u
        fade = 1 - prog(t, t_cut, t_cut + 0.16)
        draw_polyline(canvas, [(x0 + tx * 2 * L_len * max(0, u - 0.45), y0 + ty * 2 * L_len * max(0, u - 0.45)), (x1, y1)],
                      PALEGOLD * 1.3, 4, fade, 'add', glow_r=10)
        draw_polyline(canvas, [(x1 - tx * 60, y1 - ty * 60), (x1, y1)], WHITE * 1.5, 6, fade, 'add', glow_r=16)
    # sparks
    if t >= t_cut:
        dt = t - t_cut
        if dt < 0.6:
            rng = np.random.default_rng(42); n = 160
            along = rng.uniform(-wd * 0.45, wd * 0.45, n)
            px = L['cx'] + tx * along; py = L['cy'] + ty * along
            sp = rng.uniform(250, 1100, n); side = np.where(rng.random(n) < 0.5, -1, 1)
            ang = np.arctan2(ny * side, nx * side) + rng.normal(0, 0.6, n)
            x = px + np.cos(ang) * sp * dt; y = py + np.sin(ang) * sp * dt + 0.5 * 1800 * dt * dt
            life = rng.uniform(0.2, 0.6, n); v = np.clip(1 - dt / life, 0, 1) ** 1.5
            cols = np.where(rng.random(n)[:, None] < 0.5, SAFFRON, PALEGOLD)[:, :] * 1.0
            for k in range(3):   # short motion streaks
                splat(canvas, x - np.cos(ang) * sp * 0.004 * k, y - np.sin(ang) * sp * 0.004 * k, cols, v * 0.9 * (1 - k / 3), 1.1)
        fx['flash'] = max(fx.get('flash', 0), 0.3 * math.exp(-dt / 0.07))


def render(t, fx):
    c = radial_gradient(rgb('#661120'), DEEP, W / 2, H * 0.42, 1500).copy()
    phulkari(c, t, 0.085)
    F_rgb, F_a, head, M, face_px = subject_layer(t)
    # bass ring behind subject
    b = env('bass', t)
    ring_r = face_px * 1.15 + 60 * b
    th = np.linspace(0, 2 * math.pi, 180)
    hx, hy = M @ np.array([*footage().subject(max(0, t - FOOT_OFF))[:2], 1.0])
    draw_polyline(c, np.stack([hx + ring_r * np.cos(th), hy + ring_r * np.sin(th)], 1), GOLD, 3, 0.18 + 0.5 * b, 'add', True, glow_r=8)

    # words (behind subject)
    page = 0 if t < PAGE_SWITCH else 1
    for L in layout():
        if L['page'] != page or not visible(L['w'], t): continue
        if L['w']['text'] == 'SHASTAR': shastar(c, L, t, fx); continue
        draw_word(c, L, t)
    for L in layout():   # garland lives in page A but its beads keep falling across the switch
        if L['w']['text'] == 'VYAHUNA' and visible(L['w'], t) and t < PAGE_SWITCH + 0.6:
            garland(c, L, t)

    # torn paper + shadow + subject
    n = torn_noise()
    soft = cv2.GaussianBlur(F_a, (0, 0), 11)
    band = np.clip(soft / 0.04, 0, 1)
    paper = np.clip((soft - 0.16 + (n - 0.5) * 0.32) / 0.04, 0, 1) * band
    paper = np.maximum(paper, F_a)
    sh = cv2.GaussianBlur(paper, (0, 0), 14)
    sh = np.roll(np.roll(sh, 14, 0), 10, 1)
    c *= (1 - 0.55 * sh[..., None])
    pt = CREAM * (0.9 + 0.1 * n[..., None])
    c[:] = c * (1 - paper[..., None]) + pt * paper[..., None]
    sub = grade(F_rgb, 1.08, 1.06, 0.0, SAFFRON, 0.08)
    c[:] = c * (1 - F_a[..., None]) + sub * F_a[..., None]
    fx['bass_zoom'] = 1.0
    return c
