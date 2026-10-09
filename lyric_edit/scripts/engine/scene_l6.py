"""L6 'Ni modhe paundi jhummar dunaali keh ke ashke' + finale + end card.
The footage hangs in a circle that sways like a jhummar dancer (pendulum, ±8°,
extremes landing on the beats); a ring of ticks and dots orbits it. JHUMMAR
rocks with the same pendulum, leaving an echo on every beat. ASHKE lands huge,
then on the next strong beat SHATTERS into gold sparkles with a shockwave; the
sparkles swirl and re-form as the end card (outlined ASHKE + drawn camera icon)."""
import math, functools
import numpy as np, cv2
from .core import *

LINE = 5
WS = LINE_WORDS[LINE]
ASHKE = WS[-1]
T_SHATTER = float(BEATS[np.searchsorted(BEATS, ASHKE['end'])])   # first beat after 'ashke' ends
T_FORM0, T_FORM1 = T_SHATTER + 0.22, T_SHATTER + 0.78
CIRC_R = 410; PIVOT = (W / 2, 40); ARM = 720
ROWS = [(["NI", "MODHE", "PAUNDI"], 1330, 128), (["JHUMMAR"], 1550, 225), (["DUNAALI", "KEH", "KE"], 1765, 128)]


def foot_time(t):
    """b-roll: the golden polo-field shot (footage 4.98-6.80 s) at ~0.5x speed."""
    return lerp(5.0, 6.78, clamp01((t - 18.0) / (T_SHATTER - 18.0)))


def swing(t):
    k, ph = beat_phase(t)
    return -8.0 * math.cos(math.pi * (k + ph))


def circle_center(t):
    th = math.radians(swing(t))
    return PIVOT[0] + ARM * math.sin(th), PIVOT[1] + ARM * math.cos(th)


@functools.lru_cache(maxsize=1)
def layout():
    out = []; k = 0
    for row, yc, maxs in ROWS:
        txt = ' '.join(row); size = min(maxs, fit_size(txt, 'anton', 960, maxs))
        f = font('anton', size); sp = f.getlength(' ') * 1.05
        widths = [f.getlength(r) for r in row]; total = sum(widths) + sp * (len(row) - 1)
        x = (W - total) / 2
        for r, wd in zip(row, widths):
            assert WS[k]['text'] == r
            out.append(dict(w=WS[k], size=size, cx=x + wd / 2, cy=yc)); x += wd + sp; k += 1
    return out


@functools.lru_cache(maxsize=1)
def circle_mask():
    s = 4; R = CIRC_R
    m = np.zeros(((2 * R + 4) * s, (2 * R + 4) * s), np.uint8)
    cv2.circle(m, ((R + 2) * s, (R + 2) * s), R * s, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (2 * R + 4, 2 * R + 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def background(t):
    c = radial_gradient(rgb('#3a0811'), INK, W / 2, 760, 1300).copy()
    phulkari(c, t, 0.075, speed=(5, 8))
    return c


def jhummar_circle(c, t):
    F = footage(); f = foot_time(t)
    img = F.frame(f, interp=True); cx, cy, fs = F.subject(f)
    ox, oy = circle_center(t); th = swing(t)
    sc = 1.32
    M = foot_M(cx, cy + 0.9 * fs, sc, ox, oy, th)
    fr = grade(warp_full(img, M), 1.1, 1.06, 0.02, SAFFRON, 0.1)
    m = circle_mask(); R = CIRC_R
    x0, y0 = int(ox - R - 2), int(oy - R - 2)
    full = np.zeros((H, W), np.float32)
    X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(W, x0 + m.shape[1]), min(H, y0 + m.shape[0])
    full[Y0:Y1, X0:X1] = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    # chain from the pivot (the jhummar hangs)
    tx, ty = ox - R * math.sin(math.radians(th)), oy - R * math.cos(math.radians(th))
    n = 26
    for i in range(n):
        u = i / n; px, py = lerp(PIVOT[0], tx, u), lerp(PIVOT[1], ty, u)
        cv2.ellipse(c, (int(px), int(py)), (5, 9), th, 0, 360, tuple(float(v) for v in GOLD * 0.9), 2, cv2.LINE_AA)
    # bass ring + soft shadow
    b = env('bass', t)
    sh = cv2.GaussianBlur(full, (0, 0), 24)
    c *= (1 - 0.6 * np.roll(sh, 26, 0))[..., None]
    ang = np.linspace(0, 2 * math.pi, 240)
    rr = R + 56 + 46 * b
    draw_polyline(c, np.stack([ox + rr * np.cos(ang), oy + rr * np.sin(ang)], 1), GOLD, 2, 0.15 + 0.5 * b, 'add', True, glow_r=8)
    c[:] = c * (1 - full[..., None]) + fr * full[..., None]
    # rim + orbiting ticks and dots
    draw_polyline(c, np.stack([ox + R * np.cos(ang), oy + R * np.sin(ang)], 1), GOLD, 7, 1.0, 'normal', True)
    draw_polyline(c, np.stack([ox + (R + 12) * np.cos(ang), oy + (R + 12) * np.sin(ang)], 1), PALEGOLD, 2, 0.8, 'normal', True)
    rot = math.radians(t * 22 + th)
    for k in range(72):
        a = rot + 2 * math.pi * k / 72; l = 22 if k % 6 == 0 else 10
        r0 = R + 24
        draw_polyline(c, [(ox + r0 * math.cos(a), oy + r0 * math.sin(a)), (ox + (r0 + l) * math.cos(a), oy + (r0 + l) * math.sin(a))],
                      GOLD, 2, 0.85, 'normal')
    for k in range(12):
        a = -t * 0.9 + 2 * math.pi * k / 12
        x, y = ox + (R + 80) * math.cos(a), oy + (R + 80) * math.sin(a)
        cv2.circle(c, (int(x), int(y)), 6 if k % 3 == 0 else 4, tuple(float(v) for v in (SAFFRON if k % 3 == 0 else CREAM)), -1, cv2.LINE_AA)


def word_pop(c, L, t, col=CREAM, op_mul=1.0):
    w = L['w']; a, _ = text_alpha(w['text'], 'anton', L['size'])
    age = t - w['start']
    s = lerp(1.3, 1.0, ease_back(age / 0.15)) * (1 + 0.03 * pulse(BEATS, t, 0.12))
    M = M_affine(L['cx'], L['cy'], s, s, 0, a.shape[1] / 2, a.shape[0] / 2)
    return draw_sprite(c, a, M, col, clamp01(age / 0.04) * op_mul)


def jhummar(c, L, t, op_mul):
    w = L['w']; a, _ = text_alpha(w['text'], 'anton', L['size'])
    age = t - w['start']; h, wd = a.shape
    piv = (L['cx'], L['cy'] - h * 0.9)
    def M_at(th, s=1.0):
        return M_affine(piv[0], piv[1], s, s, th, wd / 2, h / 2 - h * 0.9)
    # echoes from previous beats
    past = [b for b in BEATS if w['start'] <= b <= t][-4:]
    for j, bt in enumerate(reversed(past)):
        op = [0.38, 0.22, 0.12, 0.06][j] * clamp01((t - bt) / 0.05)
        draw_sprite(c, a, M_at(swing(bt - 1e-3)), SAFFRON, op * op_mul)
    s = lerp(1.3, 1.0, ease_back(age / 0.15))
    fill = lambda bb, aa: gold_tex(bb[3] - bb[1], bb[2] - bb[0], t, bb[0], bb[1], None)
    ga, gbb = draw_sprite(c, a, M_at(swing(t), s), fill, clamp01(age / 0.04) * op_mul)
    glow(c, ga, gbb, SAFFRON, 18, 0.35 * op_mul)


@functools.lru_cache(maxsize=1)
def ashke_big():
    size = fit_size('ASHKE', 'anton', 1010, 600)
    a, _ = text_alpha('ASHKE', 'anton', size)
    return a, size


def ashke_M(t):
    a, size = ashke_big(); age = t - ASHKE['start']
    s = lerp(1.6, 1.0, ease_back(age / 0.18, 1.4)) * (1 + 0.04 * pulse(BEATS, t, 0.1))
    return M_affine(W / 2, 980, s, s, 0, a.shape[1] / 2, a.shape[0] / 2)


# ------------------------------------------------------------------ end card
@functools.lru_cache(maxsize=1)
def endcard_parts():
    a, size = text_alpha('ASHKE', 'anton', fit_size('ASHKE', 'anton', 940, 600))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outline = np.clip(cv2.dilate(a, k) - cv2.erode(a, k), 0, 1)
    # original camera icon: body + viewfinder hump + double-ring lens + shutter button (4x supersampled)
    S = 4; n = 340
    m = np.zeros((n * S, n * S), np.uint8)
    th = 16 * S; X0, X1, Y0, Y1 = 24 * S, (n - 24) * S, 100 * S, (n - 50) * S; r = 34 * S
    body = np.zeros_like(m)
    cv2.rectangle(body, (X0 + r, Y0), (X1 - r, Y1), 255, -1); cv2.rectangle(body, (X0, Y0 + r), (X1, Y1 - r), 255, -1)
    for cx, cy in ((X0 + r, Y0 + r), (X1 - r, Y0 + r), (X0 + r, Y1 - r), (X1 - r, Y1 - r)):
        cv2.circle(body, (cx, cy), r, 255, -1, cv2.LINE_AA)
    hump = np.array([[100 * S, Y0 + 4 * S], [125 * S, 58 * S], [205 * S, 58 * S], [230 * S, Y0 + 4 * S]], np.int32)
    cv2.fillPoly(body, [hump], 255, cv2.LINE_AA)
    inner = cv2.erode(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * th, 2 * th)))
    m = cv2.subtract(body, inner)
    lc = (n * S // 2, (Y0 + Y1) // 2 + 6 * S)
    cv2.circle(m, lc, 72 * S, 255, th, cv2.LINE_AA)
    cv2.circle(m, lc, 34 * S, 255, 7 * S, cv2.LINE_AA)
    cv2.rectangle(m, (X1 - 70 * S, Y0 - 22 * S), (X1 - 30 * S, Y0), 255, -1)
    cv2.circle(m, (X0 + 46 * S, Y0 + 40 * S), 11 * S, 255, -1, cv2.LINE_AA)
    icon = cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return outline, icon


OUTLINE_POS = (W / 2, 720); ICON_POS = (W / 2, 1150)


def endcard_targets(n):
    outline, icon = endcard_parts()
    rng = np.random.default_rng(4)
    pts = []
    for img, (px, py), k in ((outline, OUTLINE_POS, int(n * 0.62)), (icon, ICON_POS, n - int(n * 0.62))):
        ys, xs = np.nonzero(img > 0.5)
        sel = rng.choice(len(xs), k, replace=len(xs) < k)
        pts.append(np.stack([px + xs[sel] - img.shape[1] / 2, py + ys[sel] - img.shape[0] / 2], 1))
    return np.vstack(pts)


def endcard(c, t, op):
    outline, icon = endcard_parts()
    c[:] = c * (1 - op) + radial_gradient(rgb('#4a0b16'), rgb('#120205'), W / 2, 900, 1300) * op
    phulkari(c, t, 0.05 * op)
    draw_sprite(c, outline, M_affine(OUTLINE_POS[0], OUTLINE_POS[1], 1, 1, 0, outline.shape[1] / 2, outline.shape[0] / 2), CREAM, 0.22 * op)
    # animated gradient through the icon (gold -> saffron -> red, rotating)
    n = icon.shape[0]; yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2
    ang = math.atan2(1, 1) + t * 2.2
    u = (xx * math.cos(ang) + yy * math.sin(ang)) / n + 0.5
    u = (u + t * 0.3) % 1.0
    stops = [PALEGOLD, GOLD, SAFFRON, RED, SAFFRON, GOLD, PALEGOLD]
    idx = u * (len(stops) - 1); i0 = np.floor(idx).astype(int); fr = (idx - i0)[..., None]
    S = np.array(stops); grad = S[i0] * (1 - fr) + S[np.minimum(i0 + 1, len(stops) - 1)] * fr
    ga, gbb = draw_sprite(c, icon, M_affine(ICON_POS[0], ICON_POS[1], 1, 1, 0, n / 2, n / 2), grad.astype(np.float32), op)
    glow(c, ga, gbb, SAFFRON, 22, 0.5 * op)
    a, base = text_alpha('KARAN AUJLA  /  ASHKE', 'thin', 36, 12)
    draw_sprite(c, a, M_affine(W / 2, 1420, 1, 1, 0, a.shape[1] / 2, base), CREAM, 0.85 * op)


# ------------------------------------------------------------------ particles
@functools.lru_cache(maxsize=1)
def particles(n=1600):
    a, size = ashke_big()
    M = ashke_M(T_SHATTER - 1e-3)
    ys, xs = np.nonzero(a > 0.5)
    rng = np.random.default_rng(21)
    sel = rng.choice(len(xs), n, replace=False)
    P = np.stack([xs[sel], ys[sel], np.ones(n)], 1) @ M.T
    C = np.array([W / 2, 980.0])
    d = P - C; dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    v = dn * rng.uniform(300, 1400, (n, 1)) + rng.normal(0, 160, (n, 2))
    Q = endcard_targets(n); rng.shuffle(Q)
    col = np.where(rng.random(n)[:, None] < 0.6, GOLD, PALEGOLD) * rng.uniform(0.8, 1.3, (n, 1))
    tw = rng.uniform(0, 6.28, n)
    return P, v, Q, col.astype(np.float32), tw


def particle_pos(t):
    P, v, Q, col, tw = particles()
    tau = t - T_SHATTER
    drag = 3.2
    disp = v * (1 - math.exp(-drag * tau)) / drag
    pA = P + disp
    if t < T_FORM0: return pA, 1.0
    e = ease_io(prog(t, T_FORM0, T_FORM1))
    C = np.array([W / 2, 960.0]); rel = pA - C
    phi = 2.4 * e
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    pos = C + (rel * (1 - e)) @ R.T + e * (Q - C)
    return pos, 1.0


def render_particles(c, t):
    P, v, Q, col, tw = particles()
    if t < T_SHATTER: return
    pos, _ = particle_pos(t)
    fade = 1 - prog(t, T_FORM1 + 0.02, T_FORM1 + 0.32)
    if fade <= 0: return
    twinkle = 0.65 + 0.35 * np.sin(tw + t * 30)
    vals = twinkle * fade * 2.6
    splat(c, pos[:, 0], pos[:, 1], col, vals, 1.3)
    halo_l = np.zeros_like(c)
    splat(halo_l, pos[:, 0], pos[:, 1], col, vals * 0.5, 0)
    c += blur_full(halo_l, 12, 4) * 14


def render(t, fx):
    if t >= T_SHATTER:
        c = background(t)
        e = prog(t, T_SHATTER, T_FORM1)
        c *= 1 - 0.65 * e
        ec = ease_io(prog(t, T_FORM1 - 0.12, T_FORM1 + 0.3))
        if t >= T_FORM0: endcard(c, t, ec)
        tau = t - T_SHATTER
        if tau < 0.6:   # shockwave ring
            R = 1700 * ease_out(tau / 0.6, 2)
            ang = np.linspace(0, 2 * math.pi, 360)
            draw_polyline(c, np.stack([W / 2 + R * np.cos(ang), 980 + R * np.sin(ang)], 1), PALEGOLD, 8, (1 - tau / 0.6) ** 1.5, 'add', True, glow_r=22)
            fx['flash'] = max(fx.get('flash', 0), 0.5 * math.exp(-tau / 0.07))
            fx['shake'] = 18 * math.exp(-tau / 0.12)
        render_particles(c, t)
        return c
    c = background(t)
    jhummar_circle(c, t)
    fin = ASHKE['start']
    others_op = 1 - ease_in(prog(t, fin - 0.02, fin + 0.12), 2)
    for L in layout():
        if not visible(L['w'], t) or L['w'] is ASHKE: continue
        if others_op <= 0: continue
        if L['w']['text'] == 'JHUMMAR': jhummar(c, L, t, others_op)
        else: word_pop(c, L, t, CREAM, others_op)
    if visible(ASHKE, t):
        a, size = ashke_big(); age = t - fin
        c *= 1 - 0.45 * ease_out(age / 0.15)
        M = ashke_M(t)
        sh = M.copy(); sh[0, 2] += 14; sh[1, 2] += 20
        draw_sprite(c, cv2.GaussianBlur(a, (0, 0), 8), sh, INK, 0.7)
        sweep = lerp(-0.2, 1.2, ease_io(prog(t, fin + 0.05, fin + 0.45)))
        fill = lambda bb, aa: gold_tex(bb[3] - bb[1], bb[2] - bb[0], t, bb[0], bb[1], sweep, 0.1, 0.6, 1.15)
        ga, gbb = draw_sprite(c, a, M, fill, clamp01(age / 0.03))
        glow(c, ga, gbb, SAFFRON, 30, 0.6)
        # pre-shatter cracks of light grow before the beat
        cr = prog(t, T_SHATTER - 0.22, T_SHATTER)
        if cr > 0:
            rng = np.random.default_rng(8)
            for k in range(9):
                x0 = W / 2 + rng.uniform(-420, 420); y0 = 980 + rng.uniform(-150, 150)
                pts = [(x0, y0)]; ang = rng.uniform(0, 6.28)
                for j in range(int(2 + 6 * cr)):
                    ang += rng.normal(0, 0.6); pts.append((pts[-1][0] + 40 * math.cos(ang), pts[-1][1] + 40 * math.sin(ang)))
                draw_polyline(c, pts, WHITE * 1.4, 3, cr, 'add', glow_r=6)
        fx['flash'] = max(fx.get('flash', 0), 0.3 * math.exp(-age / 0.08))
    return c
