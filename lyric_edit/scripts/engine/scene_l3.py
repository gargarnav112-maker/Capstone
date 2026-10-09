"""L3 'Rahan vich rode behna je yaaran nal paina kass ke'
Stones on the road — if you ride with brothers, tighten up.
RAHAN = a road: the footage becomes the sky; a perspective road runs through
embroidered (phulkari) fields and each word races toward camera along it.
YAARAN: the people in the sky footage get hand-drawn ink outlines drawn on.
KASS KE: tracking squeezes inward, the word thickens, shakes, rubber-band snap."""
import math, functools
import numpy as np, cv2
from .core import *
from . import falcon

LINE = 2
HZ = 905; KY = 1015; KX = 760          # horizon, ground projection constants
Z_READ = 1.65
SKY_SCALE = 1.0


@functools.lru_cache(maxsize=1)
def ground_maps():
    yy, xx = np.mgrid[HZ + 1:H, 0:W].astype(np.float32)
    Z = KY / (yy - HZ); X = (xx - W / 2) * Z / KX
    fog = np.clip((yy - HZ) / 420, 0, 1) ** 1.3
    return X, Z, fog


def road_pos(t):
    """distance travelled along the road (units). brakes hard into KASS."""
    kass = LINE_WORDS[LINE][8]['start']
    v = 7.0; tb = kass - 0.30; dur = 0.30
    if t < tb: return v * t
    u = min(t - tb, dur)
    return v * tb + v * (u - u * u / (2 * dur))


def y_of(z): return HZ + KY / z


def word_sprite(w):
    a, base = text_alpha(w['text'], 'anton', 220)
    k = min(1.45, 960 / a.shape[1])
    return a, base, k


def word_z(i, t):
    ws = LINE_WORDS[LINE]; w = ws[i]
    ns = ws[i + 1]['start'] if i + 1 < len(ws) else 99
    if t < w['start']:
        return Z_READ + 24 * ((w['start'] - t) / 0.34) ** 1.6, 1.0
    ns = ns - 0.07
    if t < ns:
        return Z_READ - 0.22 * prog(t, w['start'], ns), 1.0
    u = prog(t, ns, ns + 0.10)
    return (Z_READ - 0.22) - 0.75 * ease_in(u, 2), (1 - u) ** 4


def rahan_targets():
    """points on the RAHAN glyphs at its reading position (feather destinations)."""
    w = LINE_WORDS[LINE][0]; a, base, k = word_sprite(w)
    s = k * 1.6 / Z_READ * 1.0
    ys, xs = np.nonzero(a[::6, ::6] > 0.5)
    rng = np.random.default_rng(1); sel = rng.choice(len(xs), min(70, len(xs)), replace=False)
    yb = y_of(Z_READ)
    px = W / 2 + (xs[sel] * 6 - a.shape[1] / 2) * s; py = yb + (ys[sel] * 6 - base) * s
    return list(zip(px, py))


def sky(t, c):
    F = footage(); f = t - FOOT_OFF
    img = F.frame(f); cx, cy, fs = F.subject(f)
    vis_w = W / SKY_SCALE
    cxx = float(np.clip(cx, vis_w / 2, F.AW - vis_w / 2))
    oy = HZ - (F.AH - cy) * SKY_SCALE + 40
    M = foot_M(cxx, cy, SKY_SCALE, W / 2 + 12 * math.sin(t * 0.8), oy)
    s = warp_full(img, M)
    a = warp_full(np.ones(img.shape[:2], np.float32), M)
    yy = np.arange(H, dtype=np.float32)
    a *= np.clip((HZ + 20 - yy) / 160, 0, 1)[:, None]
    s = grade(s, 0.95, 1.08, 0.0, SAFFRON, 0.12)
    c[:] = c * (1 - a[..., None]) + s * a[..., None]
    return M, img.shape


def ink_outlines(c, t, M, f):
    """YAARAN: hand-drawn ink outlines draw themselves around the people."""
    w = LINE_WORDS[LINE][5]
    if t < w['start']: return
    p = ease_out(prog(t, w['start'], w['start'] + 0.42), 2)
    fade = 1 - prog(t, LINE_WORDS[LINE][9]['end'] - 0.1, LINE_WORDS[LINE][9]['end'] + 0.1)
    m = footage().mask(f, False)
    small = cv2.resize(m, (m.shape[1] // 4, m.shape[0] // 4))
    cs, _ = cv2.findContours((small > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rng = np.random.default_rng(int(t * 10))           # 'boil' at ~10 fps like hand-drawn animation
    for k, cnt in enumerate(sorted(cs, key=cv2.contourArea, reverse=True)[:7]):
        if cv2.contourArea(cnt) < 120: continue
        pts = cnt[:, 0, :].astype(np.float64) * 4
        pts = pts[::3]
        pts = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
        n = int(len(pts) * min(1.0, p * (1.0 + 0.15 * k)))
        if n < 2: continue
        q = pts[:n] + rng.normal(0, 1.6, (n, 2))
        inside = q[:, 1] < HZ + 10
        if inside.sum() < 2: continue
        q = q[inside]
        draw_polyline(c, q, CREAM, 3, 0.9 * fade, 'normal', False, glow_r=0)
        draw_polyline(c, q + [5, 3], SAFFRON, 2, 0.5 * fade, 'normal', False)
        # pen nib spark at the drawing head
        if p < 1:
            draw_sprite(c, star_sprite(64), M_affine(q[-1, 0], q[-1, 1], 0.8, 0.8, 0, 32, 32), PALEGOLD, 0.9, 'add')


def ground(c, t):
    X, Z, fog = ground_maps()
    pos = road_pos(t)
    reg = c[HZ + 1:]
    base = (MAROON * 0.55)[None, None, :] * (0.35 + 0.65 * fog[..., None])
    reg[:] = base
    # embroidered fields: phulkari mapped onto the ground plane (wrap)
    tile_c, tile_a = phulkari_big()
    tc, ta = tile_c[:216, :216], tile_a[:216, :216]
    S = 216 / 1.6
    u = ((X + 20) * S) % 216; v = ((Z + pos) * S) % 216
    pc = cv2.remap(tc, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    pa = cv2.remap(ta, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    field = (np.abs(X) > 1.15).astype(np.float32)
    aa = (pa * 0.42 * fog * field)[..., None]
    reg[:] = reg * (1 - aa) + pc * aa
    # road surface
    road = np.clip((1.0 - np.abs(X)) * 40, 0, 1)
    asphalt = (INK * 1.6 + MAROON * 0.15)[None, None, :] * (0.6 + 0.4 * fog[..., None])
    reg[:] = reg * (1 - road[..., None]) + asphalt * road[..., None]
    # edge lines + gold centre dashes (moving)
    edge = np.clip(1 - np.abs(np.abs(X) - 0.95) / 0.025, 0, 1) * fog
    dash = np.clip(1 - np.abs(X) / 0.035, 0, 1) * (((Z + pos) % 1.3) < 0.62) * fog
    reg += edge[..., None] * CREAM * 0.75
    reg += dash[..., None] * GOLD * (0.9 + 0.5 * env('bass', t))
    # horizon glow
    yy = np.arange(H, dtype=np.float32)
    hg = np.exp(-((yy - HZ) / 26) ** 2)[:, None, None]
    c += hg * SAFFRON * 0.45


def draw_word(c, i, t, fx):
    ws = LINE_WORDS[LINE]; w = ws[i]
    if t < w['start'] - 0.34: return
    a, base, k = word_sprite(w)
    if i >= 8:
        return kass_ke(c, i, t, fx)
    z, op = word_z(i, t)
    if op <= 0 or z <= 0.2: return
    s = k * 1.6 / z
    yb = y_of(max(z, 0.35))
    if z < 1.0: yb = y_of(1.0) + (1.0 - z) * 2200
    M = M_affine(W / 2, yb, s, s, 0, a.shape[1] / 2, base)
    # road shadow
    sh = cv2.GaussianBlur(a, (0, 0), 6)
    Ms = M_affine(W / 2, yb + 8 * s, s, s * 0.25, 0, a.shape[1] / 2, base)
    draw_sprite(c, sh, Ms, INK, 0.55 * op)
    reading = w['start'] <= t
    col = CREAM if not reading else lerp(PALEGOLD * 1.15, CREAM, prog(t, w['start'], w['start'] + 0.25))
    ga, gbb = draw_sprite(c, a, M, col, op)
    if reading and op > 0.5:
        glow(c, ga, gbb, SAFFRON, 22, 0.35 * op * (1 - prog(t, w['start'], w['start'] + 0.3)) + 0.12)


@functools.lru_cache(maxsize=64)
def thick(text, size, tracking, r):
    a, base = text_alpha(text, 'anton', size, tracking)
    if r > 0:
        a = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return a, base


def kass_ke(c, i, t, fx):
    ws = LINE_WORDS[LINE]; kass, ke = ws[8], ws[9]
    k_snap = BEATS[np.argmin(np.abs(BEATS - (kass['start'] + 0.25)))]
    if i == 8:
        if t < kass['start'] - 0.25: return
        sq = ease_in(prog(t, kass['start'] - 0.04, kass['start'] + 0.2), 2)
        tracking = int(round(lerp(150, -8, sq) / 4) * 4)
        r = int(round(lerp(0, 5, sq)))
        a, base = thick('KASS', 230, tracking, r)
        arr = ease_out(prog(t, kass['start'] - 0.25, kass['start']))
        x = W / 2 - lerp(0, 175, ease_io(prog(t, ke['start'] - 0.05, ke['start'] + 0.12)))
        sx = 1.0; sy = 1.0; jx = jy = 0
        if t > kass['start'] + 0.2:
            rng = np.random.default_rng(int(t * FPS))
            amp = 16 * math.exp(-(t - kass['start'] - 0.2) / 0.12) if t < k_snap else 0
            jx, jy = rng.normal(0, amp, 2)
        if t >= k_snap:
            e = elastic((t - k_snap) / 0.5, 2.2, 5.0)
            sx = lerp(1.22, 1.0, e); sy = lerp(0.84, 1.0, e)
            fx['flash'] = max(fx.get('flash', 0), 0.22 * math.exp(-(t - k_snap) / 0.06))
        yb = y_of(Z_READ) + 30
        s = lerp(0.5, 1.0, arr)
        M = M_affine(x + jx, yb + jy, s * sx, s * sy, 0, a.shape[1] / 2, base)
        col = lerp(CREAM, GOLD * 1.1, sq)
        ga, gbb = draw_sprite(c, a, M, col, arr)
        glow(c, ga, gbb, RED, 26, 0.5 * sq)
        # tension lines while squeezing
        if 0 < sq < 1:
            for sg in (-1, 1):
                xx = x + sg * (a.shape[1] / 2 + 40)
                draw_polyline(c, [(xx, yb - 200), (xx, yb + 10)], CREAM, 4, 0.7, 'add')
    else:
        if t < ke['start']: return
        a, base = thick('KE', 230, -8, 5)
        e = ease_back(prog(t, ke['start'], ke['start'] + 0.16))
        x = W / 2 + 330
        M = M_affine(x, y_of(Z_READ) + 30, lerp(1.6, 1.0, e), lerp(1.6, 1.0, e), 0, a.shape[1] / 2, base)
        ga, gbb = draw_sprite(c, a, M, CREAM, clamp01((t - ke['start']) / 0.05))
        glow(c, ga, gbb, SAFFRON, 22, 0.4)


def render(t, fx):
    c = np.zeros((H, W, 3), np.float32)
    c[:HZ + 1] = DEEP
    M, _ = sky(t, c)
    ink_outlines(c, t, M, t - FOOT_OFF)
    ground(c, t)
    # speed streaks along the road shoulders
    pos = road_pos(t)
    for k in range(10):
        z = 1.2 + ((k * 2.7 - pos * 1.6) % 26)
        for sg in (-1, 1):
            x0 = W / 2 + sg * KX * 1.6 / z; y0 = y_of(z)
            x1 = W / 2 + sg * KX * 1.6 / (z * 0.8); y1 = y_of(z * 0.8)
            draw_polyline(c, [(x0, y0), (x1, y1)], PALEGOLD, 2, 0.35 * clamp01(3 / z), 'add')
    ws = LINE_WORDS[LINE]
    def outgoing(i): return i < 9 and t >= ws[i + 1]['start'] - 0.07
    order = sorted(range(10), key=lambda i: (0 if outgoing(i) else 1, -word_z(i, t)[0] if i < 8 else -1))
    for i in order:
        draw_word(c, i, t, fx)
    falcon.render_feathers(c, t, rahan_targets())
    return c
