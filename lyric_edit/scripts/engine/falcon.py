"""Original procedural falcon silhouette (top-down, swept pointed wings) and the
feather wake it leaves. Shared by L2 (flight) and L3 (feathers -> text)."""
import math, functools
import numpy as np, cv2
from .core import *

T_FLY = LINE_WORDS[1][7]['start']          # 'BAAZAN'
FLY_DUR = 1.25
T_GATHER0, T_GATHER1 = 7.02, 7.42          # feathers converge into L3's first word
RAHAN_POS = (540, 1290)                    # where L3 shows RAHAN at its start (see scene_l3)


def path(u):
    """flight path (u 0..1) as a cubic bezier from lower-left to upper-right"""
    P = np.array([[-260, 1560], [260, 1080], [760, 1180], [1360, 330]], np.float64)
    b = np.array([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3])
    db = np.array([-3 * (1 - u) ** 2, 3 * (1 - u) ** 2 - 6 * u * (1 - u), 6 * u * (1 - u) - 3 * u * u, 3 * u * u])
    return b @ P, db @ P


def falcon_polys(flap, jit_seed):
    """polygons in body units (beak +x). flap phase in radians."""
    sp = 0.58 + 0.42 * abs(math.cos(flap))            # foreshortened span
    sweep = 0.08 * math.sin(flap)
    rng = np.random.default_rng(jit_seed)
    def J(p): return np.array(p) + rng.normal(0, 0.005, np.shape(p))
    half = [[0.62, 0.0], [0.57, -0.025], [0.52, -0.06], [0.45, -0.085], [0.37, -0.08], [0.30, -0.065],
            [0.18, -0.10], [-0.05, -0.085], [-0.28, -0.05], [-0.34, -0.11], [-0.60, -0.17], [-0.67, -0.07]]
    body = J(half + [[-0.69, 0.0]] + [[x, -y] for x, y in reversed(half[1:])])
    wings = []
    for sg in (-1, 1):
        w = [[0.16, 0.06], [0.15, 0.35], [0.03, 0.70], [-0.24 - sweep, 1.08], [-0.56 - sweep, 1.32],
             [-0.47 - sweep, 1.17], [-0.51 - sweep, 1.10], [-0.41 - sweep, 1.02], [-0.44 - sweep, 0.93],
             [-0.33 - sweep, 0.86], [-0.35 - sweep, 0.75], [-0.23, 0.63], [-0.21, 0.46], [-0.15, 0.30], [-0.12, 0.08]]
        w = [[x, sg * (y * sp if y > 0.1 else y)] for x, y in w]
        wings.append(J(w))
    return body, wings


def draw_falcon(canvas, x, y, ang, size, flap, t, opacity=1.0):
    body, wings = falcon_polys(flap, int(t * 12))       # 'boil' every ~2 frames
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]]) * size
    polys = [(p @ R.T) + [x, y] for p in [*wings, body]]
    allp = np.vstack(polys)
    pad = 12
    x0, y0 = int(max(0, allp[:, 0].min() - pad)), int(max(0, allp[:, 1].min() - pad))
    x1, y1 = int(min(W, allp[:, 0].max() + pad)), int(min(H, allp[:, 1].max() + pad))
    if x1 <= x0 or y1 <= y0: return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8); e = np.zeros_like(m)
    ip = [np.round((p - [x0, y0]) * 16).astype(np.int32) for p in polys]
    cv2.fillPoly(m, ip, 255, cv2.LINE_AA, shift=4)
    cv2.polylines(e, ip, True, 255, 3, cv2.LINE_AA, shift=4)
    # feather barbs drawn on the wings (hand-drawn line work)
    for wpoly in polys[:2]:
        root = (wpoly[0] + wpoly[-1]) / 2
        for k in (5, 7, 9, 11):
            q = np.round(np.array([root, wpoly[k] * 0.85 + root * 0.15]) - [x0, y0]).astype(np.int32)
            cv2.line(e, tuple(q[0]), tuple(q[1]), 150, 2, cv2.LINE_AA)
    a = m.astype(np.float32) / 255; ea = e.astype(np.float32) / 255 * a
    bb = (x0, y0, x1, y1)
    glow(canvas, a, bb, SAFFRON, 18, 0.5 * opacity)
    comp(canvas, a, bb, INK, opacity)
    comp(canvas, ea, bb, GOLD * 1.1, opacity)


def render_falcon(canvas, t):
    if not (T_FLY <= t <= T_FLY + FLY_DUR): return
    u = (t - T_FLY) / FLY_DUR
    u = 0.5 * u + 0.5 * u * u          # accelerates as it leaves
    p, d = path(u)
    ang = math.atan2(d[1], d[0])
    size = lerp(230, 330, u)
    draw_falcon(canvas, p[0], p[1], ang, size, t * 2 * math.pi * 3.2, t)


@functools.lru_cache(maxsize=1)
def feather_seeds(n=70):
    rng = np.random.default_rng(3)
    birth = np.sort(rng.uniform(T_FLY + 0.05, T_FLY + FLY_DUR * 0.95, n))
    return dict(birth=birth, dx=rng.normal(0, 40, n), vy=rng.uniform(40, 140, n),
                spin=rng.uniform(-4, 4, n), ph=rng.uniform(0, 6.28, n), size=rng.uniform(26, 46, n),
                gold=rng.random(n) < 0.45, tgt=rng.random((n, 2)))


@functools.lru_cache(maxsize=1)
def feather_sprite():
    """small feather: shaft + vanes (alpha), pointing +x"""
    S = 4; w, h = 64 * S, 22 * S
    m = np.zeros((h, w), np.uint8)
    pts = []
    for i in range(40):
        u = i / 39; x = u * w
        half = (h / 2 - 2) * math.sin(math.pi * min(1, u * 1.15)) ** 0.7
        pts.append((x, h / 2 - half))
    pts += [(x, h - y) for x, y in reversed(pts)]
    cv2.fillPoly(m, [np.int32(pts)], 200, cv2.LINE_AA)
    for k in range(4, w - 8, 6 * S):    # barb gaps
        cv2.line(m, (k, 0), (k + 4 * S, h // 2), 60, S)
    cv2.line(m, (0, h // 2), (w, h // 2), 255, S)
    return cv2.resize(m, (64, 22), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def feather_state(t, targets=None):
    """closed-form feather positions at time t. returns list of (x,y,rot,size,op,col)."""
    s = feather_seeds(); out = []
    for i in range(len(s['birth'])):
        b = s['birth'][i]
        if t < b: continue
        u0 = (b - T_FLY) / FLY_DUR; u0 = 0.5 * u0 + 0.5 * u0 * u0
        p, _ = path(u0)
        age = t - b
        x = p[0] + s['dx'][i] * min(age, 1) + 30 * math.sin(age * 3 + s['ph'][i])
        y = p[1] + s['vy'][i] * age
        rot = math.degrees(s['ph'][i] + s['spin'][i] * age * 0.6) + 25 * math.sin(age * 4 + s['ph'][i])
        op = clamp01(age / 0.12)
        col = GOLD if s['gold'][i] else CREAM
        if t > T_GATHER0 and targets is not None and len(targets):
            g = ease_io(prog(t, T_GATHER0, T_GATHER1))
            tx, ty = targets[i % len(targets)]
            x = lerp(x, tx, g); y = lerp(y, ty, g); rot = lerp(rot, 0, g)
            op *= 1 - ease_in(prog(t, T_GATHER1 - 0.12, T_GATHER1 + 0.02), 2)
        if op > 0.01: out.append((x, y, rot, s['size'][i], op, col))
    return out


def render_feathers(canvas, t, targets=None):
    sp = feather_sprite()
    for x, y, rot, size, op, col in feather_state(t, targets):
        sc = size / 64
        draw_sprite(canvas, sp, M_affine(x, y, sc, sc, rot, 32, 11), col, op * 0.95)
