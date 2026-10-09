"""Shared engine: timing data, audio reactivity, drawing/compositing helpers,
procedural textures (phulkari, gold foil), text sprites, footage access."""
import json, math, os, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WORK = os.path.join(ROOT, 'work')
FONTS = os.path.join(ROOT, 'fonts')
W, H, FPS = 1080, 1920, 30
DUR = 23.0
SONG_T0 = 172.0           # song time at t=0
FOOT_OFF = 0.304          # footage time = t - FOOT_OFF  (RMS x-corr result)
cv2.setNumThreads(1)

# ---------------------------------------------------------------- palette
def rgb(h): return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)
MAROON = rgb('#4a0b16'); DEEP = rgb('#1c0308'); BLOOD = rgb('#8e0f1c'); RED = rgb('#c4161c')
SAFFRON = rgb('#f39a1e'); GOLD = rgb('#e8b84a'); PALEGOLD = rgb('#ffe3a0'); BRONZE = rgb('#7a4a12')
INK = rgb('#0b0807'); CREAM = rgb('#f6ecd6'); VENOM = rgb('#46e07a'); VENOMDK = rgb('#06140b')
WHITE = np.ones(3, np.float32)

# ---------------------------------------------------------------- timing
WORDS = json.load(open(os.path.join(WORK, 'words.json')))
LINE_WORDS = [[w for w in WORDS if w['line'] == i] for i in range(6)]
BEATS_J = json.load(open(os.path.join(WORK, 'beats.json')))
BEATS = np.array(BEATS_J['beats']); DOWNBEATS = np.array(BEATS_J['downbeats'])
ENV_FPS = BEATS_J['env_fps']
ENV = {k: np.array(v, np.float32) for k, v in BEATS_J['env'].items()}
LOUD = np.array(BEATS_J['loud'], np.float32)
MID_ON = np.array([o['t'] for o in BEATS_J['band_onsets']['mid'] if o['s'] > 0.25])
HIGH_ON = np.array([o['t'] for o in BEATS_J['band_onsets']['high']])
BASS_ON = np.array([o['t'] for o in BEATS_J['band_onsets']['bass']])


def env(name, t):
    a = ENV[name] if name in ENV else LOUD
    x = t * ENV_FPS; i = int(np.clip(math.floor(x), 0, len(a) - 2)); f = x - i
    return float(a[i] * (1 - f) + a[i + 1] * f)


def since(events, t):
    p = events[events <= t + 1e-6]
    return t - p[-1] if len(p) else 99.0


def pulse(events, t, decay=0.15):
    return math.exp(-since(events, t) / decay)


def beat_phase(t):
    """returns (beat index k, fraction 0..1 through beat k)"""
    k = int(np.searchsorted(BEATS, t, side='right') - 1)
    if k < 0: return -1, 0.0
    nxt = BEATS[k + 1] if k + 1 < len(BEATS) else BEATS[k] + 0.67
    return k, (t - BEATS[k]) / (nxt - BEATS[k])


def word_at(line, t):
    """index of the latest word of `line` whose start <= t (-1 if none)"""
    ws = LINE_WORDS[line]; k = -1
    for i, w in enumerate(ws):
        if t >= w['start'] - 0.5 / FPS: k = i
    return k


def visible(w, t):
    return t >= w['start'] - 0.5 / FPS

# ---------------------------------------------------------------- easing
def clamp01(x): return max(0.0, min(1.0, x))
def lerp(a, b, u): return a + (b - a) * u
def ease_out(u, p=3): u = clamp01(u); return 1 - (1 - u) ** p
def ease_in(u, p=3): u = clamp01(u); return u ** p
def ease_io(u): u = clamp01(u); return u * u * (3 - 2 * u)
def ease_back(u, s=1.9):
    u = clamp01(u) - 1; return u * u * ((s + 1) * u + s) + 1
def elastic(u, f=3.0, d=4.0):
    u = max(0.0, u); return 1 - math.exp(-d * u) * math.cos(2 * math.pi * f * u)
def prog(t, a, b): return clamp01((t - a) / (b - a)) if b > a else float(t >= a)

# ---------------------------------------------------------------- fonts / text
FONT_FILES = dict(anton='anton-latin-400-normal.woff', bebas='bebas-neue-latin-400-normal.woff',
                  thin='jost-latin-200-normal.woff', light='jost-latin-300-normal.woff',
                  mono='space-mono-latin-400-normal.woff', monob='space-mono-latin-700-normal.woff')


@functools.lru_cache(maxsize=64)
def font(key, size):
    return ImageFont.truetype(os.path.join(FONTS, FONT_FILES[key]), int(size))


@functools.lru_cache(maxsize=512)
def text_alpha(text, key='anton', size=200, tracking=0.0, pad=None):
    """float32 alpha sprite of `text`; tracking in px between glyphs.
    Returns (alpha, baseline_y_in_sprite). Sprite tightly fits ink + pad."""
    f = font(key, size)
    pad = int(size * 0.08) if pad is None else pad
    if tracking == 0:
        l, t_, r, b = f.getbbox(text)
        img = Image.new('L', (r - l + 2 * pad, b - t_ + 2 * pad), 0)
        ImageDraw.Draw(img).text((pad - l, pad - t_), text, font=f, fill=255)
        base = pad - t_ + f.getmetrics()[0]
    else:
        widths = [f.getlength(c) for c in text]
        total = sum(widths) + tracking * (len(text) - 1)
        asc, desc = f.getmetrics()
        img = Image.new('L', (int(total + 2 * pad + size), asc + desc + 2 * pad), 0)
        d = ImageDraw.Draw(img); x = pad
        for c, wd in zip(text, widths):
            d.text((x, pad), c, font=f, fill=255); x += wd + tracking
        a = np.array(img)
        ys, xs = np.nonzero(a > 0)
        if len(xs):
            img = img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), xs.max() + pad + 1, ys.max() + pad + 1))
            base = pad + asc - max(0, ys.min() - pad)
        else:
            base = pad + asc
    return np.asarray(img, np.float32) / 255.0, base


def fit_size(text, key, max_w, max_size=900, tracking=0.0):
    f = font(key, 200)
    w = f.getlength(text) + tracking * (len(text) - 1) * 1.0
    return int(min(max_size, 200 * max_w / max(w, 1)))


def glyph_advances(text, key, size, tracking=0.0):
    f = font(key, size)
    return [f.getlength(c) + tracking for c in text]

# ---------------------------------------------------------------- transforms / compositing
def M_affine(tx, ty, sx=1.0, sy=None, rot=0.0, ox=0.0, oy=0.0):
    """2x3 matrix mapping sprite point (ox,oy) to canvas (tx,ty), scale then rotate (deg)."""
    sy = sx if sy is None else sy
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]], np.float64)
    b = np.array([tx, ty]) - A @ np.array([ox, oy])
    return np.hstack([A, b[:, None]])


def _bbox_affine(M, w, h, cw, ch, grow=2):
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0 = max(0, int(math.floor(pts[:, 0].min())) - grow); y0 = max(0, int(math.floor(pts[:, 1].min())) - grow)
    x1 = min(cw, int(math.ceil(pts[:, 0].max())) + grow); y1 = min(ch, int(math.ceil(pts[:, 1].max())) + grow)
    return x0, y0, x1, y1


def warp_sprite(src, M, cw=W, ch=H, interp=cv2.INTER_LINEAR):
    """warp sprite (h,w[,c]) by M into canvas space; returns (region, bbox) or (None,None)."""
    h, w = src.shape[:2]
    x0, y0, x1, y1 = _bbox_affine(M, w, h, cw, ch)
    if x1 <= x0 or y1 <= y0: return None, None
    M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
    out = cv2.warpAffine(src, M2, (x1 - x0, y1 - y0), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, (x0, y0, x1, y1)


def comp(canvas, alpha, bbox, fill, opacity=1.0, mode='normal'):
    """composite fill (color (3,) or array (h,w,3) matching alpha) through alpha at bbox."""
    if alpha is None: return
    x0, y0, x1, y1 = bbox
    ch, cw = canvas.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
    if X1 <= X0 or Y1 <= Y0: return
    if (X0, Y0, X1, Y1) != (x0, y0, x1, y1):
        alpha = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        if isinstance(fill, np.ndarray) and fill.ndim == 3:
            fill = fill[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        x0, y0, x1, y1 = X0, Y0, X1, Y1
    reg = canvas[y0:y1, x0:x1]
    a = (alpha * opacity)[..., None]
    if mode == 'add':
        reg += fill * a
    elif mode == 'screen':
        reg[:] = 1 - (1 - reg) * (1 - np.clip(fill * a, 0, 1))
    else:
        reg[:] = reg * (1 - a) + fill * a


def draw_sprite(canvas, alpha, M, fill, opacity=1.0, mode='normal'):
    a, bb = warp_sprite(alpha, M, canvas.shape[1], canvas.shape[0])
    if a is None: return None, None
    if callable(fill): f = fill(bb, a)
    elif isinstance(fill, np.ndarray) and fill.ndim == 3 and fill.shape[:2] == alpha.shape:
        f, _ = warp_sprite(fill, M, canvas.shape[1], canvas.shape[0])
    else: f = fill
    comp(canvas, a, bb, f, opacity, mode)
    return a, bb


def glow(canvas, alpha, bbox, color, radius=20, strength=1.0, mode='add'):
    """soft glow around an already-warped alpha region."""
    if alpha is None: return
    x0, y0, x1, y1 = bbox; pad = int(radius * 2.5)
    ch, cw = canvas.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(cw, x1 + pad), min(ch, y1 + pad)
    big = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    big[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = alpha
    k = 4 if radius > 12 else 1
    small = cv2.resize(big, (max(1, big.shape[1] // k), max(1, big.shape[0] // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), radius / k)
    g = cv2.resize(small, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_LINEAR)
    comp(canvas, np.clip(g * strength, 0, 2), (X0, Y0, X1, Y1), color, 1.0, mode)


def blur_full(img, sigma, k=4):
    h, w = img.shape[:2]
    s = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), sigma / k)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)

# ---------------------------------------------------------------- procedural textures
def _phulkari_tile(n=216):
    """geometric phulkari-style motif: nested diamonds + 8-point stars, drawn
    with line work (alpha, color) on a transparent tile."""
    S = 4; N = n * S
    a = np.zeros((N, N), np.uint8); c = np.zeros((N, N, 3), np.uint8)
    cols = [(243, 154, 30), (232, 184, 74), (196, 22, 28), (246, 236, 214)]
    def diamond(cx, cy, r, col, th):
        p = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
        cv2.polylines(a, [p], True, 255, th, cv2.LINE_AA); cv2.polylines(c, [p], True, col, th, cv2.LINE_AA)
    def star(cx, cy, r, col, th):
        pts = []
        for k in range(16):
            rr = r if k % 2 == 0 else r * 0.45
            ang = math.pi * k / 8
            pts.append([cx + rr * math.cos(ang), cy + rr * math.sin(ang)])
        p = np.array(pts, np.int32)
        cv2.polylines(a, [p], True, 255, th, cv2.LINE_AA); cv2.polylines(c, [p], True, col, th, cv2.LINE_AA)
    h = N // 2
    for (cx, cy) in [(h, h), (0, 0), (N, 0), (0, N), (N, N)]:
        for i, r in enumerate([h * 0.95, h * 0.72, h * 0.5]):
            diamond(cx, cy, r, cols[i % 3], 3 * S if i == 0 else 2 * S)
        star(cx, cy, h * 0.3, cols[3], 2 * S)
    for (cx, cy) in [(h, 0), (0, h), (N, h), (h, N)]:
        star(cx, cy, h * 0.22, cols[1], 2 * S)
        diamond(cx, cy, h * 0.1, cols[0], 2 * S)
    # small stitches (dashes) along the lattice
    for k in range(0, N, 12 * S):
        cv2.line(a, (k, h), (k + 5 * S, h), 160, S); cv2.line(c, (k, h), (k + 5 * S, h), cols[2], S)
    a = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    c = cv2.resize(c, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    c = c / np.maximum(a[..., None], 1e-3)
    return np.clip(c, 0, 1), a


@functools.lru_cache(maxsize=1)
def phulkari_big():
    c, a = _phulkari_tile()
    n = c.shape[0]; rx = W // n + 3; ry = H // n + 3
    return np.tile(c, (ry, rx, 1)), np.tile(a, (ry, rx))


def phulkari(canvas, t, opacity=0.1, speed=(9, -14), rot=0.0, tint=None, mask=None):
    c, a = phulkari_big(); n = 216
    ox = int(t * speed[0]) % n; oy = int(t * speed[1]) % n
    cc = c[oy:oy + H, ox:ox + W]; aa = a[oy:oy + H, ox:ox + W] * opacity
    if tint is not None: cc = cc * 0.4 + tint * 0.6
    if mask is not None: aa = aa * mask
    canvas[:] = canvas * (1 - aa[..., None]) + cc * aa[..., None]


def gold_tex(h, w, t, x0=0, y0=0, sweep=None, sweep_w=0.12, angle=0.6, bright=1.0):
    """gold foil: banded metallic gradient with fine grain and an optional
    diagonal light sweep (sweep = 0..1 position across the region)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X = (xx + x0); Y = (yy + y0)
    v = 0.5 + 0.5 * np.sin(Y * 0.035 + X * 0.012 + t * 1.3) * np.cos(X * 0.006 - Y * 0.01 + t * 0.7)
    v = 0.55 * v + 0.45 * (yy / max(h - 1, 1))
    grain = (np.sin(X * 1.7 + Y * 0.3) * np.sin(Y * 2.1 - X * 0.5)) * 0.04
    v = np.clip(v + grain, 0, 1)[..., None]
    lo, mid, hi = BRONZE, GOLD, PALEGOLD
    col = np.where(v < 0.5, lo + (mid - lo) * (v / 0.5), mid + (hi - mid) * ((v - 0.5) / 0.5))
    if sweep is not None:
        d = (xx / max(w, 1) + (yy / max(h, 1)) * angle) / (1 + angle)
        s = np.exp(-((d - sweep) / sweep_w) ** 2)[..., None]
        col = col + s * 0.9
    return np.clip(col * bright, 0, 1.6).astype(np.float32)


@functools.lru_cache(maxsize=4)
def grain_frames(n=6, amp=0.035):
    rng = np.random.default_rng(7)
    return [(rng.standard_normal((H // 2, W // 2)).astype(np.float32) * amp) for _ in range(n)]


def add_grain(canvas, t):
    g = grain_frames()[int(t * FPS) % 6]
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
    canvas += g[..., None]


@functools.lru_cache(maxsize=2)
def vignette_map(strength=0.55):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    return (1 - strength * np.clip(d, 0, 1.4) ** 2.2).astype(np.float32)[..., None]


@functools.lru_cache(maxsize=4)
def star_sprite(size=96):
    """4-point star glint (alpha)"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - size / 2 + 0.5
    r = size / 2
    a = np.exp(-(np.abs(xx) / (r * 0.05))) * np.exp(-(np.abs(yy) / r) * 3.2)
    b = np.exp(-(np.abs(yy) / (r * 0.05))) * np.exp(-(np.abs(xx) / r) * 3.2)
    c = np.exp(-(xx ** 2 + yy ** 2) / (r * 0.12) ** 2)
    return np.clip(a + b + c * 1.2, 0, 1).astype(np.float32)


def radial_gradient(c0, c1, cx=W / 2, cy=H / 2, r=None):
    r = r or max(W, H) * 0.75
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1)[..., None]
    return (c0 * (1 - d) + c1 * d).astype(np.float32)


def noise2d(h, w, scale, seed, t=0.0):
    """smooth value noise via upsampled random grid (cheap fbm)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(3):
        gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); tot += amp
        scale /= 2.2; amp *= 0.5
    return out / tot


def splat(layer, xs, ys, cols, vals, sigma=1.5):
    """additively splat points into layer (H,W,3) with gaussian softness."""
    h, w = layer.shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    xi = np.round(xs).astype(int); yi = np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if ok.any():
        np.add.at(acc, (yi[ok], xi[ok]), cols[ok] * vals[ok][:, None])
        if sigma > 0:
            acc = cv2.GaussianBlur(acc, (0, 0), sigma) * (2 * math.pi * sigma * sigma)
    layer += acc


def draw_polyline(canvas, pts, color, th=2, opacity=1.0, mode='normal', closed=False, glow_r=0):
    """antialiased polyline composited in float."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2: return
    x0 = max(0, int(pts[:, 0].min()) - th - 4 - glow_r * 3); y0 = max(0, int(pts[:, 1].min()) - th - 4 - glow_r * 3)
    x1 = min(canvas.shape[1], int(pts[:, 0].max()) + th + 5 + glow_r * 3); y1 = min(canvas.shape[0], int(pts[:, 1].max()) + th + 5 + glow_r * 3)
    if x1 <= x0 or y1 <= y0: return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    p = np.round((pts - [x0, y0]) * 16).astype(np.int32)
    cv2.polylines(m, [p], closed, 255, max(1, int(th)), cv2.LINE_AA, shift=4)
    a = m.astype(np.float32) / 255
    if glow_r:
        g = cv2.GaussianBlur(a, (0, 0), glow_r)
        comp(canvas, np.clip(g * 2.0, 0, 1), (x0, y0, x1, y1), color, opacity * 0.8, 'add')
    comp(canvas, a, (x0, y0, x1, y1), color, opacity, mode)

# ---------------------------------------------------------------- footage
class Footage:
    AX, AY, AW, AH = 240, 2, 2140, 1068

    def __init__(self):
        s = json.load(open(os.path.join(WORK, 'subject.json')))
        self.pts = np.array(s['pts']); self.shot = np.array(s['shot'])
        self.cx = np.array(s['cx']); self.cy = np.array(s['cy']); self.fs = np.array(s['fs'])
        self.cuts = [0.0, 2.55, 3.50, 3.983, 4.35, 4.983, 6.833, 99]
        self._cache = {}; self._mcache = {}
        self._dis = None

    def idx(self, f):
        return int(np.clip(np.searchsorted(self.pts, f + 1e-4, side='right') - 1, 0, len(self.pts) - 1))

    def _load(self, i):
        if i not in self._cache:
            if len(self._cache) > 6: self._cache.pop(next(iter(self._cache)))
            im = cv2.imread(os.path.join(WORK, 'src_frames', f'{i + 1:04d}.jpg'))
            im = im[self.AY:self.AY + self.AH, self.AX:self.AX + self.AW]
            self._cache[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        return self._cache[i]

    def frame(self, f, interp=False):
        i = self.idx(f)
        if not interp: return self._load(i)
        j = min(i + 1, len(self.pts) - 1)
        if j == i or self.shot[j] != self.shot[i]: return self._load(i)
        u = np.float32((f - self.pts[i]) / (self.pts[j] - self.pts[i]))
        if u < 0.08: return self._load(i)
        if u > 0.92: return self._load(j)
        A, B = self._load(i), self._load(j)
        if self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        k = 4
        ga = cv2.cvtColor((cv2.resize(A, (self.AW // k, self.AH // k)) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor((cv2.resize(B, (self.AW // k, self.AH // k)) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fab = self._dis.calc(ga, gb, None); fba = self._dis.calc(gb, ga, None)
        fab = cv2.resize(fab, (self.AW, self.AH)) * k; fba = cv2.resize(fba, (self.AW, self.AH)) * k
        yy, xx = np.mgrid[0:self.AH, 0:self.AW].astype(np.float32)
        wa = cv2.remap(A, xx - fba[..., 0] * u, yy - fba[..., 1] * u, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(B, xx - fab[..., 0] * (1 - u), yy - fab[..., 1] * (1 - u), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return wa * (1 - u) + wb * u

    def subject(self, f):
        i = self.idx(f); return self.cx[i], self.cy[i], self.fs[i]

    def shot_of(self, f):
        return int(np.searchsorted(self.cuts, f, side='right') - 1)

    def mask(self, f, restrict=True, win=2.6):
        """person mask (AH,AW) float; restricted to a soft body window under the subject."""
        i = self.idx(f); key = (i, restrict, win)
        if key in self._mcache: return self._mcache[key]
        if len(self._mcache) > 6: self._mcache.pop(next(iter(self._mcache)))
        m = cv2.imread(os.path.join(WORK, 'masks', f'{i + 1:04d}.png'), 0).astype(np.float32) / 255
        m = m[self.AY // 2:(self.AY + self.AH) // 2, self.AX // 2:(self.AX + self.AW) // 2]
        m = cv2.resize(m, (self.AW, self.AH), interpolation=cv2.INTER_LINEAR)
        m = np.clip((m - 0.35) / 0.4, 0, 1)
        if restrict == 'cc':
            cx, cy, fs = self.cx[i], self.cy[i], self.fs[i]
            n, lab = cv2.connectedComponents((m > 0.5).astype(np.uint8), connectivity=8)
            yy, xx = int(np.clip(cy + 0.6 * fs, 0, self.AH - 1)), int(np.clip(cx, 0, self.AW - 1))
            k = lab[yy, xx]
            if k == 0:
                sub = lab[int(max(0, cy)):int(min(self.AH, cy + 2 * fs)), int(max(0, cx - fs)):int(min(self.AW, cx + fs))]
                vals, cnt = np.unique(sub[sub > 0], return_counts=True)
                k = vals[np.argmax(cnt)] if len(vals) else 0
            if k:
                keep = cv2.dilate((lab == k).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
                m = m * cv2.GaussianBlur(keep, (0, 0), 3)
        elif restrict:
            cx, cy, fs = self.cx[i], self.cy[i], self.fs[i]
            xx = np.arange(self.AW, dtype=np.float32); yy = np.arange(self.AH, dtype=np.float32)
            wx = np.clip(1 - (np.abs(xx - cx) - win * fs) / (0.6 * fs), 0, 1)
            wy = np.clip((yy - (cy - 1.25 * fs)) / (0.4 * fs), 0, 1)
            m = m * wx[None, :] * wy[:, None]
        m = cv2.GaussianBlur(m, (0, 0), 1.0)
        self._mcache[key] = m
        return m


FOOT = None
def footage():
    global FOOT
    if FOOT is None: FOOT = Footage()
    return FOOT


def foot_M(cx, cy, scale, ox=W / 2, oy=H / 2, rot=0.0):
    """affine mapping footage point (cx,cy) to canvas (ox,oy) with scale."""
    return M_affine(ox, oy, scale, scale, rot, cx, cy)


def warp_full(img, M, size=(W, H), border=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, M, size, flags=interp, borderMode=border, borderValue=0)


def fullbleed(img, cx, cy, scale=1.45, oy=H / 2, ox=W / 2, rot=0.0, ext_dark=0.55):
    """sharp footage at <=1.5x, with a blurred, darkened 'cover' extension
    filling the remaining frame (never stretched). returns (rgb, M)."""
    AH, AW = img.shape[:2]
    vis_w = W / scale
    cx = float(np.clip(cx, vis_w / 2, AW - vis_w / 2))
    M = foot_M(cx, cy, scale, ox, oy, rot)
    sharp = warp_full(img, M)
    a = warp_full(np.ones((AH, AW), np.float32), M)
    a = cv2.GaussianBlur(a, (0, 0), 14) * a
    cover = H / AH * 1.08
    small = cv2.resize(img, (AW // 8, AH // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    Mc = foot_M(cx / 8, AH / 16, cover * 8, ox, H / 2, rot)
    ext = warp_full(small, Mc, border=cv2.BORDER_REFLECT) * ext_dark
    a3 = a[..., None]
    return sharp * a3 + ext * (1 - a3), M


def grade(img, sat=1.0, contrast=1.0, lift=0.0, tint=None, tint_amt=0.0, gamma=1.0):
    g = img.mean(axis=2, keepdims=True)
    out = g + (img - g) * sat
    out = (out - 0.5) * contrast + 0.5 + lift
    if tint is not None: out = out * (1 - tint_amt) + (out.mean(axis=2, keepdims=True) * tint * 1.6) * tint_amt
    out = np.clip(out, 0, 1)
    if gamma != 1.0: out = out ** gamma
    return out.astype(np.float32)
