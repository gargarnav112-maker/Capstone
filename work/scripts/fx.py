"""Drawing / compositing primitives for the lyric edit. Canvas = float32 BGR in [0,1], 1080x1920."""
import cv2, numpy as np, functools, os
from PIL import Image, ImageDraw, ImageFont

CW, CH = 1080, 1920
FPS = 30
SAFE = dict(x0=70, x1=CW - 70, y0=150, y1=CH - 250)
D = os.path.dirname(os.path.abspath(__file__)) + '/../'
FONTS = {k: D + f'fonts/{v}.woff' for k, v in dict(
    anton='Anton', bebas='Bebas', round='Nunito900', round7='Nunito700', oswald='Oswald700', serif='Playfair900').items()}

# per-frame log of every lyric text bbox (for QA: safe zone / clipping / contrast)
TEXT_LOG = []


def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32) / 255  # BGR


# ---------------------------------------------------------------- easing
def clamp01(x): return float(min(1.0, max(0.0, x)))
def ease_out_cubic(x): x = clamp01(x); return 1 - (1 - x) ** 3
def ease_in_cubic(x): x = clamp01(x); return x ** 3
def ease_in_out(x): x = clamp01(x); return 3 * x * x - 2 * x ** 3
def ease_out_back(x, s=1.7):
    x = clamp01(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2


# ---------------------------------------------------------------- text
@functools.lru_cache(maxsize=4096)
def font(name, size):
    return ImageFont.truetype(FONTS[name], int(size))


@functools.lru_cache(maxsize=2048)
def text_alpha(text, fname, size, stroke=0, tracking=0):
    """Alpha mask (float32 0..1) tightly cropped to the ink of `text`. Returns (alpha, (ox, oy)) where ox, oy
    is the offset of the ink box relative to the pen origin, so baseline-consistent stacking is possible."""
    f = font(fname, size)
    if tracking:
        ws = [f.getbbox(c, stroke_width=stroke) for c in text]
        widths = [f.getlength(c) for c in text]
        W = int(sum(widths) + tracking * (len(text) - 1) + size + 4 * stroke)
    else:
        W = int(f.getlength(text) + size + 4 * stroke)
    Hh = int(size * 1.6 + 4 * stroke)
    im = Image.new('L', (W, Hh), 0); d = ImageDraw.Draw(im)
    if tracking:
        x = stroke
        for c, w in zip(text, widths):
            d.text((x, stroke), c, font=f, fill=255, stroke_width=stroke, stroke_fill=255); x += w + tracking
    else:
        d.text((stroke, stroke), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.01)
    if len(xs) == 0: return np.zeros((1, 1), np.float32), (0, 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return a[y0:y1, x0:x1].copy(), (x0 - stroke, y0 - stroke)


def outline_alpha(a, px):
    """hollow outline of an alpha mask"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    pad = np.pad(a, px + 2)
    return np.clip(cv2.dilate(pad, k) - cv2.erode(pad, k) * 1.0, 0, 1) - 0 * pad


def fit_size(text, fname, max_w, max_h, lo=10, hi=900, tracking=0):
    """largest font size whose ink box fits max_w x max_h"""
    while hi - lo > 1:
        m = (lo + hi) // 2
        a, _ = text_alpha(text, fname, m, 0, tracking)
        if a.shape[1] <= max_w and a.shape[0] <= max_h: lo = m
        else: hi = m
    return lo


# ---------------------------------------------------------------- compositing
def blit(canvas, alpha, color, cx, cy, scale=1.0, angle=0.0, opacity=1.0, src=None, log=None, mode='over'):
    """Place `alpha` (HxW) centred at (cx, cy) with scale/rotation. color: BGR vec (or src image HxWx3 same size as
    alpha via `src`). Returns the drawn bbox (x0,y0,x1,y1) in canvas coords. mode: over|add|screen."""
    if opacity <= 0.003 or scale <= 0.01: return None
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int); x1, y1 = np.ceil(corners.max(0)).astype(int)
    bx0, by0, bx1, by1 = max(x0, 0), max(y0, 0), min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
    if bx1 <= bx0 or by1 <= by0: return None
    M2 = M.copy(); M2[0, 2] -= bx0; M2[1, 2] -= by0
    size = (bx1 - bx0, by1 - by0)
    a = cv2.warpAffine(alpha, M2, size, flags=cv2.INTER_LINEAR, borderValue=0)[..., None] * opacity
    roi = canvas[by0:by1, bx0:bx1]
    if src is not None:
        col = cv2.warpAffine(src, M2, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        col = np.asarray(color, np.float32)[None, None, :]
    if mode == 'over': roi[:] = roi * (1 - a) + col * a
    elif mode == 'add': roi[:] = np.minimum(roi + col * a, 1)
    elif mode == 'screen': roi[:] = 1 - (1 - roi) * (1 - col * a)
    bb = (int(x0), int(y0), int(x1), int(y1))
    if log is not None: TEXT_LOG.append(dict(log, bbox=bb))
    return bb


def glow(canvas, alpha, color, cx, cy, scale=1.0, angle=0.0, radius=18, strength=0.9, opacity=1.0):
    pad = radius * 3
    a = np.pad(alpha, pad); a = cv2.GaussianBlur(a, (0, 0), radius) * strength
    blit(canvas, np.clip(a * 1.6, 0, 1), color, cx, cy, scale, angle, opacity, mode='screen')


def shadow(canvas, alpha, cx, cy, scale=1.0, angle=0, radius=14, dy=10, strength=0.75, opacity=1.0):
    pad = radius * 3
    a = cv2.GaussianBlur(np.pad(alpha, pad), (0, 0), radius) * strength
    blit(canvas, a, (0, 0, 0), cx, cy + dy, scale, angle, opacity)


def draw_text(canvas, text, fname, size, color, cx, cy, scale=1.0, angle=0.0, opacity=1.0, stroke=0,
              glow_col=None, glow_r=16, shadow_r=0, log=None, tracking=0):
    a, _ = text_alpha(text, fname, int(size), stroke, tracking)
    if shadow_r: shadow(canvas, a, cx, cy, scale, angle, radius=shadow_r, opacity=opacity)
    if glow_col is not None: glow(canvas, a, glow_col, cx, cy, scale, angle, radius=glow_r, opacity=opacity)
    lg = None if log is None else dict(log, text=text, color=list(map(float, color)), opacity=opacity)
    return blit(canvas, a, color, cx, cy, scale, angle, opacity, log=lg)


def rounded_rect_alpha(w, h, r, ss=2):
    im = Image.new('L', (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255


def paste(canvas, img, x, y, alpha=None, opacity=1.0):
    """paste BGR img at top-left (x,y) with optional alpha mask, clipped to canvas"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0: return
    sub = img[y0 - y:y1 - y, x0 - x:x1 - x]
    if alpha is None: a = opacity
    else: a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None] * opacity
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + sub * a


# ---------------------------------------------------------------- look
_vig = None
def vignette(img, amt=0.45):
    global _vig
    if _vig is None or _vig.shape[:2] != img.shape[:2]:
        h, w = img.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h / 2) / (h * 0.62)) ** 2)
        _vig = np.clip(1 - np.clip(r - 0.45, 0, 1) ** 1.6, 0, 1)[..., None]
    return img * (1 - amt + amt * _vig)


_grain = None
def grain(img, fi, amt=0.035):
    global _grain
    if _grain is None:
        rng = np.random.default_rng(7)
        _grain = [cv2.GaussianBlur(rng.normal(0, 1, (CH // 2, CW // 2)).astype(np.float32), (0, 0), 0.6) for _ in range(8)]
    g = _grain[fi % 8]
    if g.shape != img.shape[:2]: g = cv2.resize(g, (img.shape[1], img.shape[0]))
    return np.clip(img + g[..., None] * amt, 0, 1)


def grade(img, red=0.0, dark=0.0, sat=1.0):
    """warm highlights, lifted blacks; red>0 adds a red cast; dark>0 crushes exposure (for text-behind scenes)"""
    x = img
    lum = x.mean(2, keepdims=True)
    if sat != 1.0: x = lum + (x - lum) * sat
    hi = np.clip((lum - 0.45) / 0.55, 0, 1)
    x = x + hi * np.array([-0.035, 0.0, 0.05], np.float32)  # warm highlights (BGR)
    x = 0.045 + x * 0.94  # lift blacks
    x = x * np.array([0.97, 0.99, 1.02], np.float32)
    if red:
        r = np.array([1 - 0.45 * red, 1 - 0.38 * red, 1 + 0.18 * red], np.float32)
        x = x * r
    if dark: x = x * (1 - dark)
    return np.clip(x, 0, 1)


def to_f(img_u8): return img_u8.astype(np.float32) / 255


# ---------------------------------------------------------------- hit fx
def rgb_split(img, px):
    if px < 0.5: return img
    p = int(round(px)); out = img.copy()
    out[:, p:, 2] = img[:, :-p, 2]; out[:, :-p, 0] = img[:, p:, 0]
    return out


def shake_offset(t, t0, amp=14, dur=0.18, seed=0):
    if t < t0 or t > t0 + dur: return 0.0, 0.0
    k = 1 - (t - t0) / dur
    rng = np.random.default_rng(int(t * 1000) + seed)
    return float(rng.uniform(-1, 1) * amp * k), float(rng.uniform(-1, 1) * amp * k)


def shift(img, dx, dy):
    if abs(dx) < 0.5 and abs(dy) < 0.5: return img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def motion_blur_h(img, n):
    n = int(n)
    if n < 3: return img
    k = np.ones((1, n), np.float32) / n
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- shapes
def star_alpha(r_out, r_in, n, ss=3):
    s = int(r_out * 2 + 4)
    im = Image.new('L', (s * ss, s * ss), 0)
    c = s * ss / 2; pts = []
    for i in range(2 * n):
        r = (r_out if i % 2 == 0 else r_in) * ss; a = np.pi * i / n - np.pi / 2
        pts.append((c + r * np.cos(a), c + r * np.sin(a)))
    ImageDraw.Draw(im).polygon(pts, fill=255)
    return np.asarray(im.resize((s, s), Image.LANCZOS), np.float32) / 255


def ring_alpha(r, width, ss=3):
    s = int(2 * r + width * 2 + 4)
    im = Image.new('L', (s * ss, s * ss), 0); c = s * ss / 2
    ImageDraw.Draw(im).ellipse([c - r * ss, c - r * ss, c + r * ss, c + r * ss], outline=255, width=int(width * ss))
    return np.asarray(im.resize((s, s), Image.LANCZOS), np.float32) / 255


def disc_alpha(r, ss=3):
    s = int(2 * r + 4)
    im = Image.new('L', (s * ss, s * ss), 0); c = s * ss / 2
    ImageDraw.Draw(im).ellipse([c - r * ss, c - r * ss, c + r * ss, c + r * ss], fill=255)
    return np.asarray(im.resize((s, s), Image.LANCZOS), np.float32) / 255
