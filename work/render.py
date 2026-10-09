"""Frame-accurate renderer: composites every output frame in numpy/OpenCV from the
120 fps optical-flow source and pipes it to ffmpeg (H.264 ~20 Mbps + AAC 320k).

usage: python3 render.py OUT.mp4 [--frames a:b] [--stills dir n1,n2,...]
"""
import argparse
import math
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import edl
from edl import FPS, NFRAMES, S, bt, src_time
from sources import SHOTS

W, H = 1080, 1920
FONT_SERIF = "/home/user/Capstone/assets/fonts/BodoniModa.ttf"
FONT_SANS = "/home/user/Capstone/assets/fonts/Inter.ttf"
RNG = np.random.default_rng(7)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", default=None)
ap.add_argument("--stills", default=None)
ap.add_argument("--list", default="")
args = ap.parse_args()

# ----------------------------------------------------------------------------- source
print("loading src120.mp4 ...", file=sys.stderr)
cap = cv2.VideoCapture("src120.mp4")
SRC = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    SRC.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
SRC = np.stack(SRC)
NSRC = len(SRC)
SH, SW = SRC.shape[1:3]

# real cuts inside the interpolated source (mean abs diff spike) -> guard rails
_d = np.array([np.abs(SRC[i, ::8, ::8].astype(np.int16) - SRC[i - 1, ::8, ::8]).mean() for i in range(1, NSRC)])
SRC_CUTS = [i + 1 for i in np.where(_d > 25)[0]]

# --------------------------------------------------------- per-source exposure / WB match
STATS = {}
for name, (a, b, *_rest) in SHOTS.items():
    fr = SRC[int(a * 120):int(b * 120):12].astype(np.float32) / 255
    m = fr.reshape(-1, 3).mean(0)
    luma = float((fr @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())
    STATS[name] = (m, luma)
TARGET_L = 0.36
LOOKS = {}
for name, (m, luma) in STATS.items():
    # exposure: move 55% of the way (in log space) to a common mid level -> consistent
    # but keeps night scenes darker than day
    ev = 0.55 * math.log2(TARGET_L / luma)
    gain = 2 ** ev
    # white balance: 35% gray-world correction (keeps mood, tames the orange candle cast)
    gw = m.mean() / m
    wb = 1 + 0.35 * (gw - 1)
    LOOKS[name] = (gain * wb).astype(np.float32)


# ------------------------------------------------------------------------------ grade LUT
def build_lut():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    # soft S-curve + lifted blacks + rolled-off highlights
    s = x + 0.10 * np.sin(2 * np.pi * x) * -0.5 * (1 - x) + 0.0
    s = 1 / (1 + np.exp(-6.2 * (x - 0.5)))
    s = (s - s[0]) / (s[-1] - s[0])
    s = 0.62 * s + 0.38 * x
    s = 0.055 + 0.915 * s
    r = s + 0.030 * np.sin(np.pi * x) * x * 1.6           # warm highlights
    g = s + 0.010 * np.sin(np.pi * x) * x
    b = s - 0.045 * x ** 1.2 + 0.030 * (1 - x) ** 3       # gold highs, slightly cool shadows
    return np.clip(np.stack([r, g, b], 1), 0, 1)


LUT = (build_lut() * 255).astype(np.uint8)  # 256x3


def apply_lut(img8):
    out = np.empty_like(img8)
    for c in range(3):
        out[..., c] = cv2.LUT(img8[..., c], LUT[:, c])
    return out


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 * 0.75 + ((yy - H / 2) / (H / 2)) ** 2)
VIGNETTE = (1 - 0.30 * np.clip((rr - 0.45) / 0.85, 0, 1) ** 1.6).astype(np.float32)[..., None]


# ------------------------------------------------------------------------------- helpers
def ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_in_out(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def src_frame(name, t, flip):
    a, b = SHOTS[name][0], SHOTS[name][1]
    t = min(max(t, a), b)
    i = int(round(t * 120))
    img = SRC[min(i, NSRC - 1)]
    return img[:, ::-1] if flip else img


def place(img, scale=1.0, dx=0.0, dy=0.0, rot=0.0):
    """source crop (598x1062) -> 1080x1920 with zoom/translate/rotate, single lanczos resample"""
    base = max(W / SW, H / SH)
    k = base * scale
    M = cv2.getRotationMatrix2D((SW / 2, SH / 2), rot, k)
    M[0, 2] += W / 2 - SW / 2 + dx
    M[1, 2] += H / 2 - SH / 2 + dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)


def hblur(img, k):
    k = int(k)
    if k < 3:
        return img
    return cv2.blur(img, (k | 1, 1))


def rgb_shift(img, px, vertical=0):
    px = int(round(px))
    if px == 0:
        return img
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], px, axis=1)
    out[..., 2] = np.roll(img[..., 2], -px, axis=1)
    if vertical:
        out[..., 0] = np.roll(out[..., 0], vertical, axis=0)
    return out


def chroma_ab(img, strength):
    """radial chromatic aberration: scale R up / B down around the centre"""
    if strength <= 0.05:
        return img
    out = img.copy()
    for c, sgn in ((0, 1), (2, -1)):
        k = 1 + sgn * strength / 1000
        M = np.float32([[k, 0, W / 2 * (1 - k)], [0, k, H / 2 * (1 - k)]])
        out[..., c] = cv2.warpAffine(img[..., c], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out


def shake_offset(n_since, seed, amp=22.0, frames=9):
    if n_since < 0 or n_since >= frames:
        return 0.0, 0.0, 0.0
    r = np.random.default_rng(seed + n_since)
    d = (1 - n_since / frames) ** 2
    return amp * d * r.uniform(-1, 1), amp * d * r.uniform(-1, 1), 0.8 * d * r.uniform(-1, 1)


# ---------------------------------------------------------------------------- typography
def text_layer(text, font_path, size, tracking, weight=None, fill=(255, 244, 228)):
    f = ImageFont.truetype(font_path, size)
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
    widths = [f.getbbox(ch)[2] - f.getbbox(ch)[0] if ch != " " else size * 0.32 for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    asc, desc = f.getmetrics()
    img = Image.new("L", (int(total + 40), asc + desc + 40), 0)
    d = ImageDraw.Draw(img)
    x = 20
    for ch, w in zip(text, widths):
        if ch != " ":
            d.text((x - f.getbbox(ch)[0], 20), ch, font=f, fill=255)
        x += w + tracking
    a = np.asarray(img).astype(np.float32) / 255
    return a, fill


def composite_text(frame_f, alpha, color, cx, cy, opacity, glow=0.0, shift=0):
    if opacity <= 0.001:
        return frame_f
    h, w = alpha.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, W), min(y0 + h, H)
    a = alpha[y1 - y0:y2 - y0, x1 - x0:x2 - x0] * opacity
    region = frame_f[y1:y2, x1:x2]
    col = np.array(color, np.float32) / 255
    if glow > 0:
        g = cv2.GaussianBlur(a, (0, 0), 18) * glow
        region += g[..., None] * col * 0.9
    # soft shadow for legibility on bright shots
    sh = cv2.GaussianBlur(a, (0, 0), 6) * 0.35
    region *= (1 - sh[..., None])
    if shift:
        for c, s in ((0, shift), (2, -shift)):
            ac = np.roll(a, s, axis=1)
            region[..., c] = region[..., c] * (1 - ac) + col[c] * ac
        region[..., 1] = region[..., 1] * (1 - a) + col[1] * a
    else:
        region[:] = region * (1 - a[..., None]) + col * a[..., None]
    frame_f[y1:y2, x1:x2] = region
    return frame_f


TITLE_LAYERS = {}


def title_alpha(tracking):
    key = int(tracking)
    if key not in TITLE_LAYERS:
        TITLE_LAYERS[key] = text_layer(edl.TITLE, FONT_SERIF, 66, tracking, weight=b"Regular")
    return TITLE_LAYERS[key]


DROP_TXT = text_layer("AUJLA", FONT_SANS, 215, 4, weight=b"Black")
END_TXT = text_layer(edl.END_HANDLE, FONT_SANS, 82, 8, weight=b"Medium")
END_SUB = text_layer(edl.TITLE, FONT_SERIF, 42, 16, weight=b"Regular", fill=(236, 204, 150))


# ------------------------------------------------------------------------- shot render
def render_shot_frame(s, n, extra_t=None):
    """raw (ungraded) frame of shot s at output frame n (n may lie just outside the shot
    for transitions)."""
    t = n / FPS if extra_t is None else extra_t
    k = n - s["f0"]                       # frames since shot start
    kn = s["f1"] - 1 - n                  # frames until shot end
    L = s["f1"] - s["f0"]
    u = (n - s["f0"]) / max(L - 1, 1)
    img = src_frame(s["src"], src_time(s, t), s["flip"])
    img = (np.clip(img.astype(np.float32) * LOOKS[s["src"]] / 255, 0, 1) * 255).astype(np.uint8)

    scale, dx, dy, rot = 1.0, 0.0, 0.0, 0.0
    d = s["drift"]
    if d == "hero":
        scale = 1.0 + 0.10 * ease_in_out(u)
    elif d == "in":
        scale = 1.02 + 0.035 * u
    elif d == "out":
        scale = 1.055 - 0.035 * u
    elif d == "breakdown":
        scale = 1.0 + 0.09 * ease_in_out(u)
    elif d == "accel":
        scale = 1.02 + 0.16 * u ** 2.2
    elif d == "none":
        scale = 1.02
    fx = s["fx"]
    if "drop" in fx:
        scale = max(scale, 1.07)          # head-room for shake
    if "zoom_punch" in fx:
        scale *= 1.0 + 0.15 * ease_out(k / 4)
    if "shake" in fx:
        sx, sy, sr = shake_offset(k, s["f0"])
        dx += sx
        dy += sy
        rot += sr

    # ---- transitions (geometry part)
    blur = 0
    if s["fx_in"] in ("whip_in", "whip_in_l") and k < 4:
        e = 1 - ease_out((k + 1) / 4)
        dx += (1 if s["fx_in"] == "whip_in" else -1) * e * W * 0.9
        blur = 140 * e + 10
    if s["fx_out"] == "whip_out" and kn < 3:
        e = ease_in_out((3 - kn) / 3)
        dx -= e * W * 0.9
        blur = 140 * e + 10
    if s["fx_in"] == "zoom_through_in" and k < 4:
        scale *= 1 + 0.45 * (1 - ease_out((k + 1) / 4))
    zt_out = s["fx_out"] == "zoom_through_out" and kn < 3

    if zt_out:
        e = (3 - kn) / 3
        acc = np.zeros((H, W, 3), np.float32)
        for j in range(5):
            acc += place(img, scale * (1 + 0.55 * e * (1 + j * 0.12)), dx, dy, rot)
        out = (acc / 5).astype(np.uint8)
    else:
        out = place(img, scale, dx, dy, rot)
    if blur:
        out = hblur(out, blur)
    return out


def grade(img8, t, drop=False, mood=0.0):
    g = apply_lut(img8).astype(np.float32) / 255
    if mood > 0:   # breakdown: desaturate + darken slightly
        l = g @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        g = g * (1 - 0.45 * mood) + l[..., None] * 0.45 * mood
        g *= 1 - 0.08 * mood
    # saturation touch
    l = (g @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    g = l + (g - l) * 1.06
    # bloom on highlights (quarter-res blur)
    small = cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hl = np.clip((small - 0.74) / 0.26, 0, 1)
    hl = cv2.GaussianBlur(hl, (0, 0), 9)
    hl = cv2.resize(hl, (W, H), interpolation=cv2.INTER_LINEAR)
    g = g + hl * np.array([1.0, 0.82, 0.58], np.float32) * 0.17
    return g


def finish(g, n):
    g = g * VIGNETTE
    # film grain: half-res gaussian, luma weighted to the mid-tones
    gr = RNG.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
    gr = cv2.resize(gr, (W, H), interpolation=cv2.INTER_LINEAR)
    l = g.mean(2, keepdims=True)
    g = g + gr[..., None] * 0.022 * (0.35 + np.sin(np.clip(l, 0, 1) * np.pi) * 0.65)
    # gentle unsharp after the upscale
    bl = cv2.GaussianBlur(g, (0, 0), 1.3)
    g = g + (g - bl) * 0.45
    # ordered dither before 8-bit to avoid banding in dark gradients
    g = g + (RNG.random((H, W, 1), np.float32) - 0.5) / 255
    return (np.clip(g, 0, 1) * 255 + 0.5).astype(np.uint8)


def light_leak(g, k, total=7):
    """warm leak sweeping across the first frames of the shot (peaks at the cut)"""
    e = max(0.0, 1 - k / total)
    if e <= 0:
        return g
    cx = W * (0.15 + 0.9 * (k / total))
    cy = H * 0.35
    m = np.exp(-(((xx - cx) / (W * 0.55)) ** 2 + ((yy - cy) / (H * 0.45)) ** 2))
    leak = m[..., None] * np.array([1.0, 0.55, 0.18], np.float32) * 0.85 * e
    return 1 - (1 - g) * (1 - leak)   # screen blend


def glitch(img8, k, seed):
    if k > 2:
        return img8
    r = np.random.default_rng(seed + k)
    out = img8.copy()
    amp = [90, 50, 18][k]
    y = 0
    while y < H:
        hgt = int(r.integers(12, 110))
        if r.random() < 0.55:
            out[y:y + hgt] = np.roll(out[y:y + hgt], int(r.integers(-amp, amp)), axis=1)
        y += hgt
    return rgb_shift(out, [26, 14, 6][k], vertical=[8, 4, 0][k])


def end_card(n, s):
    k = n - s["f0"]
    L = s["f1"] - s["f0"]
    t = n / FPS
    # background: last drop image, heavily blurred and darkened, fading to black
    base = SRC[int(SHOTS["LOWCU"][0] * 120 + 60)]
    base = place(base, 1.15)
    base = cv2.GaussianBlur(base, (0, 0), 28).astype(np.float32) / 255
    bg_fade = 0.30 * (1 - ease_in_out(k / (L - 1)))
    g = base * LOOKS["LOWCU"] * bg_fade
    g = apply_lut((np.clip(g, 0, 1) * 255).astype(np.uint8)).astype(np.float32) / 255 * min(1, 0.35 + bg_fade * 2)
    # text: blooms in, pulses on the final bass hit, fades to black on the last frames
    t_pulse = bt(35)
    pulse = math.exp(-max(0.0, t - t_pulse) * 9) if t >= t_pulse - 1 / FPS else 0.0
    appear = ease_out((k + 1) / 5)
    fade = 1 - ease_in_out((k - (L - 6)) / 5) if k >= L - 6 else 1.0
    op = appear * fade
    a, col = END_TXT
    g = composite_text(g, a, col, W / 2, H * 0.48, op, glow=0.9 + 1.6 * pulse + 1.2 * (1 - appear))
    a2, col2 = END_SUB
    g = composite_text(g, a2, col2, W / 2, H * 0.48 + 104, op * 0.85, glow=0.4 + 0.8 * pulse)
    g = g * (1 + 0.25 * pulse)
    return g


# --------------------------------------------------------------------------- per frame
def shot_at(n):
    for i, s in enumerate(S):
        if s["f0"] <= n < s["f1"]:
            return i, s
    raise ValueError(n)


def render(n):
    i, s = shot_at(n)
    k = n - s["f0"]
    t = n / FPS
    if s["src"] == "ENDCARD":
        out = finish(end_card(n, s), n)
        # final fade to true black (grain/dither included) over the last 4 frames
        kl = s["f1"] - 1 - n
        if kl < 4:
            out = (out.astype(np.float32) * ease_in_out(kl / 3)).astype(np.uint8)
        return out

    img = render_shot_frame(s, n)

    # whip transitions: composite the neighbour shot sliding in/out alongside
    if s["fx_in"] in ("whip_in", "whip_in_l") and k < 4:
        prev = S[i - 1]
        d = 1 if s["fx_in"] == "whip_in" else -1
        e = 1 - ease_out((k + 1) / 4)
        pimg = place_src_tail(prev, n, d * (e * W * 0.9 - W), 140 * e + 10)
        mask = (xx < e * W * 0.9) if d > 0 else (xx > W - e * W * 0.9)
        img = np.where(mask[..., None], pimg, img)
    if s["fx_out"] == "whip_out" and k >= (s["f1"] - s["f0"]) - 3:
        nxt = S[i + 1]
        kn = s["f1"] - 1 - n
        e = ease_in_out((3 - kn) / 3)
        nimg = place_src_head(nxt, n, W * 0.9 * (1 - e) + W * 0.1, 140 * e + 10)
        mask = (xx > W - e * W * 0.9)[..., None]
        img = np.where(mask, nimg, img)

    if s["fx_in"] == "glitch":
        img = glitch(img, k, s["f0"])
    if s["fx_in"] == "rgb_split" and k < 5:
        img = rgb_shift(img, 30 * (1 - k / 5) ** 2)

    drop = "drop" in s["fx"]
    mood = 0.0
    if "breakdown_mood" in s["fx"]:
        mood = ease_in_out(k / 12)
    g = grade(img, t, drop=drop, mood=mood)

    if drop:
        # chromatic aberration: subtle constant + spike on hits
        hit = 0.0
        if "shake" in s["fx"] or "flash3" in s["fx"] or "flash2" in s["fx"]:
            hit = max(0.0, 1 - k / 6)
        g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
        g = chroma_ab(g8, 3.0 + 9.0 * hit).astype(np.float32) / 255

    if s["fx_in"] == "light_leak":
        g = light_leak(g, k)

    # --- text
    if "text_title" in s["fx"]:
        ta, tb = bt(2), s["t1"]
        if ta - 0.01 <= t < tb:
            u = (t - ta) / (tb - ta)
            a, col = title_alpha(14 + 16 * ease_out(u))
            op = ease_out(min(1, (t - ta) / 0.18)) * (1 - ease_in_out((t - (tb - 0.20)) / 0.20) if t > tb - 0.20 else 1)
            g = composite_text(g, a, col, W / 2, H * 0.80, op, glow=0.35)
    # drop text spans shots 24-26 (2 beats from the drop)
    td0, td1 = bt(30), bt(32)
    if td0 - 0.01 <= t < td1 - 0.5 / FPS:
        kk = n - edl.bf(30)
        a, col = DROP_TXT
        sc = 1.0 + 0.2 * (1 - ease_out((kk + 1) / 4))
        if sc != 1.0:
            a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        op = 1.0 if t < td1 - 0.12 else max(0.0, (td1 - t) / 0.12)
        sx, sy, _ = shake_offset(n - s["f0"], s["f0"], amp=14) if "shake" in s["fx"] else (0, 0, 0)
        g = composite_text(g, a, (255, 250, 242), W / 2 + sx, H * 0.50 + sy, op,
                           glow=0.25, shift=int(10 * max(0, 1 - kk / 5)) + 2)

    # --- flash frames (white, on the hardest hits)
    if "flash3" in s["fx"] and k < 3:
        a = [0.92, 0.55, 0.22][k]
        g = g * (1 - a) + a
    if "flash2" in s["fx"] and k < 2:
        a = [0.85, 0.35][k]
        g = g * (1 - a) + a

    # --- fade from black on beat 0
    if "fade_from_black" in s["fx"]:
        fa, fb = bt(0) - 0.5 / FPS, bt(1)
        if t < fb:
            g = g * ease_in_out((t - fa) / (fb - fa)) if t > fa else g * 0
    out = finish(g, n)
    if "fade_from_black" in s["fx"] and t < bt(0) - 0.5 / FPS:
        out[:] = 0
    return out


def place_src_tail(s, n, dx, blur):
    t = min(n / FPS, s["t1"] - 1 / FPS)
    img = src_frame(s["src"], src_time(s, t), s["flip"])
    img = (np.clip(img.astype(np.float32) * LOOKS[s["src"]] / 255, 0, 1) * 255).astype(np.uint8)
    return hblur(place(img, 1.04, dx), blur)


def place_src_head(s, n, dx, blur):
    img = src_frame(s["src"], s["t_in"], s["flip"])
    img = (np.clip(img.astype(np.float32) * LOOKS[s["src"]] / 255, 0, 1) * 255).astype(np.uint8)
    return hblur(place(img, 1.04, dx), blur)


# --------------------------------------------------------------------- guard: source cuts
for s in S:
    if s["src"] == "ENDCARD":
        continue
    a = src_time(s, s["t0"])
    b = src_time(s, s["t1"] - 1 / FPS)
    lo, hi = SHOTS[s["src"]][0], SHOTS[s["src"]][1]
    assert lo - 1e-6 <= a and b <= hi + 1e-6, f"shot {s} uses {a:.3f}-{b:.3f} outside {lo}-{hi}"
    ia, ib = int(round(a * 120)), int(round(b * 120))
    bad = [c for c in SRC_CUTS if ia < c <= ib]
    assert not bad, f"shot {s['src']}@{s['t_in']} crosses a source cut at frame {bad}"

# ---------------------------------------------------------------------------- output
if args.stills:
    import os
    os.makedirs(args.stills, exist_ok=True)
    for n in [int(x) for x in args.list.split(",") if x]:
        Image.fromarray(render(n)).save(f"{args.stills}/f{n:03d}.png")
    sys.exit(0)

fr = range(NFRAMES)
if args.frames:
    a, b = map(int, args.frames.split(":"))
    fr = range(a, b)

cmd = ["ffmpeg", "-v", "error", "-y",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
       "-i", "segment.wav",
       "-map", "0:v", "-map", "1:a",
       "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p",
       "-b:v", "20M", "-maxrate", "26M", "-bufsize", "40M", "-x264-params", "aq-mode=3:deblock=-1,-1",
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
       "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
       "-t", f"{len(fr) / FPS:.6f}", "-movflags", "+faststart", args.out]
if args.frames:  # quick partial render, video only
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", args.out]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for n in fr:
    p.stdin.write(render(n).tobytes())
    if n % 30 == 0:
        print(f"frame {n}/{NFRAMES}", file=sys.stderr)
p.stdin.close()
sys.exit(p.wait())
