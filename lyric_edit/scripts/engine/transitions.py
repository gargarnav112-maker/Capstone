"""L4 -> L5: a long translucent maroon DUPATTA sweeps across the frame (sine-
warped band, fabric folds, gold embroidered border, motion blur) revealing L5
underneath. L5 -> L6: iris opening into the jhummar circle."""
import math, functools
import numpy as np, cv2
from .core import *

D0, D1 = 14.22, 14.62          # dupatta sweep window
IRIS0, IRIS1 = 17.98, 18.20
DIR = np.array([0.944, 0.330]); HW = 330


@functools.lru_cache(maxsize=1)
def uv():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx * DIR[0] + yy * DIR[1]; v = -xx * DIR[1] + yy * DIR[0]
    return u, v


def dupatta(a, b, t):
    """a = outgoing (L4), b = incoming (L5)."""
    u, v = uv()
    pos = lerp(-HW - 300, 1654 + HW + 300, ease_io(prog(t, D0, D1)))
    uw = u - pos + 70 * np.sin(v / 210 + t * 7) + 25 * np.sin(v / 70 - t * 11)
    inside = np.clip((HW - np.abs(uw)) / 6, 0, 1)
    reveal = np.clip(0.5 - uw / 60, 0, 1)                    # behind the leading half -> incoming scene
    out = a * (1 - reveal[..., None]) + b * reveal[..., None]
    folds = 0.62 + 0.38 * np.sin(uw / 34 + 1.8 * np.sin(v / 260 + t * 5))
    sheer = 0.55 + 0.3 * np.cos(uw / HW * math.pi / 2) ** 2
    fab = (MAROON * 1.7 + RED * 0.25)[None, None, :] * folds[..., None]
    alpha = inside * sheer
    # gold border + embroidered dots
    border = np.clip(1 - np.abs(np.abs(uw) - HW + 9) / 5, 0, 1)
    dots = np.clip(1 - np.abs(np.abs(uw) - HW + 30) / 5, 0, 1) * ((np.abs(v) % 34) < 10)
    col = fab * (1 - border[..., None]) + GOLD * border[..., None]
    col = col * (1 - dots[..., None]) + PALEGOLD * dots[..., None]
    alpha = np.maximum(alpha, (border + dots).clip(0, 1) * inside.clip(0.0, 1) + border * 0.9)
    # motion blur along the sweep direction (half res)
    k = 31; ker = np.zeros((k, k), np.float32)
    cv2.line(ker, (0, int(k / 2 - DIR[1] * k / 2)), (k - 1, int(k / 2 + DIR[1] * k / 2)), 1.0, 1)
    ker /= ker.sum()
    def mb(x):
        s = cv2.resize(x, (W // 2, H // 2)); s = cv2.filter2D(s, -1, ker)
        return cv2.resize(s, (W, H))
    alpha = np.clip(mb(alpha.astype(np.float32)), 0, 1); col = mb(col.astype(np.float32))
    out = out * (1 - alpha[..., None]) + col * alpha[..., None]
    return out


def iris(a, b, t, center):
    r = 1500 * ease_in(prog(t, IRIS0, IRIS1), 2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    m = np.clip((r - d) / 3, 0, 1)[..., None]
    out = a * (1 - m) + b * m
    ring = np.clip(1 - np.abs(d - r) / 6, 0, 1)[..., None]
    return out + ring * GOLD * 1.2
