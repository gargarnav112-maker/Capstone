"""L2 'Dass ki kar laina kavan ni mera baazan aala raakha'
What can anyone do — my keeper is the falcon-master.
A frosted-glass phone lyric card tilts in 3D and scrolls; each word fills
left->right exactly across its sung start..end (karaoke). On BAAZAN an original
falcon silhouette launches across the screen, shedding feathers (-> L3)."""
import math, functools
import numpy as np, cv2
from .core import *
from . import falcon

LINE = 1
CW, CH = 820, 1010
ROWS = [(0, ["VI", "NA", "THAKA"]), (1, ["DASS", "KI", "KAR", "LAINA"]), (1, ["KAVAN", "NI", "MERA"]),
        (1, ["BAAZAN", "AALA", "RAAKHA"]), (2, ["RAHAN", "VICH", "RODE"])]
LH = 158; VIEW_Y0, VIEW_Y1 = 335, CH - 30; FOCUS_Y = 600


@functools.lru_cache(maxsize=1)
def rows_layout():
    out = []; size = min(fit_size("BAAZAN AALA RAAKHA", 'anton', CW - 120, 128), 128)
    for li, row in ROWS:
        ws = LINE_WORDS[li]
        words = [next(w for w in ws if w['text'] == r and (li != 0 or w['idx'] >= 7) and (li != 2 or w['idx'] < 3)) for r in row]
        f = font('anton', size); sp = f.getlength(' ')
        widths = [f.getlength(r) for r in row]
        x = 70; items = []
        for w, wd in zip(words, widths):
            items.append((w, x, wd)); x += wd + sp
        out.append(dict(line=li, items=items, size=size))
    return out


def active_row(t):
    k = 1
    for i, r in enumerate(rows_layout()):
        if r['line'] == LINE and t >= r['items'][0][0]['start'] - 0.12: k = i
    return k


def scroll_y(t):
    rows = rows_layout()
    y = FOCUS_Y - 1 * LH
    for i in (2, 3):
        t0 = rows[i]['items'][0][0]['start'] - 0.22
        y -= LH * ease_io(prog(t, t0, t0 + 0.28))
    return y


def rounded_mask(w, h, r):
    m = np.zeros((h * 2, w * 2), np.uint8)
    R = r * 2
    cv2.rectangle(m, (R, 0), (w * 2 - R, h * 2), 255, -1); cv2.rectangle(m, (0, R), (w * 2, h * 2 - R), 255, -1)
    for cx, cy in ((R, R), (w * 2 - R, R), (R, h * 2 - R), (w * 2 - R, h * 2 - R)):
        cv2.circle(m, (cx, cy), R, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


@functools.lru_cache(maxsize=1)
def card_static():
    return rounded_mask(CW, CH, 58)


def put_text(img, text, key, size, x, y, col, op=1.0, tracking=0.0, anchor='l'):
    a, base = text_alpha(text, key, size, tracking)
    if anchor == 'r': x = x - a.shape[1]
    if anchor == 'c': x = x - a.shape[1] / 2
    draw_sprite(img, a, M_affine(x, y, 1, 1, 0, 0, base), col, op)


def card_content(t):
    img = np.zeros((CH, CW, 3), np.float32); A = np.zeros((CH, CW), np.float32)
    # header
    put_text(img, "NOW PLAYING", 'mono', 22, 54, 70, CREAM, 0.7, 3)
    put_text(img, "89.6 BPM", 'mono', 22, CW - 54, 70, GOLD, 0.9, 2, 'r')
    put_text(img, "ASHKE", 'bebas', 84, 54, 170, CREAM, 1.0, 4)
    put_text(img, "KARAN AUJLA", 'thin', 30, 56, 214, CREAM, 0.85, 9)
    # progress
    song = SONG_T0 + t; pr = t / DUR
    y = 252; x0, x1 = 56, CW - 56
    cv2.rectangle(img, (x0, y - 2), (x1, y + 2), tuple(float(v) * 0.35 for v in CREAM), -1)
    xf = int(lerp(x0, x1, pr))
    cv2.rectangle(img, (x0, y - 2), (xf, y + 2), tuple(float(v) for v in GOLD), -1)
    cv2.circle(img, (xf, y), 9, tuple(float(v) for v in PALEGOLD), -1, cv2.LINE_AA)
    put_text(img, f"{int(song // 60)}:{song % 60:05.2f}", 'mono', 20, x0, y + 40, CREAM, 0.6)
    rem = SONG_T0 + DUR - song
    put_text(img, f"-0:{rem:05.2f}", 'mono', 20, x1, y + 40, CREAM, 0.6, 0, 'r')
    # lyrics viewport
    lyr = np.zeros_like(img)
    sy = scroll_y(t); ar = active_row(t)
    for i, row in enumerate(rows_layout()):
        yc = sy + i * LH
        if yc < VIEW_Y0 - 120 or yc > VIEW_Y1 + 120: continue
        dist = abs(i - ar)
        row_op = 1.0 if row['line'] == LINE else 0.55
        blur_ctx = dist > 0
        for w, x, wd in row['items']:
            a, base = text_alpha(w['text'], 'anton', row['size'])
            M = M_affine(x, yc, 1, 1, 0, 0, base - row['size'] * 0.36)
            sung = row['line'] < LINE or (row['line'] == LINE and t >= w['end'])
            p = 1.0 if sung else (prog(t, w['start'], w['end']) if row['line'] == LINE else 0.0)
            draw_sprite(lyr, a, M, CREAM, 0.26 * row_op)
            if p > 0:
                xs = np.arange(a.shape[1], dtype=np.float32)
                ramp = np.clip((p * a.shape[1] - xs) / 14 + 0.5, 0, 1)[None, :]
                active = row['line'] == LINE and not sung
                col = GOLD * 1.15 if active else CREAM
                if row['line'] != LINE: col = CREAM; ramp = ramp * 0.45
                ga, gbb = draw_sprite(lyr, a * ramp, M, col, 1.0)
                if active: glow(lyr, ga, gbb, SAFFRON, 14, 0.8)
    # fade top/bottom of viewport
    yy = np.arange(CH, dtype=np.float32)
    fade = np.clip((yy - VIEW_Y0) / 110, 0, 1) * np.clip((VIEW_Y1 - yy) / 110, 0, 1)
    img += lyr * fade[:, None, None]
    A = card_static()
    return img, A


def card_homography(t):
    k, ph = beat_phase(t)
    ay = math.radians(9 * math.sin(t * 1.25) + 2.5 * math.exp(-ph * 4) * (1 if k % 2 else -1))
    ax = math.radians(6 * math.cos(t * 0.9 + 0.4))
    az = math.radians(1.5 * math.sin(t * 0.7))
    cy = lerp(1290, 1200, prog(t, 3.68, 7.3))
    f = 2200.0
    pts3 = np.array([[-CW / 2, -CH / 2, 0], [CW / 2, -CH / 2, 0], [CW / 2, CH / 2, 0], [-CW / 2, CH / 2, 0]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    p = pts3 @ (Rz @ Rx @ Ry).T
    s = 1 + 0.02 * pulse(BEATS, t, 0.1)
    dst = np.stack([W / 2 + s * f * p[:, 0] / (f + p[:, 2]), cy + s * f * p[:, 1] / (f + p[:, 2])], 1).astype(np.float32)
    src = np.array([[0, 0], [CW, 0], [CW, CH], [0, CH]], np.float32)
    return cv2.getPerspectiveTransform(src, dst), dst


def render(t, fx):
    F = footage(); f = t - FOOT_OFF
    img = F.frame(f); cx, cy, fs = F.subject(f)
    bg, M = fullbleed(img, cx, cy, 1.45, H * 0.36 + 30 * math.sin(t * 0.6))
    bg = grade(bg, 0.85, 1.05, -0.04, MAROON, 0.25) * 0.78
    c = bg.copy()
    phulkari(c, t, 0.05)
    # glass card
    Hm, quad = card_homography(t)
    content, A = card_content(t)
    Aw = cv2.warpPerspective(A, Hm, (W, H), flags=cv2.INTER_LINEAR)
    Cw = cv2.warpPerspective(content, Hm, (W, H), flags=cv2.INTER_LINEAR)
    glass = blur_full(c, 26) * 0.72 + MAROON * 0.16 + 0.05
    # soft drop shadow
    sh = blur_full(Aw[..., None].repeat(3, 2), 30)[..., 0]
    c *= (1 - 0.5 * np.roll(sh, 30, 0))[..., None]
    a3 = Aw[..., None]
    c[:] = c * (1 - a3) + glass * a3
    # specular sheen across the glass
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sheen = np.exp(-(((xx * 0.5 + yy) / H - (0.2 + 0.1 * math.sin(t))) / 0.08) ** 2) * 0.08
    c += (sheen * Aw)[..., None]
    c += Cw * a3
    # border
    edge = np.clip(Aw - cv2.erode(Aw, np.ones((3, 3), np.uint8)), 0, 1)
    c[:] = c * (1 - 0.6 * edge[..., None]) + CREAM * 0.6 * edge[..., None]
    # falcon + feathers
    falcon.render_feathers(c, t)
    falcon.render_falcon(c, t)
    if falcon.T_FLY <= t < falcon.T_FLY + 0.12:
        fx['flash'] = max(fx.get('flash', 0), 0.3 * (1 - (t - falcon.T_FLY) / 0.12))
    return c
