"""Lyric-edit renderer. Usage: python3 scripts/render.py [--preview] [--frames a:b] [--out path]
Reads lyrics.json (word timings) + scenes.json (layout per time range) + beats.json."""
import sys, os, json, math, argparse, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx
from fx import CW, CH, FPS, hexc, ease_out_cubic, ease_out_back, ease_in_out, clamp01
import footage as F

D = fx.D
DUR = 16.0
NF = int(DUR * FPS)

LY = json.load(open(os.environ.get('LYRICS', D + 'lyrics.json')))
SC = json.load(open(os.environ.get('SCENES', D + 'scenes.json')))
BT = json.load(open(D + 'beats.json'))
BEATS = np.array(BT['beats']); DOWN = np.array(BT['downbeats'])
HANDLE = SC.get('handle', '@yourhandle')

WORDS = []
for li, line in enumerate(LY['lines']):
    for w in line['words']:
        WORDS.append(dict(w, line=li))
WORDS.sort(key=lambda w: w['s'])
for i, w in enumerate(WORDS): w['i'] = i
LINES = [[w for w in WORDS if w['line'] == li] for li in range(len(LY['lines']))]

RED = hexc('#C8102E'); DRED = hexc('#2A0306'); WHITE = np.ones(3, np.float32); BLACK = np.zeros(3, np.float32)
YEL = hexc('#FFD23F'); ORANGE = hexc('#FF7A55'); NEAR_BLACK = hexc('#0B0809')
IN_F = 5 / FPS    # in-animation length
OUT_F = 2 / FPS   # out-animation length


# cut on the line break: a scene boundary that falls just after a word start snaps back to that word
for _i in range(1, len(SC['scenes'])):
    b = SC['scenes'][_i]['start']
    near = [w['s'] for w in WORDS if b - 0.15 < w['s'] < b]
    if near:
        nb = max(near)
        for tr in SC.get('transitions', []):
            if abs(tr['t'] - b) < 1e-6: tr['t'] = nb
        SC['scenes'][_i]['start'] = nb; SC['scenes'][_i - 1]['end'] = nb


def scene_at(t):
    for s in SC['scenes']:
        if s['start'] <= t < s['end']: return s
    return SC['scenes'][-1]


def scene_words(s):
    ws = [w for w in WORDS if s['start'] - 1e-6 <= w['s'] < s['end']]
    # carry over a word that is still being sung across the cut (next word starts after the cut)
    prev = [w for w in WORDS if w['s'] < s['start'] - 1e-6]
    if prev and s['start'] - prev[-1]['s'] < 0.9 and (not ws or ws[0]['i'] == prev[-1]['i'] + 1):
        ws = [prev[-1]] + ws
    return ws


def next_start(w, s):
    """time the word must be gone by: next word start, capped by scene end"""
    nxt = WORDS[w['i'] + 1]['s'] if w['i'] + 1 < len(WORDS) else DUR
    return min(nxt, s['end'])


def current_word(t, s):
    ws = [w for w in scene_words(s) if w['s'] <= t + 1e-6]
    return ws[-1] if ws else None


def word_phase(t, w, s):
    """(p_in 0..1, p_out 0..1 where 1=fully gone)"""
    end = next_start(w, s)
    # hold: if the word is the last one before a pause, keep it on until the next word / scene end
    pin = clamp01((t - w['s']) / IN_F) if w['s'] >= s['start'] - 1e-6 else clamp01((t - s['start']) / IN_F + 0.6)
    nxt = WORDS[w['i'] + 1]['s'] if w['i'] + 1 < len(WORDS) else DUR
    carried = nxt > s['end'] + 1e-6 and s['end'] < DUR  # still sung after the cut: carried into the next scene
    pout = clamp01((t - (end - OUT_F)) / OUT_F) if end - w['s'] > OUT_F * 2 and not carried else 0.0
    return pin, pout


def in_scale(pin, w, h, cx, cy, angle=0.0, s_max=1.4, base=1.0):
    """140%->100% ease-out scale-in, capped so the word's rotated box never leaves the frame (8 px margin)"""
    c, sn = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
    bw, bh = (w * c + h * sn) * base, (w * sn + h * c) * base
    lim = min((2 * min(cx, CW - cx) - 16) / bw, (2 * min(cy, CH - cy) - 16) / bh)
    s0 = max(1.0, min(s_max, lim))
    return base * (s0 - (s0 - 1) * ease_out_cubic(pin))


def appear(pin):
    """opacity during the in-animation: visible from the very first frame of the word"""
    return 0.55 + 0.45 * ease_out_cubic(pin)


def beat_pulse(t, decay=0.12):
    d = t - BEATS[BEATS <= t + 1e-6]
    if len(d) == 0: return 0.0
    return float(math.exp(-d.min() / decay))


def down_pulse(t, decay=0.18):
    d = t - DOWN[DOWN <= t + 1e-6]
    if len(d) == 0: return 0.0
    return float(math.exp(-d.min() / decay))


# ------------------------------------------------------------------ footage
def rec_time(t, s):
    """map segment time -> recording time, with per-scene speed ramps (time remaps preserve sync at scene edges)"""
    rt = t + F.SEG_TO_REC
    r = s.get('ramp')
    if r:  # slow section [a,b] at speed k, followed by catch-up [b,c] so that rt(c) is in sync again
        a, b, c, k = r['a'], r['b'], r['c'], r['k']
        if a <= t < b: rt = a + F.SEG_TO_REC + (t - a) * k
        elif b <= t < c:
            lag = (b - a) * (1 - k); rt = t + F.SEG_TO_REC - lag * (1 - ease_in_out((t - b) / (c - b)))
    if rt < 0:  # no footage before 2.304 s: slow-motion b-roll from the close-up at the end of the recording
        br = SC.get('broll', dict(src=13.75, speed=0.5))
        rt = br['src'] + t * br['speed']
    return rt


def get_shot(t, s, w, h, zoom=1.0, face_pos=0.36, want_mask=False, xbias=0.0, blend=True):
    rt = rec_time(t, s)
    slow = abs((rec_time(t + 1 / FPS, s) - rt) * FPS - 1) > 0.05
    img = F.frame_at(rt, blend=blend and slow)
    cx, fy = F.subject(rt)
    ch = min(F.H, F.W * h / w) / zoom; cw = ch * w / h
    if cw > F.W: cw = F.W; ch = cw * h / w
    x0 = np.clip(cx + xbias * cw - cw / 2, 0, F.W - cw)
    y0 = np.clip(fy - face_pos * ch, 0, F.H - ch)
    M = np.float32([[w / cw, 0, -x0 * w / cw], [0, h / ch, -y0 * h / ch]])
    out = fx.to_f(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    if not want_mask: return out
    m = F.mask_at(rt)
    mm = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    mm = np.clip((mm - 0.35) / 0.4, 0, 1)
    return out, mm


# ------------------------------------------------------------------ layouts
def L_kinetic(t, s, fi):
    zoom = 1.0 + 0.06 * (t - s['start']) / (s['end'] - s['start']) + 0.025 * down_pulse(t)
    img, m = get_shot(t, s, CW, CH, zoom=zoom * s.get('zoom', 1.0), want_mask=True, face_pos=s.get('face_pos', 0.34))
    red = s.get('red', 0.0)
    bg = fx.grade(img, red=max(red, 0.5), dark=0.5, sat=0.75)
    if red: bg = bg * 0.6 + hexc('#5A0610') * 0.4 * (bg.mean(2, keepdims=True) * 2 + 0.25)
    subj = fx.grade(img, red=red * 0.25)
    cv = bg.copy()
    w = current_word(t, s)
    if w:
        pin, pout = word_phase(t, w, s)
        txt = w['w'].upper()
        size = min(fx.fit_size(txt, 'anton', 880, 600), 560)
        a, _ = fx.text_alpha(txt, 'anton', size)
        cy = kinetic_y(w, s, a)
        sc = in_scale(pin, a.shape[1], a.shape[0], 540, cy)
        dy = (1 - ease_out_cubic(pin)) * 40 - pout * 60
        dx, sy = fx.shake_offset(t, w['s'], 10 if w.get('strong') else 4)
        op = appear(pin) * (1 - pout)
        darken_behind(cv, a, 540, cy, sc)
        fx.shadow(cv, a, 540 + dx, cy + dy + sy, sc, radius=20, dy=14, opacity=op * 0.8)
        fx.blit(cv, a, WHITE, 540 + dx, cy + dy + sy, sc, 0, op,
                log=dict(t=t, text=txt, color=[1, 1, 1], opacity=op, layer='behind', scene=s['id'], anim=pin < 1 or pout > 0))
    # subject on top, with a thin white cut-out outline (torn-paper style: slightly jittered thickness)
    cv = cv * (1 - m[..., None]) + subj * m[..., None]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mb = (m > 0.5).astype(np.uint8)
    edge = cv2.dilate(mb, k) - mb
    rng = np.random.default_rng(fi // 3)
    noise = cv2.resize(rng.random((CH // 24, CW // 24)).astype(np.float32), (CW, CH))
    edge = edge.astype(np.float32) * (noise > 0.25)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.8)[..., None]
    cv = cv * (1 - edge * 0.95) + edge * 0.95
    return cv, m


def darken_behind(cv, a, cx, cy, sc):
    """contrast guard: if the background behind a white word is too bright, pull it down locally (soft mask)"""
    h, w = int(a.shape[0] * sc), int(a.shape[1] * sc)
    x0, y0 = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)); x1, y1 = int(min(cx + w / 2, CW)), int(min(cy + h / 2, CH))
    roi = cv[y0:y1, x0:x1]
    if roi.size == 0: return
    lum = (roi @ np.array([0.0722, 0.7152, 0.2126], np.float32))
    p = float(np.percentile(lum, 85))
    if p > 0.15:
        k = max(0.35, 0.15 / p)
        m = np.zeros((CH, CW), np.float32); m[y0:y1, x0:x1] = 1
        m = cv2.GaussianBlur(m, (0, 0), 40)[..., None]
        cv[:] = cv * (1 - m * (1 - k))


_ky = {}
def kinetic_y(w, s, a):
    """pick the vertical position: word sits behind the head/shoulders but the subject covers <=12% of its ink"""
    key = (w['i'], a.shape)
    if key in _ky: return _ky[key]
    _, m = get_shot(w['s'] + 0.05, s, CW, CH, zoom=s.get('zoom', 1.0), want_mask=True, face_pos=s.get('face_pos', 0.34))
    h, wd = a.shape
    best = None
    lo, hi = fx.SAFE['y0'] + h / 2 + 10, fx.SAFE['y1'] - h / 2 - 10
    for cy in np.linspace(lo, hi, 40):
        y0 = int(cy - h / 2); x0 = int(540 - wd / 2)
        sub = m[y0:y0 + h, x0:x0 + wd]
        cov = float((sub * a).sum() / (a.sum() + 1e-6))
        score = abs(cov - 0.08) + (0 if cov <= 0.12 else 5) + 0.0004 * abs(cy - 700)
        if best is None or score < best[0]: best = (score, cy, cov)
    _ky[key] = best[1]; w['_cover'] = best[2]
    return best[1]


def wrap_words(ws, fname, size, max_w, space=0.28):
    rows, cur, cw = [], [], 0
    for w in ws:
        a, _ = fx.text_alpha(w['w'].upper() if fname == 'anton' else w['w'], fname, size)
        ww = a.shape[1]; sp = size * space
        if cur and cw + sp + ww > max_w:
            rows.append((cur, cw)); cur, cw = [], 0
        cw = cw + (sp if cur else 0) + ww; cur.append((w, ww))
    if cur: rows.append((cur, cw))
    return rows


def L_phone(t, s, fi):
    bgimg = get_shot(t, s, CW // 4, CH // 4, zoom=1.3)
    bg = cv2.resize(cv2.GaussianBlur(fx.grade(bgimg, red=1.0, dark=0.55), (0, 0), 6), (CW, CH))
    yy = np.linspace(0, 1, CH, dtype=np.float32)[:, None, None]
    cv = bg * 0.45 + (hexc('#1A0204') * (1 - yy) + hexc('#050102') * yy) * 0.55
    # card (drawn flat, then perspective-warped)
    cwid, chei = 920, 1300
    card = np.zeros((chei, cwid, 3), np.float32)
    card[:] = hexc('#120507')
    card += cv2.resize(bg[300:1600, 80:1000], (cwid, chei)) * 0.12
    # lyric lines
    lines_in = sorted({w['line'] for w in scene_words(s)})
    w_now = current_word(t, s)
    cur_line = w_now['line'] if w_now else lines_in[0]
    # scroll: animate towards the current line over 0.3 s after the line's first word
    first = LINES[cur_line][0]['s']
    prev = cur_line - 1 if cur_line - 1 in lines_in else cur_line
    pscroll = ease_out_cubic((t - first) / 0.3) if prev != cur_line else 1.0
    BIG, SMALL, MAXW = 84, 58, 800
    blocks = {}
    for li in range(min(lines_in) - 1, max(lines_in) + 2):
        if li < 0 or li >= len(LINES): continue
        rows = wrap_words(LINES[li], 'round', BIG, MAXW)
        blocks[li] = rows
    def block_h(li, big): return len(blocks[li]) * (BIG * 1.18 if big else SMALL * 1.2)
    def layout_y(center_line):
        ys = {}; y = 0
        for li in sorted(blocks):
            ys[li] = y; y += block_h(li, li == center_line) + 46
        off = ys[center_line] + block_h(center_line, True) / 2
        return {k: v - off for k, v in ys.items()}
    ya, yb = layout_y(prev), layout_y(cur_line)
    base = 820
    card_logs = []
    for li, rows in blocks.items():
        y = base + ya[li] * (1 - pscroll) + yb[li] * pscroll
        big = li == cur_line
        size = BIG if big else SMALL
        if not big:
            rows = wrap_words(LINES[li], 'round', SMALL, MAXW)
        rh = size * (1.18 if big else 1.2)
        for ri, (row, rw) in enumerate(rows):
            x = 70
            cyr = y + ri * rh + rh / 2
            if cyr < 400 or cyr > chei - 60: x += 0  # clipped by card window below
            for w, ww in row:
                a, _ = fx.text_alpha(w['w'], 'round', size)
                if big:
                    sung = t >= w['s'] - 1e-6
                    pin = clamp01((t - w['s']) / IN_F)
                    col = ORANGE if sung else hexc('#7A3B33')
                    sc = 1.0 + 0.22 * (1 - ease_out_cubic(pin)) if sung else 1.0
                    op = 1.0 if sung else 0.55
                    if sung and pin < 1: sc = 1.25 - 0.25 * ease_out_cubic(pin)
                    if sung: fx.glow(card, a, hexc('#FF3A1A'), x + ww / 2, cyr, sc, radius=12, strength=0.38 * (1.5 - 0.5 * pin))
                    lg = dict(t=t, text=w['w'], color=list(map(float, col)), opacity=op, layer='card', scene=s['id']) if sung and w is w_now else None
                    bb = fx.blit(card, a, col, x + ww / 2, cyr, sc, 0, op)
                    if lg and bb: card_logs.append((lg, bb))
                else:
                    d = abs(li - cur_line)
                    fx.blit(card, a, ORANGE, x + ww / 2, cyr, 1, 0, 0.32 if d == 1 else 0.18)
                x += ww + size * 0.28
    # fade lyrics area top/bottom inside the card
    fade = np.ones((chei, 1, 1), np.float32)
    fade[:380] = 0.0
    fade[380:470, 0, 0] = np.linspace(0, 1, 90)
    fade[chei - 110:, 0, 0] = np.linspace(1, 0, 110)
    card_bg = np.zeros_like(card); card_bg[:] = hexc('#120507'); card_bg += cv2.resize(bg[300:1600, 80:1000], (cwid, chei)) * 0.12
    card = card_bg * (1 - fade) + card * fade
    # album art: live square crop of the footage
    art = get_shot(t, s, 210, 210, zoom=1.6, face_pos=0.42)
    fx.paste(card, fx.grade(art, red=0.3), 50, 50, fx.rounded_rect_alpha(210, 210, 26))
    a, o = fx.text_alpha('Ashke', 'round', 72); fx.blit(card, a, WHITE, 300 + a.shape[1] / 2, 105, 1)
    a, o = fx.text_alpha('Karan Aujla', 'round7', 46); fx.blit(card, a, hexc('#C9A9A6'), 300 + a.shape[1] / 2, 190, 1)
    # progress bar (song time 2:50..)
    st = 170 + t
    card[300:306, 50:870] = card[300:306, 50:870] * 0.4 + 0.6 * hexc('#4A2A2C')
    px = int(50 + 820 * (st / 218.1))
    card[299:307, 50:px] = ORANGE
    fx.blit(card, fx.disc_alpha(11), WHITE, px, 303)
    for txt, x in ((f'{int(st // 60)}:{int(st % 60):02d}', 50), ('-' + f'{int((218 - st) // 60)}:{int((218 - st) % 60):02d}', 870)):
        a, _ = fx.text_alpha(txt, 'round7', 30)
        fx.blit(card, a, hexc('#A88A88'), x + (a.shape[1] / 2 if x < 400 else -a.shape[1] / 2), 345)
    # border
    edge = fx.outline_alpha(fx.rounded_rect_alpha(cwid, chei, 56), 1)[3:-3, 3:-3]
    edge = cv2.resize(edge, (cwid, chei))
    card = card * (1 - edge[..., None] * 0.35) + edge[..., None] * 0.35
    # 3-D tilt: idle sway + kick on each line change
    kick = math.exp(-max(0, t - first) / 0.35) if prev != cur_line else 0
    ay = 5 * math.sin(t * 1.3) + 7 * kick
    ax = 3 * math.cos(t * 0.9) - 5 * kick
    yshift = -40 * (1 - pscroll) * 0
    H = tilt_h(cwid, chei, ax, ay, 540, 960 + yshift)
    cardA = fx.rounded_rect_alpha(cwid, chei, 56)
    # drop shadow
    sh = cv2.warpPerspective(cardA, H, (CW, CH)); sh = cv2.GaussianBlur(sh, (0, 0), 30)
    cv = cv * (1 - 0.6 * sh[..., None])
    wc = cv2.warpPerspective(card, H, (CW, CH), flags=cv2.INTER_LINEAR)
    wa = cv2.warpPerspective(cardA, H, (CW, CH))[..., None]
    cv = cv * (1 - wa) + wc * wa
    for lg, (x0, y0, x1, y1) in card_logs:
        pts = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x0, y1]], [[x1, y1]]]), H)[:, 0]
        fx.TEXT_LOG.append(dict(lg, bbox=(int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max()))))
    return cv, None


def tilt_h(w, h, ax, ay, cx, cy, f=1800):
    ax, ay = math.radians(ax), math.radians(ay)
    pts = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    p = pts @ (Ry @ Rx).T
    z = p[:, 2] + f
    dst = np.float32(np.stack([cx + p[:, 0] * f / z, cy + p[:, 1] * f / z], 1))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(src, dst)


_wall = {}
def build_wall(s):
    if s['id'] in _wall: return _wall[s['id']]
    size = s.get('wall_size', 150)
    WW, WH = 3600, 3600
    img = np.zeros((WH, WW), np.float32)
    pos = {}
    sw = scene_words(s)
    lines_in = sorted({w['line'] for w in sw})
    seq = [w for li in lines_in for w in LINES[li]]  # the scene's lyrics in order
    allw = WORDS
    rowh = int(size * 1.25)
    nrows = WH // rowh
    mid = nrows // 2
    # rows near the middle carry the scene's lyrics (2 rows apart so the camera move is diagonal), others filler
    import itertools
    k = 0
    filler = itertools.cycle(allw)
    rng = np.random.default_rng(3)
    lyric_rows = {}
    r = mid - 2
    for w in seq:
        lyric_rows.setdefault(r, []).append(w)
        a, _ = fx.text_alpha(w['w'].upper(), 'anton', size)
        if sum(fx.text_alpha(x['w'].upper(), 'anton', size)[0].shape[1] + size * 0.35 for x in lyric_rows[r]) > 1500:
            lyric_rows[r].pop(); r += 1; lyric_rows.setdefault(r, []).append(w)
    for row in range(nrows):
        y = row * rowh + rowh // 2
        x = -rng.integers(0, 400) if row not in lyric_rows else WW // 2 - 700 + (row - mid) * 260
        if row in lyric_rows:
            # filler before
            xx = x
            fl = []
            while xx > 0:
                w = next(filler); a, _ = fx.text_alpha(w['w'].upper(), 'anton', size); xx -= a.shape[1] + size * 0.35; fl.append((w, a, xx))
            for w, a, xx2 in fl: _stamp(img, a, xx2, y)
            for w in lyric_rows[row]:
                a, _ = fx.text_alpha(w['w'].upper(), 'anton', size)
                if w['i'] not in pos: pos[w['i']] = (x + a.shape[1] / 2, y)
                _stamp(img, a, x, y); x += a.shape[1] + size * 0.35
        while x < WW:
            w = next(filler); a, _ = fx.text_alpha(w['w'].upper(), 'anton', size)
            _stamp(img, a, x, y); x += a.shape[1] + size * 0.35
    _wall[s['id']] = (img, pos, size)
    return _wall[s['id']]


def _stamp(img, a, x, y):
    x, y = int(x), int(y - a.shape[0] / 2)
    h, w = a.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 > x0 and y1 > y0: img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], a[y0 - y:y1 - y, x0 - x:x1 - x])


def L_wall(t, s, fi):
    img, pos, size = build_wall(s)
    sw = scene_words(s)
    w = current_word(t, s) or sw[0]
    # camera target: eased move from previous word to current word
    prv = WORDS[w['i'] - 1] if w['i'] > 0 and WORDS[w['i'] - 1]['i'] in pos else w
    pm = ease_out_cubic((t - w['s']) / 0.14) if w['s'] <= t else 0
    tx = pos[prv['i']][0] * (1 - pm) + pos[w['i']][0] * pm
    ty = pos[prv['i']][1] * (1 - pm) + pos[w['i']][1] * pm
    u = (t - s['start']) / (s['end'] - s['start'])
    ang = s.get('angle', 30) + 10 * u  # slow spin
    zoom = 0.92 + 0.22 * u + 0.04 * beat_pulse(t)
    M = cv2.getRotationMatrix2D((tx, ty), ang, zoom)
    M[0, 2] += 540 - tx; M[1, 2] += 900 - ty
    wa = cv2.warpAffine(img, M, (CW, CH), flags=cv2.INTER_LINEAR)
    base = np.empty((CH, CW, 3), np.float32); base[:] = DRED
    foot = fx.grade(get_shot(t, s, CW // 2, CH // 2, zoom=1.1), red=1.0, dark=0.3)
    base = base + cv2.resize(foot, (CW, CH)) * 0.16
    cv = base * (1 - wa[..., None] * 0.9) + hexc('#A8121C') * wa[..., None] * 0.9
    # current word in white, flashing on its start
    if w['s'] <= t:
        pin, pout = word_phase(t, w, s)
        a, _ = fx.text_alpha(w['w'].upper(), 'anton', size)
        p = np.array([pos[w['i']][0], pos[w['i']][1], 1]) @ M.T
        c_, s_ = abs(math.cos(math.radians(ang))), abs(math.sin(math.radians(ang)))
        fitb = min(zoom, 880 / (a.shape[1] * c_ + a.shape[0] * s_), 1300 / (a.shape[1] * s_ + a.shape[0] * c_))
        bw_, bh_ = (a.shape[1] * c_ + a.shape[0] * s_) * fitb / 2, (a.shape[1] * s_ + a.shape[0] * c_) * fitb / 2
        p = (float(np.clip(p[0], 75 + bw_, CW - 75 - bw_)), float(np.clip(p[1], 155 + bh_, CH - 255 - bh_)))
        sc = in_scale(pin, a.shape[1] + 24, a.shape[0] + 24, p[0], p[1], ang, base=fitb)
        flash = 1 - ease_out_cubic(pin)
        fx.blit(cv, np.pad(a, 12), DRED * 0.6, p[0], p[1], sc * 1.04, ang, 0.9 * (1 - pout))
        fx.glow(cv, a, hexc('#FF8A8A'), p[0], p[1], sc, ang, radius=22, strength=0.6 + 0.6 * flash, opacity=1 - pout)
        fx.blit(cv, a, WHITE, p[0], p[1], sc, ang, appear(pin) * (1 - pout),
                log=dict(t=t, text=w['w'].upper(), color=[1, 1, 1], opacity=appear(pin) * (1 - pout), layer='wall', scene=s['id'], anim=pin < 1 or pout > 0))
    return cv, None


def L_circle(t, s, fi):
    cv = np.empty((CH, CW, 3), np.float32)
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    cx, cy = 540, s.get('cy', 860)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cv[:] = (np.clip(1 - rr / 1300, 0, 1) ** 2 * 0.07)[..., None] * hexc('#FF4040') + 0.012
    R = int(390 * (1 + 0.045 * beat_pulse(t) + 0.03 * down_pulse(t)) * (0.85 + 0.15 * ease_out_back((t - s['start']) / 0.35)))
    img = get_shot(t, s, 2 * R, 2 * R, zoom=s.get('zoom', 1.25), face_pos=0.40)
    img = fx.grade(img, red=s.get('red', 0.15))
    d = fx.disc_alpha(R)[1:2 * R + 1, 1:2 * R + 1]
    d = cv2.resize(fx.disc_alpha(R), (2 * R, 2 * R))
    fx.paste(cv, img, cx - R, cy - R, d)
    ring = fx.ring_alpha(R + 18, 3)
    fx.blit(cv, ring, WHITE, cx, cy, 1, 0, 0.85)
    # ticks
    rot = (t - s['start']) * 12
    for k in range(72):
        ang = math.radians(k * 5 + rot)
        long = k % 6 == 0
        r0, r1 = R + 34, R + (60 if long else 46)
        p0 = (int(cx + r0 * math.cos(ang)), int(cy + r0 * math.sin(ang)))
        p1 = (int(cx + r1 * math.cos(ang)), int(cy + r1 * math.sin(ang)))
        cv2.line(cv, p0, p1, (0.9, 0.9, 0.9) if long else (0.5, 0.5, 0.5), 3 if long else 2, cv2.LINE_AA)
    for k in range(3):  # HUD dots orbiting
        ang = math.radians(-rot * 2.5 + k * 120)
        fx.blit(cv, fx.disc_alpha(7), hexc('#FF3B3B') if k == 0 else WHITE, cx + (R + 84) * math.cos(ang), cy + (R + 84) * math.sin(ang))
    # HUD labels
    fx.blit(cv, fx.disc_alpha(9), hexc('#FF2D2D'), 110, 196, 1, 0, 0.5 + 0.5 * (int(t * 2) % 2))
    a, _ = fx.text_alpha('REC', 'bebas', 46); fx.blit(cv, a, WHITE, 132 + a.shape[1] / 2, 196)
    st = 170 + t
    tc = f'{int(st // 60):02d}:{int(st % 60):02d}:{int((st % 1) * 30):02d}'
    a, _ = fx.text_alpha(tc, 'bebas', 46, tracking=3); fx.blit(cv, a, WHITE, CW - 100 - a.shape[1] / 2, 196, 1, 0, 0.85)
    a, _ = fx.text_alpha('ASHKE  /  KARAN AUJLA', 'bebas', 40, tracking=4); fx.blit(cv, a, hexc('#BBBBBB'), 540, cy + R + 130, 1, 0, 0.9)
    # word pop
    w = current_word(t, s)
    if w:
        pin, pout = word_phase(t, w, s)
        txt = w['w'].upper()
        size = min(fx.fit_size(txt, 'anton', 2 * R - 140, 170), 170)
        a, _ = fx.text_alpha(txt, 'anton', size)
        wy = cy + R * 0.52
        sc = in_scale(pin, a.shape[1] + 70, a.shape[0] + 50, 540, wy)
        op = appear(pin) * (1 - pout)
        pill = fx.rounded_rect_alpha(a.shape[1] + 70, a.shape[0] + 50, 22)
        fx.blit(cv, pill, BLACK, 540, wy, sc, 0, 0.62 * op)
        fx.blit(cv, a, WHITE, 540, wy, sc, 0, op, log=dict(t=t, text=txt, color=[1, 1, 1], opacity=op, layer='circle', scene=s['id'], anim=pin < 1 or pout > 0))
    return cv, None


def L_strip(t, s, fi):
    cv = np.empty((CH, CW, 3), np.float32); cv[:] = RED
    # subtle band texture
    cv *= (0.92 + 0.08 * np.linspace(0, 1, CH, dtype=np.float32)[:, None, None])
    sy0, sy1 = s.get('strip', [560, 1290])
    sh = sy1 - sy0
    zoom = s.get('zoom', 1.0) * (1 + 0.03 * (t - s['start']))
    img = get_shot(t, s, CW, sh, zoom=zoom, face_pos=0.30, xbias=s.get('xbias', 0.0))
    cv[sy0:sy1] = fx.grade(img, red=0.25)
    cv[sy0 - 6:sy0] = YEL; cv[sy1:sy1 + 6] = YEL
    # sunbursts at strip corners
    for k, (x, y, col, r) in enumerate([(90, sy0 - 10, YEL, 100), (CW - 90, sy0 - 10, WHITE, 70), (90, sy1 + 10, WHITE, 70), (CW - 90, sy1 + 10, YEL, 100)]):
        st = fx.star_alpha(r, r * 0.42, 12 if k % 3 == 0 else 8)
        fx.blit(cv, st, col, x, y, 1 + 0.08 * beat_pulse(t), (t - s['start']) * (40 if k % 2 else -40), 1.0)
    # words: the current line builds word by word in two rows above and two rows below the strip
    w_now = current_word(t, s)
    if w_now:
        sw_ids = {x['i'] for x in scene_words(s)}
        line = [w for w in LINES[w_now['line']] if w['i'] in sw_ids]
        BIG = s.get('size', 128)
        rows = wrap_words(line, 'anton', BIG, 900)
        slots = [sy0 - 270, sy0 - 105, sy1 + 135, sy1 + 300] if len(rows) > 2 else [sy1 + 135, sy1 + 300]
        if len(rows) > len(slots):  # shrink until it fits
            while len(rows) > 4 and BIG > 60: BIG -= 8; rows = wrap_words(line, 'anton', BIG, 900)
            slots = [sy0 - 270, sy0 - 105, sy1 + 135, sy1 + 300]
        for ri, (row, rw) in enumerate(rows):
            x = 90; y = slots[ri]
            for w, ww in row:
                if t + 1e-6 < w['s']: x += ww + BIG * 0.28; continue
                pin, _ = word_phase(t, w, s)
                pout = 0.0
                txt = w['w'].upper()
                secondary = len(txt) <= 3 and not w.get('strong')
                e = ease_out_cubic(pin)
                xo = -(1 - e) * 36
                col = YEL if w.get('strong') else WHITE
                if secondary:
                    a, _ = fx.text_alpha(w['w'].lower(), 'serif', int(BIG * 0.62))
                    cxw = x + ww / 2 + xo
                    fx.blit(cv, wipe(a, e), WHITE, cxw, y + BIG * 0.12, 1, 0, appear(pin) * (1 - pout),
                            log=dict(t=t, text=txt, color=[1, 1, 1], opacity=appear(pin) * (1 - pout), layer='strip', scene=s['id'], anim=pin < 1) if w is w_now else None)
                    lw = int(a.shape[1] * e)
                    for dy in (-a.shape[0] / 2 - 14, -a.shape[0] / 2 - 6, a.shape[0] / 2 + 6, a.shape[0] / 2 + 14):
                        yl = int(y + BIG * 0.12 + dy)
                        cv[yl:yl + 3, int(cxw - a.shape[1] / 2):int(cxw - a.shape[1] / 2) + lw] = YEL
                else:
                    a, _ = fx.text_alpha(txt, 'anton', BIG)
                    fx.shadow(cv, a, x + ww / 2 + xo, y, 1, radius=8, dy=6, opacity=0.6 * e)
                    fx.blit(cv, wipe(a, e), col, x + ww / 2 + xo, y, 1, 0, appear(pin) * (1 - pout),
                            log=dict(t=t, text=txt, color=list(map(float, col)), opacity=appear(pin) * (1 - pout), layer='strip', scene=s['id'], anim=pin < 1) if w is w_now else None)
                x += ww + BIG * 0.28
    return cv, None


def wipe(a, e):
    """left-to-right mask wipe (soft edge); at e=0 the first ~45% is already revealed so the word reads on its first frame"""
    e = 0.45 + 0.55 * e
    if e >= 1: return a
    w = a.shape[1]; x = np.arange(w, dtype=np.float32)
    ramp = np.clip((e * (w + 40) - x) / 40, 0, 1)
    return a * ramp[None, :]


def L_panel(t, s, fi):
    cv = np.empty((CH, CW, 3), np.float32); cv[:] = hexc('#F4F1EA')
    img = get_shot(t, s, 540, CH, zoom=s.get('zoom', 1.0), face_pos=0.30)
    cv[:, :540] = fx.grade(img, red=s.get('red', 0.1))
    cv[:, 536:544] = 0.04
    sw = [w for w in scene_words(s) if w['s'] <= t + 1e-6]
    # pack into boxes of <= 2 words, line breaks start a new box
    boxes = []
    for w in sw:
        if boxes and len(boxes[-1]) < 2 and boxes[-1][-1]['line'] == w['line']:
            test = ' '.join(x['w'].upper() for x in boxes[-1] + [w])
            if fx.text_alpha(test, 'anton', 110)[0].shape[1] <= 330: boxes[-1].append(w); continue
        boxes.append([w])
    ROWH = 165; top = 250
    visible = 8
    first = max(0, len(boxes) - visible)
    # smooth scroll when a new box pushes the stack
    scroll = 0.0
    if len(boxes) > visible:
        p = ease_out_cubic((t - boxes[-1][0]['s']) / 0.15)
        scroll = (1 - p) * ROWH
    for bi in range(first, len(boxes)):
        b = boxes[bi]
        txt = ' '.join(x['w'].upper() for x in b)
        size = min(fx.fit_size(txt, 'anton', 330, 110), 110)
        a, _ = fx.text_alpha(txt, 'anton', size)
        y = top + (bi - first) * ROWH + scroll
        last = b[-1]
        pin = clamp01((t - last['s']) / IN_F)
        sc = in_scale(pin, a.shape[1] + 60, a.shape[0] + 54, 0, 0, s_max=1.5) if False else 1.0
        sc = 1.0 + 0.4 * (1 - ease_out_cubic(pin))
        bw, bh = a.shape[1] + 50, a.shape[0] + 44
        cx = 580 + bw / 2
        box = np.zeros((bh + 10, bw + 10), np.float32)
        box[5:-5, 5:-5] = 1; inner = np.zeros_like(box); inner[11:-11, 11:-11] = 1
        border = box - inner
        cur = b[-1] is current_word(t, s)
        dx, dy = fx.shake_offset(t, last['s'], 8)
        if cur:  # current box inverted: black box, white word
            fx.blit(cv, box, BLACK, cx + dx, y + dy, sc, 0, 1)
            fx.blit(cv, a, WHITE, cx + dx, y + dy, sc, 0, 1,
                    log=dict(t=t, text=txt, color=[1, 1, 1], opacity=1, layer='panel', scene=s['id'], anim=pin < 1))
        else:
            fx.blit(cv, border, BLACK, cx, y, 1, 0, 1)
            fx.blit(cv, a, BLACK, cx, y, 1, 0, 1)
    return cv, None


def L_end(t, s, fi):
    cv = np.empty((CH, CW, 3), np.float32)
    yy = np.linspace(0, 1, CH, dtype=np.float32)[:, None, None]
    cv[:] = hexc('#120C14') * (1 - yy) + hexc('#050407') * yy
    u = t - s['start']
    # faint huge outlined handle text
    hs = fx.fit_size(HANDLE.upper(), 'anton', 940, 420)
    a, _ = fx.text_alpha(HANDLE.upper(), 'anton', hs)
    ol = fx.outline_alpha(a, 2)
    for k, yb in enumerate((330, 760, 1190, 1620)):
        fx.blit(cv, ol, WHITE, 540, yb, 1 + 0.02 * u, 0, 0.07)
    # camera glyph, drawn from vector shapes, gradient fill rotating over time
    icon_y = 900
    S = 330
    e = ease_out_back(u / 0.45)
    sz = int(S)
    outer = fx.rounded_rect_alpha(sz, sz, int(sz * 0.28))
    inner = np.pad(fx.rounded_rect_alpha(sz - 56, sz - 56, int((sz - 56) * 0.22)), 28)
    glyph = np.clip(outer - inner, 0, 1)
    ring = fx.ring_alpha(sz * 0.2, 28)
    rh, rw = ring.shape; oy, ox = (sz - rh) // 2, (sz - rw) // 2
    glyph[oy:oy + rh, ox:ox + rw] = np.maximum(glyph[oy:oy + rh, ox:ox + rw], ring)
    dot = fx.disc_alpha(sz * 0.055); dh = dot.shape[0]
    dx0, dy0 = int(sz * 0.75 - dh / 2), int(sz * 0.25 - dh / 2)
    glyph[dy0:dy0 + dh, dx0:dx0 + dh] = np.maximum(glyph[dy0:dy0 + dh, dx0:dx0 + dh], dot)
    gy, gx = np.mgrid[0:sz, 0:sz].astype(np.float32) / sz
    ang = u * 1.6
    g = np.clip((gx - 0.5) * math.cos(ang) + (gy - 0.5) * math.sin(ang) + 0.5, 0, 1)
    stops = [hexc(c) for c in ('#FEDA75', '#FA7E1E', '#D62976', '#962FBF', '#4F5BD5')]
    gi = g * (len(stops) - 1); i0 = np.clip(gi.astype(int), 0, len(stops) - 2); fr = (gi - i0)[..., None]
    grad = np.stack(stops)[i0] * (1 - fr) + np.stack(stops)[i0 + 1] * fr
    fx.glow(cv, glyph, hexc('#D62976'), 540, icon_y, e, 0, radius=30, strength=0.5, opacity=clamp01(u / 0.3))
    fx.blit(cv, glyph, None, 540, icon_y, e, 0, clamp01(u / 0.2), src=grad)
    # handle
    hp = ease_out_cubic((u - 0.18) / 0.4)
    if hp > 0:
        hsz = min(fx.fit_size(HANDLE, 'bebas', 900, 130), 130)
        fx.draw_text(cv, HANDLE, 'bebas', hsz, WHITE, 540, icon_y + 290 + (1 - hp) * 40, opacity=hp, tracking=4)
    # sung words continuing into the end card
    w = current_word(t, s)
    if w:
        pin, pout = word_phase(t, w, s)
        txt = w['w'].upper()
        size = min(fx.fit_size(txt, 'anton', 900, 240), 240)
        a, _ = fx.text_alpha(txt, 'anton', size)
        op = appear(pin) * (1 - pout)
        sc = in_scale(pin, a.shape[1], a.shape[0], 540, 470)
        fx.blit(cv, a, WHITE, 540, 470, sc, 0, op, log=dict(t=t, text=txt, color=[1, 1, 1], opacity=op, layer='end', scene=s['id'], anim=pin < 1 or pout > 0))
    return cv, None


LAYOUTS = dict(kinetic=L_kinetic, phone=L_phone, wall=L_wall, circle=L_circle, strip=L_strip, panel=L_panel, end=L_end)


# ------------------------------------------------------------------ frame
def render_frame(fi):
    t = fi / FPS
    s = scene_at(t)
    cv, _ = LAYOUTS[s['layout']](t, s, fi)
    is_foot = s['layout'] in ('kinetic', 'strip', 'panel', 'circle')
    # hit emphasis on strong words + downbeats: shake, 2-frame white flash, RGB split
    hits = [w['s'] for w in WORDS if w.get('strong')]
    hits += [d for d in DOWN if any(abs(d - x['s']) < 0.09 for x in WORDS)]
    dt = min([t - h for h in hits if t - h >= -1e-6] + [9])
    if dt < 0.2 and s['layout'] not in ('end',):
        dx, dy = fx.shake_offset(t, t - dt, 12, 0.2, 11)
        cv = fx.shift(cv, dx, dy)
        cv = fx.rgb_split(cv, 10 * (1 - dt / 0.2) if dt < 0.14 else 0)
        if dt < 2 / FPS: cv = cv + (1 - cv) * (0.32 if dt < 1 / FPS else 0.14)
    # transitions
    for tr in SC.get('transitions', []):
        d = t - tr['t']
        if tr['type'] == 'streak' and abs(d) < 0.12:
            k = 1 - abs(d) / 0.12
            cv = fx.motion_blur_h(cv, 140 * k)
            cv = fx.shift(cv, -np.sign(d if d else 1) * 160 * k ** 2, 0)
            rng = np.random.default_rng(fi)
            for _ in range(int(14 * k)):
                y = int(rng.integers(0, CH)); hgt = int(rng.integers(2, 10))
                cv[y:y + hgt] = cv[y:y + hgt] * (1 - 0.7 * k) + 0.7 * k
            if abs(d) < 1 / FPS: cv = cv + (1 - cv) * 0.45
        elif tr['type'] == 'zoom' and abs(d) < 0.12:
            k = 1 - abs(d) / 0.12
            z = 1 + 0.35 * k ** 2
            M = cv2.getRotationMatrix2D((540, 960), 0, z)
            cv = cv2.warpAffine(cv, M, (CW, CH), borderMode=cv2.BORDER_REFLECT)
            # radial-ish blur via averaging zoom copies
            acc = cv.copy()
            for j in range(1, 4):
                M = cv2.getRotationMatrix2D((540, 960), 0, 1 + 0.03 * j * k)
                acc += cv2.warpAffine(cv, M, (CW, CH), borderMode=cv2.BORDER_REFLECT)
            cv = acc / 4
            if abs(d) < 1 / FPS: cv = cv + (1 - cv) * 0.35
        elif tr['type'] == 'flash' and 0 <= d < 2 / FPS:
            cv = cv + (1 - cv) * (0.5 if d < 1 / FPS else 0.2)
    if s['layout'] == 'end':
        cv = cv * (1 - ease_in_out((t - (DUR - 0.4)) / 0.4))
    cv = fx.vignette(cv, 0.35 if is_foot else 0.2)
    cv = fx.grain(cv, fi, 0.03 if is_foot else 0.018)
    return (np.clip(cv, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=f'0:{NF}')
    ap.add_argument('--out', default=D + 'out/chunk.mkv')
    ap.add_argument('--png', default=None, help='dump selected frames to this dir instead of encoding')
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    f0, f1 = map(int, a.frames.split(':'))
    if a.png:
        os.makedirs(a.png, exist_ok=True)
        for fi in range(f0, f1):
            cv2.imwrite(f'{a.png}/{fi:04d}.png', render_frame(fi))
    else:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{CW}x{CH}', '-r', str(FPS),
                              '-i', '-', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'yuv444p', a.out], stdin=subprocess.PIPE)
        for fi in range(f0, f1):
            p.stdin.write(render_frame(fi).tobytes())
            if fi % 30 == 0: print('frame', fi, flush=True)
        p.stdin.close(); p.wait()
    if a.log:
        json.dump(fx.TEXT_LOG, open(a.log, 'w'))


if __name__ == '__main__':
    main()
