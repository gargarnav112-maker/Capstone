"""Master renderer: schedules scenes/transitions, applies global audio-reactive
post, pipes frames to ffmpeg.
usage: python3 render.py OUT.mp4 [--frames a:b] [--workers 4] [--crf 12]"""
import sys, os, math, argparse, subprocess, time
from multiprocessing import Pool
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import *
from engine import scene_l1, scene_l2, scene_l3, scene_l4, scene_l5, scene_l6, transitions

B = [0.0,
     LINE_WORDS[1][0]['start'] - 0.08,     # L1 -> L2 (cut on 'DASS')
     LINE_WORDS[2][0]['start'] - 0.06,     # L2 -> L3
     LINE_WORDS[3][0]['start'] - 0.06,     # L3 -> L4
     transitions.D1,                       # L4 -> L5 (dupatta)
     transitions.IRIS1,                    # L5 -> L6 (iris)
     DUR]
SCENES = [scene_l1, scene_l2, scene_l3, scene_l4, scene_l5, scene_l6]


def scene_frame(t, fx):
    if transitions.D0 <= t < transitions.D1:
        a = scene_l4.render(t, fx); b = scene_l5.render(t, fx)
        return transitions.dupatta(a, b, t), 3
    if transitions.IRIS0 <= t < transitions.IRIS1:
        a = scene_l5.render(t, fx); b = scene_l6.render(t, fx)
        return transitions.iris(a, b, t, scene_l6.circle_center(t)), 4
    for i in range(6):
        if B[i] <= t < B[i + 1]: return SCENES[i].render(t, fx), i
    return SCENES[5].render(t, fx), 5


def ribbon(c, t, op):
    N = 160; xs = np.linspace(0, W, N)
    ts = t - 1.6 + 1.6 * np.arange(N) / (N - 1)
    amp = np.array([env('loud', max(0, x)) for x in ts]) * 34 + np.array([env('mid', max(0, x)) for x in ts]) * 10
    wig = np.sin(np.arange(N) * 0.9 + t * 20) * 0.5 + 0.5
    a = amp * (0.55 + 0.45 * wig) * np.clip(np.arange(N) / 18, 0, 1) * np.clip((N - np.arange(N)) / 10, 0, 1)
    yb = H - 44
    top = np.stack([xs, yb - a], 1); bot = np.stack([xs, yb + a * 0.6], 1)
    poly = np.vstack([top, bot[::-1]])
    m = np.zeros((140, W), np.uint8)
    cv2.fillPoly(m, [np.round((poly - [0, yb - 70]) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    comp(c, m.astype(np.float32) / 255, (0, int(yb - 70), W, int(yb + 70)), GOLD, 0.28 * op)
    draw_polyline(c, top, GOLD, 2, 0.9 * op, 'normal', glow_r=4)


def hud(c, t, li, op):
    song = SONG_T0 + t
    a, base = text_alpha(f"ASHKE // {int(song // 60):02d}:{song % 60:06.3f}", 'mono', 22, 0)
    draw_sprite(c, a, M_affine(44, 66, 1, 1, 0, 0, base), CREAM, 0.55 * op)
    a, base = text_alpha(f"L{li + 1}/6  {BEATS_J['tempo']:.1f} BPM", 'mono', 22, 0)
    draw_sprite(c, a, M_affine(W - 44, 66, 1, 1, 0, a.shape[1], base), CREAM, 0.55 * op)


def glitch(c, amt, t):
    rng = np.random.default_rng(int(t * FPS) * 7 + 3)
    dx = int(6 + 22 * amt)
    out = c.copy()
    out[..., 0] = np.roll(c[..., 0], dx, 1); out[..., 2] = np.roll(c[..., 2], -dx, 1)
    for _ in range(int(3 + 8 * amt)):
        y0 = rng.integers(0, H - 40); h = rng.integers(8, 90); s = int(rng.normal(0, 60 * amt))
        out[y0:y0 + h] = np.roll(out[y0:y0 + h], s, 1)
    return out


def render_frame(n):
    t = n / FPS
    fx = {}
    c, li = scene_frame(t, fx)
    c = np.ascontiguousarray(c, np.float32)
    end_mode = t >= scene_l6.T_SHATTER
    # camera shake
    if fx.get('shake', 0) > 0.3:
        rng = np.random.default_rng(n)
        sx, sy = rng.normal(0, fx['shake'], 2)
        c = warp_full(c, np.float32([[1, 0, sx], [0, 1, sy]]), border=cv2.BORDER_REFLECT)
    # bass zoom punch
    if not end_mode:
        z = 1 + 0.02 * pulse(BEATS, t, 0.11) + 0.012 * env('bass', t)
        M = M_affine(W / 2, H / 2, z, z, 0, W / 2, H / 2)
        c = warp_full(c, M, border=cv2.BORDER_REFLECT)
    # transition glitch L3->L4 + scene glitch
    g = fx.get('glitch', 0)
    if B[3] - 0.03 <= t < B[3] + 0.1: g = max(g, 0.8)
    if g > 0.05: c = glitch(c, g, t)
    # flashes: scene flashes + snare (mid onsets) + cut flashes
    fl = fx.get('flash', 0)
    if not end_mode: fl += 0.07 * pulse(MID_ON, t, 0.05)
    for b in B[1:3]:
        if 0 <= t - b < 0.12: fl = max(fl, 0.3 * (1 - (t - b) / 0.12))
    if fl > 0: c = 1 - (1 - np.clip(c, 0, 1)) * (1 - min(fl, 0.9) * np.array([1.0, 0.92, 0.8], np.float32))
    # overlays
    ov = 1 - prog(t, scene_l6.T_SHATTER - 0.05, scene_l6.T_SHATTER + 0.1)
    if ov > 0:
        ribbon(c, t, ov); hud(c, t, li, ov)
    c *= vignette_map(0.38)
    add_grain(c, t)
    # fade in / out (audio has 0.15 s in, 0.4 s out)
    c *= ease_io(prog(t, 0, 0.12)) * (1 - prog(t, DUR - 0.4, DUR))
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--frames', default=None)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--crf', type=int, default=12)
    ap.add_argument('--png', default=None)
    a = ap.parse_args()
    N = int(round(DUR * FPS))
    f0, f1 = (map(int, a.frames.split(':'))) if a.frames else (0, N)
    frames = range(f0, f1)
    if a.png:
        os.makedirs(a.png, exist_ok=True)
        with Pool(a.workers) as p:
            for n, im in zip(frames, p.imap(render_frame, frames, chunksize=2)):
                cv2.imwrite(f'{a.png}/{n:04d}.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        return
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-i', os.path.join(WORK, 'segment.wav'), '-map', '0:v', '-map', '1:a',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(a.crf), '-pix_fmt', 'yuv420p',
           '-c:a', 'pcm_s16le' if a.out.endswith('.mov') else 'aac', '-b:a', '320k', '-shortest', a.out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(a.workers) as p:
        for k, im in enumerate(p.imap(render_frame, frames, chunksize=2)):
            pr.stdin.write(im.tobytes())
            if k % 60 == 0: print(f'frame {f0 + k}/{f1}  {time.time() - t0:.0f}s', flush=True)
    pr.stdin.close(); pr.wait()
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
