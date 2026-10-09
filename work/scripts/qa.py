"""QA: ffprobe specs, per-word sync from the render text log, safe zone / clipping, contrast, black frames, frame dumps."""
import json, sys, os, subprocess, numpy as np, cv2
sys.path.insert(0, 'scripts')
os.environ.setdefault('LYRICS', 'lyrics.json'); os.environ.setdefault('SCENES', 'scenes.json')
LY = json.load(open(os.environ['LYRICS'])); SC = json.load(open(os.environ['SCENES']))
vid = sys.argv[1]
pr = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', vid]))
v = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]; a = [s for s in pr['streams'] if s['codec_type'] == 'audio']
print(f"container {float(pr['format']['duration']):.3f}s  video {v['width']}x{v['height']} {v['r_frame_rate']} {v['codec_name']} {int(v.get('bit_rate', 0)) / 1e6:.1f}Mbps frames={v.get('nb_frames')} | audio {a[0]['codec_name'] if a else 'MISSING'} {int(a[0]['bit_rate']) // 1000 if a else 0}k dur={float(a[0]['duration']):.3f}s | size {int(pr['format']['size']) / 1e6:.1f}MB")
cap = cv2.VideoCapture(vid); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
print('decoded frames', len(frames))
words = sorted([w for l in LY['lines'] for w in l['words']], key=lambda w: w['s'])
log = json.load(open('qa/text_log.json'))
FPS = 30
bylog = {}
for e in log: bylog.setdefault(round(e['t'] * FPS), []).append(e)
problems = []
# 1) sync: each word must be visible (opacity>=0.3) within 2 frames of its start
norm = lambda s: ''.join(c for c in s.upper() if c.isalnum())
for i, w in enumerate(words):
    f0 = int(np.ceil(w['s'] * FPS - 1e-6))
    hit = None
    for f in range(f0, f0 + 8):
        if any(norm(e['text']).find(norm(w['w'])) >= 0 and e['opacity'] >= 0.3 for e in bylog.get(f, [])): hit = f; break
    if hit is None or hit - w['s'] * FPS > 2: problems.append(f"SYNC {w['w']} s={w['s']:.3f} first visible frame={hit}")
# 2) safe zone / clipping
for e in log:
    x0, y0, x1, y1 = e['bbox']
    if e['opacity'] < 0.05: continue
    if x0 < 0 or y0 < 0 or x1 > 1080 or y1 > 1920: problems.append(f"CLIP t={e['t']:.3f} {e['text']} {e['bbox']}")
    elif not e.get('anim') and (x0 < 70 - 4 or x1 > 1010 + 4 or y0 < 150 - 4 or y1 > 1670 + 4):
        problems.append(f"SAFE t={e['t']:.3f} {e['text']} {e['bbox']} [{e.get('layer')}]")
# 3) contrast on settled words: text colour vs. darker-than-text pixels in the bbox (behind-subject: background only)
def L(c):
    c = np.asarray(c, float); c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.0722 * c[..., 0] + 0.7152 * c[..., 1] + 0.2126 * c[..., 2]
HITS = [w['s'] for w in words if w.get('strong')] + [d for d in json.load(open('beats.json'))['downbeats'] if any(abs(d - x['s']) < 0.09 for x in words)] + [tr['t'] + d for tr in SC.get('transitions', []) for d in (-0.12, -0.06, 0)]
worst = {}
for e in log:
    fi = round(e['t'] * FPS)
    if e['opacity'] < 0.99 or e.get('anim') or fi >= len(frames) or e['t'] > 15.6: continue
    if any(0 <= e['t'] - h < 0.15 for h in HITS): continue  # intentional 2-frame flash / RGB-split on hits
    x0, y0, x1, y1 = [max(0, v) for v in e['bbox']]
    roi = frames[fi][y0:y1, x0:x1].astype(float) / 255
    lt = float(L(np.array(e['color'])))
    lum = L(roi).ravel()
    light_text = np.median(lum) < lt
    lum2 = L(roi)
    ink = (np.abs(lum2 - lt) < 0.12 * max(lt, 0.2)).astype(np.uint8)
    halo = cv2.dilate(ink, np.ones((7, 7), np.uint8)) > 0  # exclude ink + antialias/glow ring
    bg = lum2[~halo]
    bg = bg[bg < 0.5 * lt] if light_text else bg[bg > lt + 0.3]
    if len(bg) < 50: continue
    lb = float(np.percentile(bg, 75)) if light_text else float(np.percentile(bg, 25))
    cr = (max(lt, lb) + 0.05) / (min(lt, lb) + 0.05)
    k = e['text']
    if k not in worst or cr < worst[k][0]: worst[k] = (cr, e['t'], e.get('layer'))
low = {k: v for k, v in worst.items() if v[0] < 4.5}
for k, v in low.items(): problems.append(f"CONTRAST {k} {v[0]:.2f} at t={v[1]:.2f} [{v[2]}]")
print('min contrast', min(v[0] for v in worst.values()) if worst else None)
# 4) black / flicker
lum = np.array([f.mean() for f in frames])
for i, m in enumerate(lum):
    if m < 4 and i < len(frames) - 14: problems.append(f'BLACK frame {i}')
# 5) per-frame expected word vs shown word
miss = 0
for fi in range(len(frames)):
    t = fi / FPS
    cur = [w for w in words if w['s'] <= t + 1e-6]
    if not cur: continue
    w = cur[-1]
    nxt = words[words.index(w) + 1]['s'] if words.index(w) + 1 < len(words) else 16
    if t > nxt - 2.5 / FPS: continue  # out-animation window
    shown = [e['text'] for e in bylog.get(fi, []) if e['opacity'] > 0.2]
    if not any(norm(w['w']) in norm(x) for x in shown): miss += 1; problems.append(f'WRONG f{fi} t={t:.3f} expect {w["w"]} shown {shown}')
print('PROBLEMS', len(problems)); print('\n'.join(problems[:80]))
# frame dumps at every word start and every scene change
os.makedirs('qa/frames', exist_ok=True)
for i, w in enumerate(words):
    fi = min(int(np.ceil(w['s'] * FPS)) + 3, len(frames) - 1)
    cv2.imwrite(f'qa/frames/w{i:02d}_{fi:03d}.jpg', cv2.resize(frames[fi], (360, 640)))
for s in SC['scenes']:
    fi = int(round(s['start'] * FPS))
    for d in (-1, 0, 1):
        if 0 <= fi + d < len(frames): cv2.imwrite(f'qa/frames/sc{s["id"]}_{fi + d:03d}.jpg', cv2.resize(frames[fi + d], (360, 640)))
