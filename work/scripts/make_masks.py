# Person masks (u2net_human_seg) on a square region around the subject, temporally smoothed within shots.
import sys, cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, 'scripts'); import footage as F
s = ort.InferenceSession('models/u2net_human_seg.onnx', providers=['CPUExecutionProvider'])
inp = s.get_inputs()[0].name
import os; os.makedirs('masks', exist_ok=True)
raw = []
for i, t in enumerate(F.PTS):
    im = F.frame(i); cx, _ = F.subject(t)
    x0 = int(np.clip(cx - 534, 0, F.W - 1068))
    reg = im[:, x0:x0 + 1068]
    x = cv2.resize(reg, (320, 320))[:, :, ::-1].astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    o = s.run(None, {inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    o = (o - o.min()) / (o.max() - o.min() + 1e-6)
    full = np.zeros((F.H, F.W), np.float32); full[:, x0:x0 + 1068] = cv2.resize(o, (1068, 1068))
    raw.append(full)
    if i % 50 == 0: print(i, flush=True)
for i in range(len(raw)):
    sh = F.shot_of(F.PTS[i]); acc = raw[i] * 0.5; w = 0.5
    for j, wt in ((i - 1, .25), (i + 1, .25)):
        if 0 <= j < len(raw) and F.shot_of(F.PTS[j]) == sh: acc += raw[j] * wt; w += wt
    m = cv2.GaussianBlur(acc / w, (0, 0), 1.5)
    cv2.imwrite(f'masks/{i + 1:04d}.png', (np.clip(m, 0, 1) * 255).astype(np.uint8))
print('done')
