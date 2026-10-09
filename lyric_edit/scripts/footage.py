"""Step 2: per-frame subject analysis of the (letterbox-cropped) footage.
- face detection (MediaPipe BlazeFace) -> main subject box per frame
- person mask (MediaPipe selfie multiclass), temporally smoothed per shot
Outputs: subject.json (per-frame centre/face box), masks/NNNN.png (half res)."""
import json, os, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MD = '/tmp/claude-0/models'
CUTS = [0.0, 2.55, 3.50, 3.983, 4.35, 4.983, 6.833, 99]
pts = np.array([float(l.split(',')[0]) for l in open('src_pts.txt')])
N = len(pts)
shot_of = np.searchsorted(CUTS, pts, side='right') - 1

fd = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
    base_options=mpt.BaseOptions(model_asset_path=f'{MD}/blaze_face_short_range.tflite'),
    min_detection_confidence=0.35))
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=mpt.BaseOptions(model_asset_path=f'{MD}/selfie_multiclass_256x256.tflite'),
    output_confidence_masks=True))

os.makedirs('masks', exist_ok=True)
faces_all, masks_raw = [], []
for i in range(N):
    bgr = cv2.imread(f'src_frames/{i+1:04d}.jpg')
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # face detector is short-range: run on overlapping tiles for small faces
    faces = []
    for x0 in range(0, W - H + 1, (W - H) // 3):
        tile = np.ascontiguousarray(rgb[:, x0:x0 + H])
        r = fd.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile))
        for d in r.detections:
            b = d.bounding_box
            faces.append([x0 + b.origin_x, b.origin_y, b.width, b.height, d.categories[0].score])
    # merge duplicates from overlapping tiles
    faces.sort(key=lambda f: -f[4]); keep = []
    for f in faces:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        if all(abs(cx - (k[0] + k[2] / 2)) > k[2] * 0.6 or abs(cy - (k[1] + k[3] / 2)) > k[3] * 0.6 for k in keep):
            keep.append(f)
    faces_all.append(keep)
    # person mask: 1 - background confidence (half res)
    small = cv2.resize(rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
    bg = r.confidence_masks[0].numpy_view()
    masks_raw.append((1.0 - bg).astype(np.float32))
    if i % 60 == 0: print('frame', i, 'faces', [(int(f[0]), int(f[1]), int(f[2]), round(f[4], 2)) for f in keep])

# temporal smoothing of masks within each shot: centred EMA (forward + backward)
alpha = 0.55
sm = [None] * N
for s in range(len(CUTS) - 1):
    idx = [i for i in range(N) if shot_of[i] == s]
    if not idx: continue
    fwd, bwd = {}, {}
    acc = None
    for i in idx: acc = masks_raw[i] if acc is None else alpha * masks_raw[i] + (1 - alpha) * acc; fwd[i] = acc
    acc = None
    for i in reversed(idx): acc = masks_raw[i] if acc is None else alpha * masks_raw[i] + (1 - alpha) * acc; bwd[i] = acc
    for i in idx:
        m = 0.5 * (fwd[i] + bwd[i])
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        sm[i] = m
for i in range(N):
    cv2.imwrite(f'masks/{i+1:04d}.png', np.clip(sm[i] * 255, 0, 255).astype(np.uint8))

json.dump(dict(pts=pts.tolist(), shot=shot_of.tolist(), cuts=CUTS, faces=faces_all), open('faces_raw.json', 'w'))
print('done', N)
