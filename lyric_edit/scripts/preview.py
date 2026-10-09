"""render selected times of a scene to a contact sheet: python3 preview.py l1 0.3 1.0 ..."""
import sys, importlib, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from engine import core
mod = importlib.import_module('engine.scene_' + sys.argv[1])
ts = [float(x) for x in sys.argv[2:]]
tiles = []
for t in ts:
    fx = {}
    c = np.clip(mod.render(t, fx), 0, 1)
    im = cv2.cvtColor((c * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    im = cv2.resize(im, (360, 640), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f'{t:.2f} {fx}', (6, 20), 0, 0.5, (0, 255, 0), 1)
    tiles.append(im)
cv2.imwrite(f'/tmp/claude-0/prev_{sys.argv[1]}.jpg', np.hstack(tiles))
