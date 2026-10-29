import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEFB_INPUT = os.path.join(ROOT, 'datasets', 'MEFB', 'input')
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def imread_rgb(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cannot read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f'cannot encode {path}')
    buf.tofile(path)


def list_mefb_pairs(root=MEFB_INPUT):
    """Return [(scene, under_path, over_path)]. In MEFB, *_A is under-exposed and *_B is over-exposed."""
    pairs = []
    for scene in sorted(os.listdir(root)):
        d = os.path.join(root, scene)
        if not os.path.isdir(d):
            continue
        found = {}
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS and stem.lower() in (scene.lower() + '_a', scene.lower() + '_b'):
                found[stem[-1].upper()] = os.path.join(d, f)
        if len(found) != 2:
            raise FileNotFoundError(f'MEFB scene {scene}: cannot find _A/_B pair')
        pairs.append((scene, found['A'], found['B']))
    return pairs
