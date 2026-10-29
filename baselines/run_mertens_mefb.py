import os
import sys
import csv
import time
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from mef_io import ROOT, imread_rgb, imwrite_rgb, list_mefb_pairs


def main():
    out_dir = os.path.join(ROOT, 'results', 'mertens', 'MEFB')
    merge = cv2.createMergeMertens()
    rows = []
    for scene, pa, pb in tqdm(list_mefb_pairs()):
        imgs = [imread_rgb(pa), imread_rgb(pb)]
        t0 = time.perf_counter()
        out = merge.process(imgs)
        dt = time.perf_counter() - t0
        imwrite_rgb(os.path.join(out_dir, scene + '.png'), np.uint8(np.clip(out * 255, 0, 255) + 0.5))
        rows.append([scene, f'{dt:.4f}'])
    with open(os.path.join(out_dir, '_runtime.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['scene', 'seconds']] + rows)
    print(f'{len(rows)} images -> {out_dir}; mean time {np.mean([float(r[1]) for r in rows]):.3f}s')


if __name__ == '__main__':
    main()
