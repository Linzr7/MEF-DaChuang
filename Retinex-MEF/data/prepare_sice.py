"""SICE Part1 -> Retinex-MEF training patches, sized for an 8GB laptop GPU.

Combines the official SICE_arrange.py (sort frames by mean gray level, drop exact-duplicate brightness) and
SICE_processing.py (128x128 stride-128 patches; img_1 = darkest, img_2 = brightest, img_3 = each frame j),
with two changes: frames are first downscaled to --long-side, and patch positions are randomly subsampled so the
total number of samples is at most --max-samples. A fixed set of held-out scenes is excluded for evaluation.

Outputs
  datasets/SICE/sorted_L<long>/<scene>/<k>.png      sorted, resized frames (all scenes, k = 1..N dark->bright)
  datasets/SICE/sorted_L<long>_label/<scene>.png    resized SICE reference images
  datasets/SICE/split.json                          {"train": [...], "test": [...]}
  datasets/SICE/patches_L<long>_<n>k/<idx>/img_{1,2,3}.png
"""
import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from mef_io import imread_rgb, imwrite_rgb

SICE = os.path.join(ROOT, 'datasets', 'SICE')
RAW = os.path.join(SICE, 'Dataset_Part1')
PATCH = 128


def load_resized(path, long_side, size=None):
    img = Image.open(path)
    w, h = img.size
    if size is None:
        s = long_side / max(w, h)
        size = (round(w * s), round(h * s)) if s < 1 else (w, h)
    img.draft('RGB', size)  # fast JPEG DCT-domain downscale to >= size
    img = np.asarray(img.convert('RGB'))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if img.shape[1::-1] != size else img


def arrange_scene(args):
    scene, long_side, out_dir, label_dir = args
    d = os.path.join(RAW, scene)
    names = sorted(f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    first = load_resized(os.path.join(d, names[0]), long_side)
    size = first.shape[1::-1]
    frames = [first] + [load_resized(os.path.join(d, n), long_side, size) for n in names[1:]]
    bright = [np.mean(np.round(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float64))) for f in frames]
    order = np.argsort(bright, kind='stable')
    kept = [order[0]] + [b for a, b in zip(order[:-1], order[1:]) if bright[b] != bright[a]]
    os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
    for k, i in enumerate(kept, 1):
        imwrite_rgb(os.path.join(out_dir, scene, f'{k}.png'), frames[i])
    imwrite_rgb(os.path.join(label_dir, scene + '.png'), load_resized(os.path.join(RAW, 'Label', scene + '.JPG'), long_side, size))
    return scene, len(names), len(kept), size


def patch_scene(args):
    scene, sorted_dir, out_dir, start_idx, positions = args
    n = len(os.listdir(os.path.join(sorted_dir, scene)))
    frames = [imread_rgb(os.path.join(sorted_dir, scene, f'{k}.png')) for k in range(1, n + 1)]
    idx = start_idx
    for j, pos in enumerate(positions):
        for (y, x) in pos:
            d = os.path.join(out_dir, str(idx))
            os.makedirs(d, exist_ok=True)
            for name, img in (('img_1', frames[0]), ('img_2', frames[-1]), ('img_3', frames[j])):
                imwrite_rgb(os.path.join(d, name + '.png'), img[y:y + PATCH, x:x + PATCH])
            idx += 1
    return idx - start_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--long-side', type=int, default=1024)
    p.add_argument('--max-samples', type=int, default=35000)
    p.add_argument('--n-test', type=int, default=30, help='held-out scenes (every k-th scene by numeric id)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    args = p.parse_args()

    scenes = sorted((s for s in os.listdir(RAW) if s.isdigit()), key=int)
    step = len(scenes) // args.n_test
    test = scenes[step - 1::step][:args.n_test]
    train = [s for s in scenes if s not in set(test)]
    with open(os.path.join(SICE, 'split.json'), 'w') as f:
        json.dump({'train': train, 'test': test}, f, indent=1)

    sorted_dir = os.path.join(SICE, f'sorted_L{args.long_side}')
    label_dir = sorted_dir + '_label'
    os.makedirs(label_dir, exist_ok=True)
    todo = [s for s in scenes if not os.path.exists(os.path.join(label_dir, s + '.png'))]
    with ProcessPoolExecutor(args.workers) as ex:
        for scene, n_raw, n_kept, size in tqdm(ex.map(arrange_scene, [(s, args.long_side, sorted_dir, label_dir) for s in todo]),
                                               total=len(todo), desc='arrange'):
            if n_kept != n_raw:
                tqdm.write(f'scene {scene}: {n_raw} frames -> {n_kept} after dedup, size {size}')

    rng = np.random.default_rng(args.seed)
    grids, total = {}, 0
    for s in train:
        n = len(os.listdir(os.path.join(sorted_dir, s)))
        h, w = imread_rgb(os.path.join(sorted_dir, s, '1.png')).shape[:2]
        grid = [(y, x) for y in range(0, h - PATCH + 1, PATCH) for x in range(0, w - PATCH + 1, PATCH)]
        grids[s] = (n, grid)
        total += n * len(grid)
    ratio = min(1.0, args.max_samples / total)
    jobs, idx = [], 0
    for s in train:
        n, grid = grids[s]
        k = max(1, round(ratio * len(grid)))
        positions = [[grid[i] for i in rng.choice(len(grid), k, replace=False)] for _ in range(n)]
        jobs.append((s, sorted_dir, None, idx, positions))
        idx += n * k
    out_dir = os.path.join(SICE, f'patches_L{args.long_side}_{round(idx / 1000)}k')
    jobs = [(s, sd, out_dir, i, pos) for s, sd, _, i, pos in jobs]
    print(f'train scenes {len(train)}, test scenes {len(test)}; available patches {total}, keep ratio {ratio:.3f} -> {idx} samples')
    with ProcessPoolExecutor(args.workers) as ex:
        written = sum(tqdm(ex.map(patch_scene, jobs), total=len(jobs), desc='patches'))
    print(f'wrote {written} samples to {out_dir}')


if __name__ == '__main__':
    main()
