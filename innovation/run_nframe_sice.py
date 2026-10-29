"""Innovation 1: N-frame Retinex-MEF ablation on the 30 held-out SICE scenes (never used for training).

  python innovation/run_nframe_sice.py [--ckpts self official] [--modes pair2 meanL ...]
Inputs : datasets/SICE/sorted_L1024/<scene>/<k>.png (dark->bright), GT: sorted_L1024_label/<scene>.png
Outputs: results/nframe/<ckpt>_<mode>/SICE/<scene>.png, results/metrics/SICE_nframe_{perimage.csv,summary.csv,summary.md}
Metrics: MEF-SSIM against ALL frames (MEF-SSIM_N) and against the two end frames (MEF-SSIM_2),
         PSNR / SSIM against the SICE reference image, plus EN / SD / AG.
"""
import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'eval')]
from mef_io import imread_rgb, imwrite_rgb
import mef_metrics as M

SICE = os.path.join(ROOT, 'datasets', 'SICE')
SORTED = os.path.join(SICE, 'sorted_L1024')
OUT = os.path.join(ROOT, 'results', 'nframe')


def load_scene(s):
    n = len(os.listdir(os.path.join(SORTED, s)))
    return [imread_rgb(os.path.join(SORTED, s, f'{k}.png')) for k in range(1, n + 1)]


def metric_job(args):
    method, scene, path = args
    frames = load_scene(scene)
    gt = imread_rgb(os.path.join(SORTED + '_label', scene + '.png')).astype(np.float64)
    f = imread_rgb(path)
    psnr = 10 * np.log10(255 ** 2 / np.mean((f - gt) ** 2))
    from skimage.metrics import structural_similarity
    ssim = structural_similarity(M.rgb2gray(f), M.rgb2gray(gt), data_range=255, gaussian_weights=True,
                                 sigma=1.5, use_sample_covariance=False)
    return {'method': method, 'scene': scene, 'N': len(frames),
            'MEF-SSIM_N': M.mef_ssim(frames, f), 'MEF-SSIM_2': M.mef_ssim([frames[0], frames[-1]], f),
            'PSNR_gt': psnr, 'SSIM_gt': ssim, 'EN': M.entropy(frames, f), 'SD': M.sd(frames, f), 'AG': M.ag(frames, f)}


def main():
    import torch
    from nframe import MODES, DEFAULT_CKPT, OFFICIAL_CKPT, load_model, fuse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', default=['self', 'official'])
    p.add_argument('--modes', nargs='+', default=MODES)
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    args = p.parse_args()
    test = json.load(open(os.path.join(SICE, 'split.json')))['test']

    jobs = []
    merge = cv2.createMergeMertens()
    for name, frames_sel in (('mertens_N', slice(None)), ('mertens_2', [0, -1])):
        for s in tqdm(test, desc=name):
            path = os.path.join(OUT, name, 'SICE', s + '.png')
            if not os.path.exists(path):
                fr = load_scene(s)
                fr = fr if isinstance(frames_sel, slice) else [fr[i] for i in frames_sel]
                imwrite_rgb(path, np.uint8(np.clip(merge.process(fr) * 255, 0, 255) + 0.5))
            jobs.append((name, s, path))

    for ck in args.ckpts:
        net_L, net_f = load_model({'self': DEFAULT_CKPT, 'official': OFFICIAL_CKPT}[ck])
        for mode in args.modes:
            name = f'{ck}_{mode}'
            for s in tqdm(test, desc=name):
                path = os.path.join(OUT, name, 'SICE', s + '.png')
                if not os.path.exists(path):
                    imwrite_rgb(path, fuse(net_L, net_f, load_scene(s), mode))
                jobs.append((name, s, path))
        del net_L, net_f
        torch.cuda.empty_cache()

    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(metric_job, jobs, chunksize=2), total=len(jobs), desc='metrics'))
    df = pd.DataFrame(rows)
    mdir = os.path.join(ROOT, 'results', 'metrics')
    os.makedirs(mdir, exist_ok=True)
    df.to_csv(os.path.join(mdir, 'SICE_nframe_perimage.csv'), index=False, float_format='%.6f')
    cols = ['MEF-SSIM_N', 'MEF-SSIM_2', 'PSNR_gt', 'SSIM_gt', 'EN', 'SD', 'AG']
    order = list(dict.fromkeys(df.method))
    summary = df.groupby('method')[cols].mean().loc[order]
    many = df[df.N >= 7].groupby('method')[['MEF-SSIM_N', 'PSNR_gt', 'SSIM_gt']].mean().loc[order]
    summary = summary.join(many.add_suffix(' (N>=7)'))
    summary.to_csv(os.path.join(mdir, 'SICE_nframe_summary.csv'), float_format='%.4f')
    with open(os.path.join(mdir, 'SICE_nframe_summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(f'30 held-out SICE scenes (N = {df.groupby("scene").N.first().min()}..{df.groupby("scene").N.first().max()}, '
                 f'{(df.groupby("scene").N.first() >= 7).sum()} scenes with N>=7)\n\n')
        fh.write('| method | ' + ' | '.join(summary.columns) + ' |\n|---' * 1 + '|---' * len(summary.columns) + '|\n')
        for m, r in summary.iterrows():
            fh.write(f'| {m} | ' + ' | '.join(f'{v:.4f}' for v in r) + ' |\n')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
