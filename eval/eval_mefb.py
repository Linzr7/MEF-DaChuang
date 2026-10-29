"""Evaluate fused results on MEFB with the MEFB metric suite.

  python eval/eval_mefb.py --methods mertens mefnet transmef retinex_mef
Fused images are read from results/<method>/MEFB/<scene>.png (or --dir <method>=<path> for other folders).
Writes results/metrics/MEFB_<method>.csv (per image) and results/metrics/MEFB_summary.{csv,md}.
"""
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
from mef_metrics import METRICS, compute_all
from mef_io import ROOT, imread_rgb, list_mefb_pairs

HIGHER_IS_BETTER = set(METRICS)  # every metric in the suite is "higher is better"


MEFB_OFFICIAL_DIR = os.path.join(ROOT, 'datasets', 'MEFB', 'output', 'fused_images')
_listing = {}


def find_fused(folder, scene, template='{scene}'):
    """Case-insensitive lookup of folder/<template>.<ext> (MEFB official files are '{scene}_{method}.tif')."""
    if folder not in _listing:
        _listing[folder] = {os.path.splitext(f)[0].lower(): f for f in os.listdir(folder)
                            if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.tif', '.bmp')}
    key = template.format(scene=scene).lower()
    if key not in _listing[folder]:
        raise FileNotFoundError(f'{folder}: no fused image for {key}')
    return os.path.join(folder, _listing[folder][key])


def _job(args):
    scene, pa, pb, pf, names = args
    srcs = [imread_rgb(pa), imread_rgb(pb)]
    f = imread_rgb(pf)
    if f.shape != srcs[0].shape:
        raise ValueError(f'{pf}: shape {f.shape} != source {srcs[0].shape}')
    return {'scene': scene, **compute_all(srcs, f, names)}


def evaluate(folder, names=None, workers=None, template='{scene}', desc=None):
    jobs = [(s, a, b, find_fused(folder, s, template), names) for s, a, b in list_mefb_pairs()]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(tqdm(ex.map(_job, jobs), total=len(jobs), desc=desc))
    return pd.DataFrame(rows).set_index('scene')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='*', default=[])
    p.add_argument('--dir', nargs='*', default=[], help='extra entries as name=folder')
    p.add_argument('--mefb-official', nargs='*', default=[],
                   help='methods from MEFB fused_images (e.g. GFF MEFNet U2Fusion), stored as "<method>(MEFB)"')
    p.add_argument('--metrics', nargs='*', default=None)
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument('--out', default=os.path.join(ROOT, 'results', 'metrics'))
    args = p.parse_args()
    entries = [(m, os.path.join(ROOT, 'results', m, 'MEFB'), '{scene}') for m in args.methods]
    entries += [(*d.split('=', 1), '{scene}') for d in args.dir]
    entries += [(f'{m}(MEFB)', MEFB_OFFICIAL_DIR, '{scene}_' + m) for m in args.mefb_official]
    os.makedirs(args.out, exist_ok=True)

    for name, folder, template in entries:
        df = evaluate(folder, args.metrics, args.workers, template, desc=name)
        df.to_csv(os.path.join(args.out, f'MEFB_{name}.csv'), float_format='%.6f')

    summary_csv = os.path.join(args.out, 'MEFB_summary.csv')
    rows = pd.read_csv(summary_csv, index_col=0).to_dict('index') if os.path.exists(summary_csv) else {}
    for name, _, _ in entries:
        rows[name] = pd.read_csv(os.path.join(args.out, f'MEFB_{name}.csv'), index_col=0).mean().to_dict()
    summary = pd.DataFrame.from_dict(rows, orient='index')
    summary = summary[[c for c in METRICS if c in summary.columns]]
    summary.to_csv(summary_csv, float_format='%.4f')
    write_markdown(summary, os.path.join(args.out, 'MEFB_summary.md'))
    print(summary.round(4).to_string())


def write_markdown(summary, path):
    best = {c: summary[c].idxmax() for c in summary.columns}
    lines = ['| Method | ' + ' | '.join(summary.columns) + ' |', '|---' * (len(summary.columns) + 1) + '|']
    for m, row in summary.iterrows():
        cells = [f'**{v:.4f}**' if best[c] == m else f'{v:.4f}' for c, v in row.items()]
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
