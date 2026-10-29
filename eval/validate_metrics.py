"""Check the Python metric port against MEFB's published MATLAB scores (evaluationMetrics.mat).

Run eval_mefb.py --mefb-official <methods> first; this compares results/metrics/MEFB_<method>(MEFB).csv
with the .mat values image-by-image and reports relative error per metric.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
from mef_metrics import MEFB_MAT_INDEX
from mef_io import ROOT

OUT = os.path.join(ROOT, 'datasets', 'MEFB', 'output')


def read_info():
    sections, cur = {}, None
    for line in open(os.path.join(OUT, 'information.txt'), encoding='utf-8', errors='ignore'):
        s = line.strip()
        if s.startswith('The ') and s.endswith(':'):
            cur = s.split()[1]
            sections[cur] = []
        elif s and cur:
            sections[cur].append(s)
    return sections['image'], sections['methods'], sections['metrics']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', required=True)
    args = p.parse_args()
    imgs, methods, _ = read_info()
    mat = sio.loadmat(os.path.join(OUT, 'evaluation_metrics', 'evaluationMetrics.mat'))['resultsMetrics']
    row = {n.lower(): i for i, n in enumerate(imgs)}

    report = {}
    for m in args.methods:
        ours = pd.read_csv(os.path.join(ROOT, 'results', 'metrics', f'MEFB_{m}(MEFB).csv'), index_col=0)
        ref = mat[[row[s.lower()] for s in ours.index], methods.index(m)]
        for k, j in MEFB_MAT_INDEX.items():
            if k not in ours:
                continue
            rel = np.abs(ours[k].values - ref[:, j]) / np.maximum(np.abs(ref[:, j]), 1e-12)
            report[(m, k)] = {'ours_mean': ours[k].mean(), 'matlab_mean': ref[:, j].mean(),
                              'median_rel_err': np.median(rel), 'max_rel_err': rel.max(),
                              'n_rel_err>1%': int((rel > 0.01).sum())}
    df = pd.DataFrame(report).T
    pd.set_option('display.width', 200)
    print(df.to_string(float_format=lambda v: f'{v:.5g}'))


if __name__ == '__main__':
    main()
