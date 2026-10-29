"""TransMEF on MEFB at native resolution: official 256x256 block fusion on Y (fusion_arbitary_size_TransMEF_gray.py)
+ Cb/Cr fusion weighted by |C-128| (fusedY2RGB.m)."""
import os
import sys
import csv
import time
from collections import OrderedDict
import cv2
import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
from Network_TransMEF import TransNet
from mef_io import ROOT, imread_rgb, imwrite_rgb, list_mefb_pairs

BLOCK = 256


def to_blocks(y):
    m, n = y.shape
    pad = np.pad(y, ((0, -m % BLOCK), (0, -n % BLOCK)), 'reflect')
    mb, nb = pad.shape[0] // BLOCK, pad.shape[1] // BLOCK
    return pad.reshape(mb, BLOCK, nb, BLOCK).transpose(0, 2, 1, 3).reshape(-1, 1, BLOCK, BLOCK), mb, nb


@torch.no_grad()
def fuse_y(model, y1, y2, device):
    b1, mb, nb = to_blocks(y1)
    b2, _, _ = to_blocks(y2)
    x1 = torch.from_numpy(b1).float().div(255).to(device)
    x2 = torch.from_numpy(b2).float().div(255).to(device)
    out = torch.cat([model.decoder((model.encoder(x1[i:i + 8]) + model.encoder(x2[i:i + 8])) / 2)
                     for i in range(0, len(x1), 8)]).clamp(0, 1).cpu().numpy()
    full = out.reshape(mb, nb, BLOCK, BLOCK).transpose(0, 2, 1, 3).reshape(mb * BLOCK, nb * BLOCK)
    return full[:y1.shape[0], :y1.shape[1]] * 255


def fuse_chroma(c1, c2):
    w1, w2 = np.abs(c1 - 128), np.abs(c2 - 128)
    den = w1 + w2
    return np.where(den == 0, 128, (c1 * w1 + c2 * w2) / np.maximum(den, 1e-12))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransNet().to(device).eval()
    sd = torch.load(os.path.join(HERE, 'best_model.pth'), map_location=device, weights_only=False)['model']
    model.load_state_dict(OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in sd.items()))

    out_dir = os.path.join(ROOT, 'results', 'transmef', 'MEFB')
    rows = []
    for scene, pa, pb in tqdm(list_mefb_pairs()):
        ycc = [cv2.cvtColor(imread_rgb(p), cv2.COLOR_RGB2YCrCb).astype(np.float64) for p in (pa, pb)]
        t0 = time.perf_counter()
        y = fuse_y(model, ycc[0][..., 0].astype(np.uint8), ycc[1][..., 0].astype(np.uint8), device)
        dt = time.perf_counter() - t0
        f = np.stack([y, fuse_chroma(ycc[0][..., 1], ycc[1][..., 1]), fuse_chroma(ycc[0][..., 2], ycc[1][..., 2])], -1)
        rgb = cv2.cvtColor(np.clip(f + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        imwrite_rgb(os.path.join(out_dir, scene + '.png'), rgb)
        rows.append([scene, f'{dt:.4f}'])
    with open(os.path.join(out_dir, '_runtime.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows([['scene', 'seconds']] + rows)
    print(f'{len(rows)} images -> {out_dir}; mean Y-fusion time (excl. first) {np.mean([float(r[1]) for r in rows[1:]]):.3f}s')


if __name__ == '__main__':
    main()
