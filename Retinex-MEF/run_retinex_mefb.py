import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
from nets.net import L_net, fusionnet
from utils import calculate_k, FuseIlluminance
from mef_io import ROOT, MEFB_INPUT, imread_rgb, imwrite_rgb, list_mefb_pairs


def to_tensor(img, device):
    return torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.).to(device)


@torch.no_grad()
def fuse(net_L, net_fusion, img1, img2, exposure=None):
    L1, L2 = net_L(img1), net_L(img2)
    L = FuseIlluminance(L1, L2, calculate_k(L1, L2, exposure)) if exposure is not None else (L1 + L2) / 2
    _, _, Rhat = net_fusion(img1, img2, L)
    return (L * Rhat).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=os.path.join(HERE, 'model', 'ckpt.pth'))
    p.add_argument('--input', default=MEFB_INPUT)
    p.add_argument('--output', default=None)
    p.add_argument('--exposure', type=float, default=None, help='target mean exposure level, e.g. 0.6; default: (L1+L2)/2')
    args = p.parse_args()
    tag = 'MEFB' if args.exposure is None else f'MEFB_E{args.exposure}'
    out_dir = args.output or os.path.join(ROOT, 'results', 'retinex_mef', tag)
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    net_L, net_fusion = L_net().to(device).eval(), fusionnet().to(device).eval()
    net_L.load_state_dict(ckpt['net_L'])
    net_fusion.load_state_dict(ckpt['net_fusion'])

    rows = []
    for scene, pa, pb in tqdm(list_mefb_pairs(args.input)):
        img1, img2 = to_tensor(imread_rgb(pa), device), to_tensor(imread_rgb(pb), device)
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fuse(net_L, net_fusion, img1, img2, args.exposure)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else 0
        imwrite_rgb(os.path.join(out_dir, scene + '.png'),
                    np.uint8(out[0].cpu().numpy().transpose(1, 2, 0) * 255))
        rows.append([scene, img1.shape[2], img1.shape[3], f'{dt:.4f}', f'{mem:.0f}'])

    with open(os.path.join(out_dir, '_runtime.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['scene', 'H', 'W', 'seconds', 'peak_MiB']] + rows)
    times = [float(r[3]) for r in rows[1:]]
    print(f'{len(rows)} images -> {out_dir}; mean time (excl. first) {np.mean(times):.3f}s; '
          f'max peak mem {max(float(r[4]) for r in rows):.0f} MiB')


if __name__ == '__main__':
    main()
