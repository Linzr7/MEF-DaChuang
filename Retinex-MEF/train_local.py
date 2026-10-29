"""Retinex-MEF training adapted for an 8GB laptop GPU (Windows, Chinese paths).

Identical to the official train.py in model, losses, loss weights, optimizers (two Adams, lr 1e-4),
StepLR(10, 0.1) and 40 epochs. Additions: bf16 autocast (batch 8 fits in ~4GB instead of ~8GB),
unicode-safe image IO, resume, CSV logs, per-epoch validation on held-out SICE scenes.

  ..\.venv\Scripts\python.exe train_local.py --data ..\datasets\SICE\patches_L1024_35k
  ..\.venv\Scripts\python.exe train_local.py --resume exp\<name>\state_last.pth
"""
import os
import sys
import csv
import json
import time
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'eval'))
from nets.net import L_net, fusionnet
from utils import illu_smooth
from mef_io import imread_rgb
from mef_metrics import mef_ssim

COEF = (1, 0.5, 0.1, 0.2, 0.1)
LOSS_NAMES = ['total', 'recon', 'smooth', 'initialize', 'suppress', 'consist']


class SICEPatches(Dataset):
    def __init__(self, root):
        self.root = root
        self.n = len(os.listdir(root))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = os.path.join(self.root, str(i))
        return tuple(torch.from_numpy(imread_rgb(os.path.join(d, f'img_{k}.png')).transpose(2, 0, 1).astype(np.float32) / 255.)
                     for k in (1, 2, 3))


def losses(net_L, net_fusion, img1, img2, img3, amp):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        L1, L2, L3 = net_L(img1), net_L(img2), net_L(img3)
        y3, R3, Rhat = net_fusion(img1, img2, L3)
    L1, L2, L3, y3, R3, Rhat = (t.float() for t in (L1, L2, L3, y3, R3, Rhat))
    recon = F.l1_loss(y3, img3)
    smooth = illu_smooth(L3, img3)
    initialize = F.l1_loss(L3, torch.max(img3, 1, keepdim=True)[0])
    suppress = torch.mean(F.relu(Rhat * L3.detach() - img3) + F.relu(Rhat * L2.detach() - img2) + F.relu(Rhat * L1.detach() - img1))
    consist = F.l1_loss(Rhat, R3)
    total = COEF[0] * recon + COEF[1] * smooth + COEF[2] * initialize + COEF[3] * suppress + COEF[4] * consist
    return total, [total, recon, smooth, initialize, suppress, consist]


def load_val_set(n_scenes, long_side):
    import cv2
    sice = os.path.join(ROOT, 'datasets', 'SICE')
    sorted_dir = os.path.join(sice, 'sorted_L1024')
    test = json.load(open(os.path.join(sice, 'split.json')))['test'][:n_scenes]
    out = []
    for s in test:
        n = len(os.listdir(os.path.join(sorted_dir, s)))
        imgs = [imread_rgb(os.path.join(sorted_dir, s, f'{k}.png')) for k in (1, n)]
        imgs.append(imread_rgb(os.path.join(sorted_dir + '_label', s + '.png')))
        h, w = imgs[0].shape[:2]
        sc = long_side / max(h, w)
        out.append([cv2.resize(im, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA) for im in imgs])
    return out


@torch.no_grad()
def validate(net_L, net_fusion, val_set):
    net_L.eval(), net_fusion.eval()
    ssims, psnrs = [], []
    for under, over, gt in val_set:
        t = lambda x: torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32) / 255.).cuda()
        i1, i2 = t(under), t(over)
        L = (net_L(i1) + net_L(i2)) / 2
        _, _, Rhat = net_fusion(i1, i2, L)
        f = np.uint8((L * Rhat).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255)
        ssims.append(mef_ssim([under, over], f))
        psnrs.append(10 * np.log10(255 ** 2 / np.mean((f.astype(np.float64) - gt) ** 2)))
    net_L.train(), net_fusion.train()
    return float(np.mean(ssims)), float(np.mean(psnrs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=None)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--step', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val-scenes', type=int, default=10)
    p.add_argument('--val-long-side', type=int, default=512)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--resume', default=None, help='exp/<name>/state_last.pth')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    amp = not args.no_amp

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    net_fusion, net_L = fusionnet().cuda(), L_net().cuda()
    opt1 = torch.optim.Adam(net_fusion.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(net_L.parameters(), lr=args.lr)
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=args.step, gamma=args.gamma)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=args.step, gamma=args.gamma)

    start_epoch = 0
    if args.resume:
        st = torch.load(args.resume, map_location='cuda', weights_only=False)
        for obj, key in ((net_fusion, 'net_fusion'), (net_L, 'net_L'), (opt1, 'opt1'), (opt2, 'opt2'), (sch1, 'sch1'), (sch2, 'sch2')):
            obj.load_state_dict(st[key])
        start_epoch, exppath, saved = st['epoch'], os.path.dirname(args.resume), st['args']
        args.data = args.data or saved['data']
        print(f'resumed from {args.resume} at epoch {start_epoch}')
    else:
        exppath = os.path.join(HERE, 'exp', time.strftime('%m_%d_%H_%M') + f'_b{args.batch}' + ('_bf16' if amp else ''))
        os.makedirs(os.path.join(exppath, 'model'), exist_ok=True)
        json.dump(vars(args), open(os.path.join(exppath, 'args.json'), 'w'), indent=1)
    assert args.data, '--data is required'

    loader = DataLoader(SICEPatches(args.data), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    val_set = load_val_set(args.val_scenes, args.val_long_side)
    log_csv = os.path.join(exppath, 'train_log.csv')
    if not os.path.exists(log_csv):
        with open(log_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'lr'] + LOSS_NAMES + ['val_mef_ssim', 'val_psnr_gt', 'epoch_minutes'])
    if start_epoch == 0:
        print('epoch 0 validation (random init): MEF-SSIM %.4f, PSNR(gt) %.2f' % validate(net_L, net_fusion, val_set))

    n_iter = len(loader)
    print(f'{len(loader.dataset)} samples, {n_iter} iters/epoch, batch {args.batch}, bf16={amp}, exp={exppath}')
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        net_fusion.train(), net_L.train()
        sums, t_ep = np.zeros(len(LOSS_NAMES)), time.time()
        for i, (img1, img2, img3) in enumerate(loader):
            img1, img2, img3 = (x.cuda(non_blocking=True) for x in (img1, img2, img3))
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            total, parts = losses(net_L, net_fusion, img1, img2, img3, amp)
            total.backward()
            opt1.step()
            opt2.step()
            sums += [v.item() for v in parts]
            if (i + 1) % args.log_every == 0 or i + 1 == n_iter:
                done = (epoch - start_epoch) * n_iter + i + 1
                eta = (time.time() - t_start) / done * ((args.epochs - start_epoch) * n_iter - done)
                m = sums / (i + 1)
                print(f'[Epoch {epoch + 1}/{args.epochs}] [Batch {i + 1}/{n_iter}] ' +
                      ' '.join(f'{k}={v:.4f}' for k, v in zip(LOSS_NAMES, m)) +
                      f' ETA {datetime.timedelta(seconds=int(eta))}', flush=True)
        lr = opt1.param_groups[0]['lr']
        sch1.step(), sch2.step()
        val = validate(net_L, net_fusion, val_set)
        minutes = (time.time() - t_ep) / 60
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, lr] + [f'{v:.6f}' for v in sums / n_iter] + [f'{val[0]:.5f}', f'{val[1]:.3f}', f'{minutes:.1f}'])
        print(f'== epoch {epoch + 1}: val MEF-SSIM {val[0]:.4f}, PSNR(gt) {val[1]:.2f}, {minutes:.1f} min', flush=True)
        torch.save({'net_fusion': net_fusion.state_dict(), 'net_L': net_L.state_dict()},
                   os.path.join(exppath, 'model', f'ckpt_{epoch + 1}.pth'))
        torch.save({'net_fusion': net_fusion.state_dict(), 'net_L': net_L.state_dict(), 'opt1': opt1.state_dict(),
                    'opt2': opt2.state_dict(), 'sch1': sch1.state_dict(), 'sch2': sch2.state_dict(),
                    'epoch': epoch + 1, 'args': vars(args)}, os.path.join(exppath, 'state_last.pth'))


if __name__ == '__main__':
    main()
