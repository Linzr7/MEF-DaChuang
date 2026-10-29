"""Plot Retinex-MEF train_log.csv (losses + held-out validation) -> docs/figures/train_curves_<exp>.png"""
import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mef_io import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exp', help='Retinex-MEF/exp/<name> directory')
    p.add_argument('--official-val', type=float, nargs=2, default=None, metavar=('MEF_SSIM', 'PSNR'),
                   help='validation scores of the official checkpoint, drawn as reference lines')
    args = p.parse_args()
    df = pd.read_csv(os.path.join(args.exp, 'train_log.csv'))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in ['recon', 'smooth', 'initialize', 'suppress', 'consist']:
        ax[0].plot(df.epoch, df[k], label=k)
    ax[0].set_yscale('log'), ax[0].set_title('loss terms'), ax[0].legend()
    ax[1].plot(df.epoch, df.total, 'k'), ax[1].set_title('total loss')
    ax[2].plot(df.epoch, df.val_mef_ssim, 'b-o', ms=3, label='self-trained')
    ax[2].set_title('held-out SICE: MEF-SSIM'), ax[2].set_ylabel('MEF-SSIM', color='b')
    a2 = ax[2].twinx()
    a2.plot(df.epoch, df.val_psnr_gt, 'r-s', ms=3), a2.set_ylabel('PSNR vs SICE label (dB)', color='r')
    if args.official_val:
        ax[2].axhline(args.official_val[0], color='b', ls='--', label='official ckpt')
        a2.axhline(args.official_val[1], color='r', ls='--')
        ax[2].legend(loc='lower right')
    for a in ax:
        a.set_xlabel('epoch'), a.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(ROOT, 'docs', 'figures', f'train_curves_{os.path.basename(os.path.normpath(args.exp))}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()
