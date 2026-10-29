import os
import sys
import csv
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
from e2emef import E2EMEF
from batch_transformers import BatchRGBToYCbCr, YCbCrToRGB
from mef_io import ROOT, imread_rgb, imwrite_rgb, list_mefb_pairs

EPS = 1e-8
LOW_SIZE, TEST_MAX = 128, 2048
BILINEAR = transforms.InterpolationMode.BILINEAR


def load_seq(paths):
    imgs = [Image.fromarray(imread_rgb(p)) for p in paths]
    w, h = imgs[0].size
    hr = [transforms.Resize(TEST_MAX, BILINEAR)(i) for i in imgs] if w > TEST_MAX and h > TEST_MAX else imgs
    lr = [transforms.Resize(LOW_SIZE, BILINEAR)(i) for i in imgs]
    to_ycc = lambda seq: torch.stack(BatchRGBToYCbCr()([transforms.ToTensor()(i) for i in seq]))
    return to_ycc(hr), to_ycc(lr)


@torch.no_grad()
def fuse(model, i_hr, i_lr, device):
    Y_hr, Cb_hr, Cr_hr = i_hr[:, 0:1], i_hr[:, 1:2], i_hr[:, 2:3]
    Wb = (torch.abs(Cb_hr - 0.5) + EPS) / torch.sum(torch.abs(Cb_hr - 0.5) + EPS, dim=0)
    Wr = (torch.abs(Cr_hr - 0.5) + EPS) / torch.sum(torch.abs(Cr_hr - 0.5) + EPS, dim=0)
    Cb_f = torch.sum(Wb * Cb_hr, dim=0, keepdim=True).clamp(0, 1)
    Cr_f = torch.sum(Wr * Cr_hr, dim=0, keepdim=True).clamp(0, 1)
    O_hr, _ = model(i_lr[:, 0:1].to(device), Y_hr.to(device))
    return YCbCrToRGB()(torch.cat((O_hr.cpu(), Cb_f, Cr_f), dim=1)).clamp(0, 1)[0]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = E2EMEF(is_guided=True).to(device).eval()
    ckpt = torch.load(os.path.join(HERE, 'checkpoint', 'MEFNet_release.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    out_dir = os.path.join(ROOT, 'results', 'mefnet', 'MEFB')
    rows = []
    for scene, pa, pb in tqdm(list_mefb_pairs()):
        i_hr, i_lr = load_seq([pa, pb])
        t0 = time.perf_counter()
        out = fuse(model, i_hr, i_lr, device)
        dt = time.perf_counter() - t0
        imwrite_rgb(os.path.join(out_dir, scene + '.png'), np.uint8(out.numpy().transpose(1, 2, 0) * 255 + 0.5))
        rows.append([scene, f'{dt:.4f}'])
    with open(os.path.join(out_dir, '_runtime.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['scene', 'seconds']] + rows)
    print(f'{len(rows)} images -> {out_dir}; mean time (excl. first) {np.mean([float(r[1]) for r in rows[1:]]):.3f}s')


if __name__ == '__main__':
    main()
