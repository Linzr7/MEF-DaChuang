"""N-frame extension of Retinex-MEF (training-free).

Official inference (test.py) outputs  F = L_hat * R_hat,  R_hat = SRE(I_under, I_over),  L_hat = (L_u + L_o) / 2,
i.e. only two frames are used and the refinement decoder output is not used. For a sequence I_1..I_N sorted dark->bright
we combine the per-frame illuminations L_k = N_L(I_k) and shared reflectances from several (dark, bright) pairs:

  pair2     : L = (L_1 + L_N)/2,                R = SRE(I_1, I_N)                  (official, 2 frames)
  meanL     : L = mean_k L_k,                   R = SRE(I_1, I_N)
  wL        : L = sum_k w_k L_k / sum_k w_k,    R = SRE(I_1, I_N),  w_k = well-exposedness of I_k (Mertens)
  meanR     : L = (L_1 + L_N)/2,                R = mean over symmetric pairs (k, N+1-k) of SRE(I_k, I_{N+1-k})
  meanLR    : L = mean_k L_k,                   R = meanR
  wLR       : L = wL,                           R = meanR
Optional exposure control (paper Eq. 15-17): L <- f(L; k) with k solved so that mean(L) = E.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Retinex-MEF'))
from nets.net import L_net, fusionnet  # noqa: E402

MODES = ['pair2', 'meanL', 'wL', 'meanR', 'meanLR', 'wLR']
DEFAULT_CKPT = os.path.join(ROOT, 'Retinex-MEF', 'exp', '10_18_13_59_b8_bf16', 'model', 'ckpt_40.pth')
OFFICIAL_CKPT = os.path.join(ROOT, 'Retinex-MEF', 'model', 'ckpt.pth')


def load_model(ckpt=DEFAULT_CKPT, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    st = torch.load(ckpt, map_location=device)
    net_L, net_f = L_net().to(device).eval(), fusionnet().to(device).eval()
    net_L.load_state_dict(st['net_L'])
    net_f.load_state_dict(st['net_fusion'])
    return net_L, net_f


def to_tensor(img, device):
    return torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.).to(device)


def sort_by_brightness(frames):
    return sorted(frames, key=lambda f: float(np.mean(f)))


def _exposure_curve(x, k):
    return k * x / (k * x + (1 - k) * (1 - x))


def adjust_exposure(L, E, iters=40):
    """Bisection for k such that mean(f(L; k)) = E (paper Eq. 16-17)."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        k = (lo + hi) / 2
        if _exposure_curve(L, k).mean().item() < E:
            lo = k
        else:
            hi = k
    return _exposure_curve(L, (lo + hi) / 2)


@torch.no_grad()
def fuse(net_L, net_f, frames, mode='meanLR', exposure=None, sigma=0.2):
    """frames: list of uint8 RGB arrays of equal size (any order, >= 2). Returns uint8 RGB."""
    return render(*decompose(net_L, net_f, frames, mode, sigma), exposure)


@torch.no_grad()
def render(L, R, exposure=None):
    if exposure is not None:
        L = adjust_exposure(L, exposure)
    return np.uint8((L * R).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255)


@torch.no_grad()
def decompose(net_L, net_f, frames, mode='meanLR', sigma=0.2):
    """Returns the fused illumination L (1x1xHxW) and shared reflectance R (1x3xHxW)."""
    assert mode in MODES, mode
    device = next(net_L.parameters()).device
    frames = sort_by_brightness(frames)
    xs = [to_tensor(f, device) for f in frames]
    n = len(xs)

    if mode in ('pair2', 'meanR'):
        L = (net_L(xs[0]) + net_L(xs[-1])) / 2
    else:
        Ls = [net_L(x) for x in xs]
        if mode in ('meanL', 'meanLR'):
            L = sum(Ls) / n
        else:
            ws = [torch.exp(-((x.mean(1, keepdim=True) - 0.5) ** 2) / (2 * sigma ** 2)) + 1e-6 for x in xs]
            L = sum(w * l for w, l in zip(ws, Ls)) / sum(ws)

    if mode in ('pair2', 'meanL', 'wL'):
        R = net_f.SRE(xs[0], xs[-1])[0]
    else:
        pairs = [(k, n - 1 - k) for k in range(n // 2)] or [(0, n - 1)]
        R = sum(net_f.SRE(xs[i], xs[j])[0] for i, j in pairs) / len(pairs)
    return L, R
