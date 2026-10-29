"""Measure Retinex-MEF training throughput / memory on random 128x128 patches (same loss as train.py)."""
import sys
import time
import torch
import torch.nn.functional as F
from nets.net import L_net, fusionnet
from utils import illu_smooth

coef = (1, 0.5, 0.1, 0.2, 0.1)
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
amp = len(sys.argv) > 2 and sys.argv[2] == 'amp'
net_fusion, net_L = fusionnet().cuda(), L_net().cuda()
opt = torch.optim.Adam(list(net_fusion.parameters()) + list(net_L.parameters()), lr=1e-4)
scaler = torch.amp.GradScaler(enabled=amp)
img1, img2, img3 = (torch.rand(bs, 3, 128, 128, device='cuda') for _ in range(3))
torch.cuda.reset_peak_memory_stats()
for it in range(25):
    if it == 5:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        L1, L2, L3 = net_L(img1), net_L(img2), net_L(img3)
        y3, R3, Rhat = net_fusion(img1, img2, L3)
    y3, R3, Rhat, L1, L2, L3 = (t.float() for t in (y3, R3, Rhat, L1, L2, L3))
    loss = (coef[0] * F.l1_loss(y3, img3) + coef[1] * illu_smooth(L3, img3)
            + coef[2] * F.l1_loss(L3, img3.max(1, keepdim=True)[0])
            + coef[3] * torch.mean(F.relu(Rhat * L3.detach() - img3) + F.relu(Rhat * L2.detach() - img2) + F.relu(Rhat * L1.detach() - img1))
            + coef[4] * F.l1_loss(Rhat, R3))
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20
print(f'bs={bs} amp={amp}: {dt * 1000:.1f} ms/iter, {bs / dt:.1f} samples/s, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
