"""Vectorized Python ports of the MEFB (Zhang, Inf. Fusion 2021) MATLAB metrics in datasets/MEFB/metrics.

Conventions follow MEFB exactly: inputs are uint8 RGB (H, W, 3); per-channel metrics are averaged over R, G, B;
MEF-SSIM (Ma et al. TIP 2015, 3-scale MEF-MS-SSIM) is computed on MATLAB-style rgb2gray images.
All functions take a list of source images `srcs` (MEFB uses 2) and the fused image `f`.
"""
import numpy as np
from scipy import ndimage, signal

EPS = np.finfo(np.float64).eps
RGB2GRAY = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])  # MATLAB rgb2gray


def rgb2gray(img):
    return np.round(img.astype(np.float64) @ RGB2GRAY)


def _channels(img):
    img = img.astype(np.float64)
    return [img] if img.ndim == 2 else [img[..., k] for k in range(img.shape[2])]


def _per_channel(fn, srcs, f):
    chans = [_channels(s) for s in srcs]
    return float(np.mean([fn([c[k] for c in chans], fk) for k, fk in enumerate(_channels(f))]))


# ---------------------------------------------------------------- MEF-SSIM
def _gauss2d(n, sigma):
    ax = np.arange(n) - (n - 1) / 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _filt_valid(x, k1d, bd):
    """Separable correlation, cropped to MATLAB filter2(...,'valid')."""
    y = ndimage.correlate1d(ndimage.correlate1d(x, k1d, axis=0, mode='constant'), k1d, axis=1, mode='constant')
    return y[bd:-bd, bd:-bd]


def _mef_ssim_single(seq, fI, K=0.03, ws=11, sigma=1.5):
    bd = ws // 2
    box = np.full(ws, 1.0 / ws)
    g1 = np.exp(-((np.arange(ws) - bd) ** 2) / (2 * sigma ** 2))
    g1 /= g1.sum()
    n = len(seq)
    U = lambda x: _filt_valid(x, box, bd)
    G = lambda x: _filt_valid(x, g1, bd)

    mu = [U(x) for x in seq]
    sig_norm = [np.sqrt(np.maximum(ws * ws * (U(x * x) - m * m), 0)) for x, m in zip(seq, mu)]
    ed = np.stack([s + 0.001 for s in sig_norm])

    S = sum(seq)
    muS = U(S)
    num = np.sqrt(np.maximum(ws * ws * (U(S * S) - muS * muS), 0))
    R = (num + EPS) / (sum(sig_norm) + EPS)
    R[R > 1] = 1 - EPS
    R[R < 0] = EPS
    p = np.minimum(np.tan(np.pi / 2 * R), 10)

    w = (ed / ws) ** p[None] + EPS
    w /= w.sum(0, keepdims=True)
    max_ed = ed.max(0)
    c = w / ed  # rBlock = sum_k c_k * (X_k - mu_k)

    # quadratic forms sum_ij c_i c_j cov_ij, accumulated pair by pair (O(n) memory instead of O(n^2) maps)
    gm = [G(x) for x in seq]
    gmf = G(fI)
    uq = np.zeros_like(max_ed)
    gq = np.zeros_like(max_ed)
    for i in range(n):
        for j in range(i, n):
            xx = seq[i] * seq[j]
            cc = c[i] * c[j] * (1 if i == j else 2)
            uq += cc * (U(xx) - mu[i] * mu[j])
            gq += cc * (G(xx) - gm[i] * gm[j])
    rnorm = np.sqrt(np.maximum(ws * ws * uq, 0))
    s = np.where(rnorm > 0, max_ed / np.where(rnorm > 0, rnorm, 1), 1.0)

    sigma1_sq = s ** 2 * gq
    sigma2_sq = G(fI * fI) - gmf * gmf
    sigma12 = s * sum(c[k] * (G(seq[k] * fI) - gm[k] * gmf) for k in range(n))
    C = (K * 255) ** 2
    return float(np.mean((2 * sigma12 + C) / (sigma1_sq + sigma2_sq + C)))


def _down2(x):
    x = np.pad(x, ((0, 1), (0, 1)), mode='symmetric')
    y = (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:]) / 4
    return y[::2, ::2]


def mef_ssim(srcs, f, level=3, weight=(0.0448, 0.2856, 0.3001)):
    seq = [rgb2gray(s) if s.ndim == 3 else s.astype(np.float64) for s in srcs]
    fI = rgb2gray(f) if f.ndim == 3 else f.astype(np.float64)
    weight = np.array(weight) / np.sum(weight)
    Q = []
    for l in range(level):
        Q.append(_mef_ssim_single(seq, fI))
        if l < level - 1:
            seq, fI = [_down2(x) for x in seq], _down2(fI)
    return float(np.prod(np.array(Q) ** weight))


# ---------------------------------------------------------------- no-reference statistics
def _entropy(_, f):
    h = np.bincount(np.floor(f).astype(np.int64).ravel(), minlength=256) / f.size
    h = h[h > 0]
    return -np.sum(h * np.log2(h))


def _sd(_, f):
    return np.sqrt(np.mean((f - f.mean()) ** 2))


def _sf(_, f):
    rf = np.sum(np.diff(f, axis=1) ** 2) / f.size
    cf = np.sum(np.diff(f, axis=0) ** 2) / f.size
    return np.sqrt(rf + cf)


def _ag(_, f):
    gy, gx = np.gradient(f)
    return np.sum(np.sqrt((gx ** 2 + gy ** 2) / 2)) / ((f.shape[0] - 1) * (f.shape[1] - 1))


def entropy(srcs, f): return _per_channel(_entropy, srcs, f)
def sd(srcs, f): return _per_channel(_sd, srcs, f)
def sf(srcs, f): return _per_channel(_sf, srcs, f)
def ag(srcs, f): return _per_channel(_ag, srcs, f)


# ---------------------------------------------------------------- Qabf (Xydeas & Petrovic)
_H1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
_H3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def _sobel(x):
    sx = signal.convolve2d(x, _H3, mode='same')
    sy = signal.convolve2d(x, _H1, mode='same')
    a = np.full(x.shape, np.pi / 2)
    nz = sx != 0
    a[nz] = np.arctan(sy[nz] / sx[nz])
    return np.sqrt(sx ** 2 + sy ** 2), a


def _qabf(srcs, f, Tg=0.9994, kg=-15, Dg=0.5, Ta=0.9879, ka=-22, Da=0.8):
    gF, aF = _sobel(f)
    nume = deno = 0.0
    for s in srcs:
        gA, aA = _sobel(s)
        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.where(gA > gF, gF / gA, np.where(gA == gF, gF, gA / gF))
        A = 1 - np.abs(aA - aF) / (np.pi / 2)
        Q = Tg / (1 + np.exp(kg * (G - Dg))) * Ta / (1 + np.exp(ka * (A - Da)))
        nume += np.sum(Q * gA)
        deno += np.sum(gA)
    return nume / deno


def qabf(srcs, f): return _per_channel(_qabf, srcs, f)


# ---------------------------------------------------------------- NMI (Hossny)
def _normalize1(x):
    lo, hi = x.min(), x.max()
    if hi == 0 and lo == 0:
        return x
    return np.round((x - lo) / (hi - lo) * 255) if hi > lo else np.full_like(x, np.nan)


def _mutual_info(a, b):
    h = np.bincount((a.astype(np.int64) * 256 + b.astype(np.int64)).ravel(), minlength=65536).reshape(256, 256)
    h = h / h.sum()
    ent = lambda p: -np.sum(p[p > 0] * np.log2(p[p > 0]))
    hx, hy = ent(h.sum(1)), ent(h.sum(0))
    return hx + hy - ent(h), hx, hy


def _nmi(srcs, f):
    f = _normalize1(f)
    total = 0.0
    for s in srcs:
        mi, hx, hf = _mutual_info(_normalize1(s), f)
        total += mi / (hf + hx)
    return 2 * total


def nmi(srcs, f): return _per_channel(_nmi, srcs, f)


# ---------------------------------------------------------------- VIF (Han et al. 2013)
def _vid_vind(ref, dist, sq):
    out = []
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = _gauss2d(N, N / 5)
        fv = lambda x: signal.correlate(x, win, mode='valid', method='fft' if x.size > 40000 else 'direct')
        if scale > 1:
            ref, dist = fv(ref)[::2, ::2], fv(dist)[::2, ::2]
        mu1, mu2 = fv(ref), fv(dist)
        s1 = np.maximum(fv(ref * ref) - mu1 * mu1, 0)
        s2 = np.maximum(fv(dist * dist) - mu2 * mu2, 0)
        s12 = fv(ref * dist) - mu1 * mu2
        g = s12 / (s1 + 1e-10)
        sv = s2 - g * s12
        m = s1 < 1e-10
        g[m], sv[m], s1[m] = 0, s2[m], 0
        m = s2 < 1e-10
        g[m], sv[m] = 0, 0
        m = g < 0
        sv[m], g[m] = s2[m], 0
        sv[sv <= 1e-10] = 1e-10
        out.append((np.log10(1 + g ** 2 * s1 / (sv + sq)), np.log10(1 + s1 / sq), g))
    return out


def _vif(srcs, f, C=1e-7):
    sq = 0.005 * 255 * 255
    p = np.array([1, 0, 0.15, 1]) / 2.15
    a, b = (_vid_vind(s, f, sq) for s in srcs[:2])
    F = []
    for (n1, d1, g1), (n2, d2, g2) in zip(a, b):
        L = g1 < g2
        F.append(np.sum(np.where(L, n1, n2) + C) / np.sum(np.where(L, d1, d2) + C))
    return float(np.sum(np.array(F) * p))


def vif(srcs, f): return _per_channel(_vif, srcs, f)


# ---------------------------------------------------------------- PSNR (MEFB definition)
def _psnr(srcs, f):
    mse = np.mean([np.sqrt(np.sum((s - f) ** 2)) / f.size for s in srcs])
    return 20 * np.log10(255 / np.sqrt(mse))


def psnr(srcs, f): return _per_channel(_psnr, srcs, f)


# ---------------------------------------------------------------- Qncie (Wang et al.)
def _ncc(a, b):
    mi, hx, hy = _mutual_info(a, b)
    return mi / 8.0  # (H_x + H_y - H_xy) / log2(256)


def _qncie(srcs, f):
    a, b, f = _normalize1(srcs[0]), _normalize1(srcs[1]), _normalize1(f)
    nxy, nxf, nyf = _ncc(a, b), _ncc(a, f), _ncc(b, f)
    r = np.linalg.eigvals(np.array([[1, nxy, nxf], [nxy, 1, nyf], [nxf, nyf, 1]])).astype(np.complex128)
    hr = -np.sum(r * np.log2(r / 3) / 3).real / 8.0
    return 1 - hr


def qncie(srcs, f): return _per_channel(_qncie, srcs, f)


# ---------------------------------------------------------------- Qcb (Chen & Blum)
_G31 = lambda s: np.exp(-(np.arange(-15, 16)[:, None] ** 2 + np.arange(-15, 16)[None] ** 2) / (2 * s * s)) / (2 * np.pi * s * s)


def _qcb(srcs, f, f0=15.3870, f1=1.3456, a=0.7622, p=3, q=2, Z=1e-4):
    ims = [_normalize1(x / 255.0) for x in (*srcs[:2], f)]
    m, n = f.shape
    fx = (np.arange(n) - n // 2) * (2 / n) * (n / 30)
    fy = (np.arange(m) - m // 2) * (2 / m) * (m / 30)
    r = np.sqrt(fx[None] ** 2 + fy[:, None] ** 2)
    Sd = np.exp(-(r / f0) ** 2) - a * np.exp(-(r / f1) ** 2)
    G1, G2 = _G31(2), _G31(4)
    CP = []
    for im in ims:
        x = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im)) * Sd))
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.abs(signal.fftconvolve(x, G1, mode='same') / signal.fftconvolve(x, G2, mode='same') - 1)
        CP.append(C ** p / (C ** q + Z))
    c1, c2, cf = CP
    with np.errstate(divide='ignore', invalid='ignore'):
        q1 = np.where(c1 < cf, c1 / cf, cf / c1)
        q2 = np.where(c2 < cf, c2 / cf, cf / c2)
        den = c1 * c1 + c2 * c2
        Q = (c1 * c1 * q1 + c2 * c2 * q2) / den
    return float(np.mean(Q))


def qcb(srcs, f): return _per_channel(_qcb, srcs, f)


# ---------------------------------------------------------------- CC / SSIM (source-averaged, gray)
def cc(srcs, f):
    g = rgb2gray(f).ravel() if f.ndim == 3 else f.astype(np.float64).ravel()
    return float(np.mean([np.corrcoef((rgb2gray(s) if s.ndim == 3 else s.astype(np.float64)).ravel(), g)[0, 1] for s in srcs]))


def ssim(srcs, f):
    from skimage.metrics import structural_similarity
    g = rgb2gray(f) if f.ndim == 3 else f.astype(np.float64)
    return float(np.mean([structural_similarity(rgb2gray(s) if s.ndim == 3 else s.astype(np.float64), g,
                                                data_range=255, gaussian_weights=True, sigma=1.5,
                                                use_sample_covariance=False) for s in srcs]))


METRICS = {
    'MEF-SSIM': mef_ssim, 'Qabf': qabf, 'VIF': vif, 'NMI': nmi, 'Qcb': qcb, 'Qncie': qncie,
    'EN': entropy, 'SD': sd, 'SF': sf, 'AG': ag, 'PSNR': psnr, 'CC': cc, 'SSIM': ssim,
}
# index of each metric in MEFB's evaluationMetrics.mat (see datasets/MEFB/output/information.txt)
MEFB_MAT_INDEX = {'EN': 1, 'NMI': 3, 'PSNR': 4, 'Qncie': 5, 'AG': 7, 'Qabf': 9, 'SD': 11, 'SF': 12,
                  'MEF-SSIM': 16, 'Qcb': 17, 'VIF': 19}


def compute_all(srcs, f, names=None):
    return {k: METRICS[k](srcs, f) for k in (names or METRICS)}
