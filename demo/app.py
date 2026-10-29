"""Interactive multi-exposure fusion demo (Retinex-MEF + N-frame extension + exposure control).

  .\.venv\Scripts\python.exe demo\app.py            -> http://127.0.0.1:7860
Upload 2..N differently exposed shots of a static scene (any order; phone JPGs are fine), pick a fusion mode,
then drag the exposure slider - only the cheap illumination curve is recomputed, the network runs once per upload.
"""
import os
import sys
import json
import time
import tempfile

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'innovation'), os.path.join(ROOT, 'tools')]
from nframe import MODES, DEFAULT_CKPT, OFFICIAL_CKPT, load_model, decompose, render, sort_by_brightness
from mef_io import list_mefb_pairs

MODE_HELP = {
    'pair2': 'official: darkest + brightest only',
    'meanL': 'all-frame mean illumination',
    'wL': 'well-exposedness weighted illumination',
    'meanR': 'reflectance averaged over symmetric pairs',
    'meanLR': 'meanL + meanR (recommended for N>2)',
    'wLR': 'wL + meanR',
}
MODELS = {'self-trained (ours)': DEFAULT_CKPT, 'official': OFFICIAL_CKPT}
_models, _cache = {}, {}


def get_model(name):
    if name not in _models:
        _models[name] = load_model(MODELS[name])
    return _models[name]


def read_frames(paths, long_side, align):
    imgs = [np.asarray(ImageOps.exif_transpose(Image.open(p)).convert('RGB')) for p in paths]
    h, w = imgs[0].shape[:2]
    s = min(1.0, long_side / max(h, w))
    size = (round(w * s), round(h * s))
    imgs = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) if im.shape[1::-1] != size else im for im in imgs]
    if align and len(imgs) > 1:
        cv2.createAlignMTB().process(imgs, imgs)
    return sort_by_brightness(imgs)


def mertens(frames):
    return np.uint8(np.clip(cv2.createMergeMertens().process(frames) * 255, 0, 255) + 0.5)


def save_png(img, tag):
    path = os.path.join(tempfile.gettempdir(), f'mef_{tag}_{int(time.time() * 1000)}.png')
    Image.fromarray(img).save(path)
    return path


def run(files, model, mode, use_exposure, exposure, long_side, align, show_mertens):
    if not files or len(files) < 2:
        raise gr.Error('Please upload at least 2 images of the same scene with different exposures.')
    paths = [f if isinstance(f, str) else f.name for f in files]
    key = (tuple((p, os.path.getmtime(p)) for p in paths), model, mode, int(long_side), bool(align))
    t0 = time.perf_counter()
    if key not in _cache:
        _cache.clear()
        frames = read_frames(paths, long_side, align)
        L, R = decompose(*get_model(model), frames, mode)
        _cache[key] = (frames, L, R, mertens(frames) if show_mertens else None)
    frames, L, R, mer = _cache[key]
    if show_mertens and mer is None:
        mer = mertens(frames)
        _cache[key] = (frames, L, R, mer)
    out = render(L, R, exposure if use_exposure else None)
    dt = time.perf_counter() - t0
    info = (f'{len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, mode={mode} ({MODE_HELP[mode]}), '
            f'mean brightness of inputs: {", ".join(f"{f.mean() / 255:.2f}" for f in frames)}; '
            f'output mean {out.mean() / 255:.2f}; {dt:.2f}s')
    return ([(f, f'#{i + 1}  mean {f.mean() / 255:.2f}') for i, f in enumerate(frames)], out,
            save_png(out, 'fused'), mer if show_mertens else None, info)


def examples():
    ex = []
    pairs = {s: (a, b) for s, a, b in list_mefb_pairs()}
    for s in ('Lighthouse', 'Arno', 'garage'):
        if s in pairs:
            ex.append([list(pairs[s])])
    sorted_dir = os.path.join(ROOT, 'datasets', 'SICE', 'sorted_L1024')
    split = os.path.join(ROOT, 'datasets', 'SICE', 'split.json')
    if os.path.exists(split):
        for s in json.load(open(split))['test'][:3]:
            d = os.path.join(sorted_dir, s)
            ex.append([[os.path.join(d, f) for f in sorted(os.listdir(d), key=lambda x: int(x.split('.')[0]))]])
    return ex


with gr.Blocks(title='Retinex-MEF N-frame Fusion Demo') as demo:
    gr.Markdown('## Multi-exposure fusion demo: Retinex-MEF (ICCV 2025) + N-frame extension + controllable exposure\n'
                'Upload 2..N shots of a static scene (tripod / bracketing). The network runs once per upload; '
                'the exposure slider only re-maps the fused illumination (paper Eq. 15-17).')
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(file_count='multiple', file_types=['image'], label='Exposure sequence (2..N images)')
            model = gr.Dropdown(list(MODELS), value='self-trained (ours)', label='Weights')
            mode = gr.Dropdown(MODES, value='meanLR', label='N-frame fusion mode',
                               info='; '.join(f'{k}: {v}' for k, v in MODE_HELP.items()))
            use_exp = gr.Checkbox(False, label='Exposure control (otherwise plain average illumination)')
            exposure = gr.Slider(0.2, 0.8, value=0.5, step=0.01, label='Target mean exposure E')
            long_side = gr.Slider(512, 2048, value=1280, step=64, label='Max long side (px)')
            align = gr.Checkbox(True, label='MTB alignment (hand-held shots)')
            show_mer = gr.Checkbox(True, label='Also show Mertens (OpenCV) for comparison')
            btn = gr.Button('Fuse', variant='primary')
        with gr.Column(scale=2):
            gallery = gr.Gallery(label='Inputs (sorted dark -> bright)', columns=6, height=160)
            out = gr.Image(label='Retinex-MEF fused', type='numpy', format='png')
            dl = gr.File(label='Download fused PNG')
            mer = gr.Image(label='Mertens exposure fusion', type='numpy')
            info = gr.Textbox(label='Info', lines=2)
    inputs = [files, model, mode, use_exp, exposure, long_side, align, show_mer]
    outputs = [gallery, out, dl, mer, info]
    btn.click(run, inputs, outputs)
    for c in (model, mode, use_exp, align, show_mer):
        c.change(run, inputs, outputs)
    exposure.release(run, inputs, outputs)
    long_side.release(run, inputs, outputs)
    gr.Examples(examples(), [files], label='Examples (MEFB pairs, held-out SICE sequences)')

if __name__ == '__main__':
    demo.queue().launch(server_name='127.0.0.1', server_port=7860, inbrowser=False,
                        allowed_paths=[os.path.join(ROOT, 'datasets')])
