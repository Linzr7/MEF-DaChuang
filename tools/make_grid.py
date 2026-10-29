import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mef_io import ROOT, imread_rgb, imwrite_rgb, list_mefb_pairs

METHODS = ['mertens', 'mefnet', 'transmef', 'retinex_mef']


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['Arno', 'Lighthouse', 'Sky', 'garage'])
    p.add_argument('--methods', nargs='+', default=METHODS)
    p.add_argument('--width', type=int, default=400)
    p.add_argument('--out', default=os.path.join(ROOT, 'docs', 'figures', 'mefb_compare.jpg'))
    args = p.parse_args()

    pairs = {s: (a, b) for s, a, b in list_mefb_pairs()}
    rows = []
    for s in args.scenes:
        a, b = pairs[s]
        tiles = [('under', imread_rgb(a)), ('over', imread_rgb(b))]
        tiles += [(m, imread_rgb(os.path.join(ROOT, 'results', m, 'MEFB', s + '.png'))) for m in args.methods]
        h = int(tiles[0][1].shape[0] * args.width / tiles[0][1].shape[1])
        rows.append(np.hstack([label(cv2.resize(t, (args.width, h), interpolation=cv2.INTER_AREA), n) for n, t in tiles]))
    imwrite_rgb(args.out, np.vstack(rows))
    print('saved', args.out)


if __name__ == '__main__':
    main()
