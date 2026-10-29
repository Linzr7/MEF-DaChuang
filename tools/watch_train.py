"""Live terminal monitor for Retinex-MEF training (Ctrl+C to quit).

  .\.venv\Scripts\python.exe tools\watch_train.py            # newest exp + newest Retinex-MEF\train_run*.log
  .\.venv\Scripts\python.exe tools\watch_train.py --interval 10
"""
import os
import re
import csv
import glob
import json
import time
import argparse
import datetime
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RMEF = os.path.join(ROOT, 'Retinex-MEF')
OFFICIAL_VAL = (0.8710, 16.61)  # official ckpt on the same 10 held-out SICE scenes
LINE = re.compile(r'\[Epoch (\d+)/(\d+)\] \[Batch (\d+)/(\d+)\] (.*) ETA (\S+)')


def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def last_progress(log):
    with open(log, 'rb') as f:
        f.seek(max(0, os.path.getsize(log) - 20000))
        lines = f.read().decode('utf-8', 'ignore').splitlines()
    for l in reversed(lines):
        m = LINE.search(l)
        if m:
            return m, lines[-1]
    return None, lines[-1] if lines else ''


def gpu():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        u, mu, mt, t = [x.strip() for x in out.strip().split(',')]
        return f'{u}% util, {mu}/{mt} MiB, {t} C'
    except Exception as e:
        return f'n/a ({e})'


def bar(frac, width=40):
    n = int(frac * width)
    return '[' + '#' * n + '-' * (width - n) + f'] {frac * 100:5.1f}%'


def render(exp, log):
    out = [f'Retinex-MEF training monitor   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}',
           f'exp: {os.path.relpath(exp, ROOT)}   log: {os.path.relpath(log, ROOT)}', '']
    m, last = last_progress(log)
    age = time.time() - os.path.getmtime(log)
    if m:
        ep, n_ep, b, n_b = map(int, m.groups()[:4])
        total = ((ep - 1) * n_b + b) / (n_ep * n_b)
        out += [f'overall  {bar(total)}   epoch {ep}/{n_ep}',
                f'epoch    {bar(b / n_b)}   batch {b}/{n_b}',
                f'ETA      {m.group(6)}  (finish ~ {eta_clock(m.group(6))})',
                f'losses   {m.group(5)}']
    else:
        out.append(f'(no progress line yet) {last[-150:]}')
    finished = m is not None and m.group(1) == m.group(2) and m.group(3) == m.group(4)
    status = ('FINISHED' if finished and age > 60 else 'RUNNING' if age < 300 else
              f'!! log not updated for {age / 60:.0f} min - training may have stopped (see --resume in AGENTS.md)')
    out += [f'status   {status}', f'GPU      {gpu()}', '']

    log_csv = os.path.join(exp, 'train_log.csv')
    rows = list(csv.DictReader(open(log_csv))) if os.path.exists(log_csv) else []
    out.append(f'{"epoch":>5} {"lr":>8} {"total":>8} {"recon":>8} {"val MEF-SSIM":>13} {"val PSNR":>9} {"min":>6}')
    for r in rows[-12:]:
        out.append(f'{r["epoch"]:>5} {float(r["lr"]):8.1e} {float(r["total"]):8.4f} {float(r["recon"]):8.4f} '
                   f'{float(r["val_mef_ssim"]):13.4f} {float(r["val_psnr_gt"]):9.2f} {r["epoch_minutes"]:>6}')
    if rows:
        best = max(rows, key=lambda r: float(r['val_mef_ssim']))
        out.append(f'best val MEF-SSIM {float(best["val_mef_ssim"]):.4f} @ epoch {best["epoch"]}   '
                   f'(official ckpt: {OFFICIAL_VAL[0]:.4f} / {OFFICIAL_VAL[1]:.2f} dB)')
    ckpts = glob.glob(os.path.join(exp, 'model', 'ckpt_*.pth'))
    out.append(f'checkpoints saved: {len(ckpts)}')
    return '\n'.join(out)


def eta_clock(s):
    try:
        if 'day' in s:
            return s
        h, mnt, sec = map(int, s.split(':'))
        return (datetime.datetime.now() + datetime.timedelta(hours=h, minutes=mnt, seconds=sec)).strftime('%m-%d %H:%M')
    except ValueError:
        return '?'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default=None)
    p.add_argument('--log', default=None)
    p.add_argument('--interval', type=int, default=30)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    while True:
        exp = a.exp or newest(os.path.join(RMEF, 'exp', '*'))
        log = a.log or newest(os.path.join(RMEF, 'train_run*.log'))
        text = render(exp, log) if exp and log else 'no experiment / log found'
        if a.once:
            print(text)
            return
        os.system('cls' if os.name == 'nt' else 'clear')
        print(text + f'\n\nrefresh every {a.interval}s, Ctrl+C to quit')
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
