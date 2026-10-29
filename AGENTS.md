# 大创 MEF 项目 — 运行说明

- Shell 为 PowerShell；Python 环境：`D:\大创\.venv`（Python 3.10, torch 2.6.0+cu124, RTX 4070 Laptop 8GB）。
- PowerShell 的 `>`/`*>` 重定向会写成 UTF-16；需要普通文本/二进制时用 `cmd /c "... > file 2>&1"`。
- 路径含中文：读写图片统一用 `tools/mef_io.py` 的 `imread_rgb/imwrite_rgb`（`cv2.imdecode/np.fromfile`），不要直接 `cv2.imread` 绝对路径。
- MEFB：`datasets/MEFB/input/<scene>/<scene>_A|_B.*`，A=欠曝，B=过曝（大小写不一致，用 `list_mefb_pairs()`）。
  官方 21 种方法融合结果在 `datasets/MEFB/output/fused_images/<scene>_<method>.tif`，官方 MATLAB 指标在 `output/evaluation_metrics/*.mat`。
- SICE Part1：`datasets/SICE/Dataset_Part1/<1..360>/*.JPG`（每场景 3~28 帧），GT 在 `Dataset_Part1/Label/<scene>.JPG`。
  预处理产物：`sorted_L1024/`（按亮度排序+长边 1024）、`sorted_L1024_label/`、`split.json`（330 训练 / 30 留出测试）、`patches_L1024_35k/`（34848 个训练样本）。
- 结果目录：`results/<method>/<dataset>/<scene>.png`，每个目录附 `_runtime.csv`；指标在 `results/metrics/`。
- Retinex-MEF 训练显存：fp32 batch 8 ≈ 7.7GB 会溢出到共享内存（极慢）；用 bf16 autocast batch 8 ≈ 4–5GB，~27 samples/s。

## 命令（在 D:\大创 下执行）
- Retinex-MEF 官方样例：`cd Retinex-MEF; ..\.venv\Scripts\python.exe test.py`
- Retinex-MEF 自训练（train_local.py，每 epoch 存 ckpt + state_last 断点）：`cd Retinex-MEF; ..\.venv\Scripts\python.exe train_local.py --data ..\datasets\SICE\patches_L1024_35k`
- 断点续训：`..\.venv\Scripts\python.exe train_local.py --resume exp\<exp名>\state_last.pth [--workers 2]`（日志重定向到 `train_runN.log`；建议用 `Start-Process -WindowStyle Hidden` 脱离终端启动，避免关窗口杀进程）
- 训练监视：`..\.venv\Scripts\python.exe tools\watch_train.py`（自动找最新 exp + 最新 train_run*.log）
- Retinex-MEF @MEFB：`.\.venv\Scripts\python.exe Retinex-MEF\run_retinex_mefb.py [--ckpt <pth>] [--output <dir>] [--exposure 0.6]`
- Mertens @MEFB：`.\.venv\Scripts\python.exe baselines\run_mertens_mefb.py`
- MEF-Net @MEFB（已移植 torch2）：`.\.venv\Scripts\python.exe MEFNet-master\run_mefnet_mefb.py`
- TransMEF @MEFB（256 块滑窗 Y + CbCr 融合）：`.\.venv\Scripts\python.exe TransMEF-main\run_transmef_mefb.py`
- 对比图：`.\.venv\Scripts\python.exe tools\make_grid.py --scenes Arno Sky --width 320` → `docs/figures/`
- SICE 预处理：`.\.venv\Scripts\python.exe Retinex-MEF\data\prepare_sice.py [--long-side 1024 --max-samples 35000]`
- 训练：`cd Retinex-MEF; ..\.venv\Scripts\python.exe -u train_local.py --data ..\datasets\SICE\patches_L1024_35k`
  进度监控：`.\.venv\Scripts\python.exe tools\watch_train.py`（每 30s 刷新，`--once` 只看一次）
  续训：`--resume exp\<name>\state_last.pth`；日志 `exp\<name>\train_log.csv`；曲线 `python tools\plot_train_log.py Retinex-MEF\exp\<name> --official-val 0.8710 16.61`
- 指标（MEFB MATLAB 指标的 Python 移植，已与官方 .mat 逐图校验，误差 ~1e-5）：
  `.\.venv\Scripts\python.exe eval\eval_mefb.py --methods <results 下的方法名...> [--mefb-official GFF MEFNet ...] [--dir name=folder]`
  → `results/metrics/MEFB_<method>.csv`、`MEFB_summary.{csv,md}`；校验：`eval\validate_metrics.py --methods GFF MEFNet`
- 自训练权重：`Retinex-MEF/exp/10_18_13_59_b8_bf16/model/ckpt_40.pth`（MEFB 结果 `results/retinex_mef_self/`）
- 创新① N 帧消融（SICE 留出 30 场景）：`.\.venv\Scripts\python.exe innovation\run_nframe_sice.py --workers 4`
  → `results/nframe/`、`results/metrics/SICE_nframe_summary.md`。融合函数在 `innovation/nframe.py`（`fuse/decompose/render`）。
- 创新② Demo：`.\.venv\Scripts\python.exe demo\app.py` → http://127.0.0.1:7860
- 内存注意：16GB RAM；多进程指标评测 `--workers` 不要超过 4（曾在 14 进程 + 大 N 序列时蓝屏 0x4E）。
  控制台编码可能是 cp1252，打印中文路径前设 `$env:PYTHONIOENCODING='utf-8'`。
