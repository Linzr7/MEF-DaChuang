# MEF-DaChuang：基于 Retinex-MEF 的多曝光图像融合（大创项目）

以 **Retinex-MEF (ICCV 2025)** 为基座，在单张 RTX 4070 Laptop (8GB) 上完成：

1. **复现**：官方权重推理 + 在 SICE 上从零自训练（bf16，batch 8），自训练权重在 MEFB 上 MEF-SSIM 略优于官方权重；
2. **基准评测**：把 MEFB 的 MATLAB 指标完整移植到 Python（与官方 `.mat` 逐图校验，相对误差 ~1e-5），对比 25 种方法；
3. **创新①：N 帧输入**（免训练）：把只支持欠曝/过曝两帧的 Retinex-MEF 扩展到任意 N 帧；
4. **创新②：交互式 Demo**（Gradio）：上传多帧 → 选择融合策略 → 滑块实时调曝光 → 下载。

> 本仓库为**私有仓库**，包含第三方代码（Retinex-MEF / MEF-Net / TransMEF，均无开源许可证），仅供项目组内部学习使用，请勿公开。

---

## 目录结构

```
├── Retinex-MEF/          基座（官方代码 + 我们新增的脚本）
│   ├── nets/ utils.py test.py train.py       官方原文件（未改动）
│   ├── model/ckpt.pth                        官方预训练权重
│   ├── train_local.py                        8GB 显卡训练脚本（bf16、断点续训、CSV 日志、留出集验证）
│   ├── run_retinex_mefb.py                   MEFB 批量推理（--ckpt / --exposure）
│   ├── bench_train.py                        训练吞吐/显存测试
│   ├── data/prepare_sice.py                  SICE 预处理（排序+降采样+切块+留出划分）
│   └── exp/10_18_13_59_b8_bf16/              我们的训练记录：model/ckpt_40.pth（自训练权重）、train_log.csv
├── MEFNet-master/        baseline，已移植到 torch 2.x；run_mefnet_mefb.py
├── TransMEF-main/        baseline；run_transmef_mefb.py（权重需另下，见下）
├── baselines/            Mertens（OpenCV）
├── eval/                 指标：mef_metrics.py（13 项指标）、eval_mefb.py、validate_metrics.py
├── innovation/           创新①：nframe.py（N 帧融合）、run_nframe_sice.py（消融实验）
├── demo/app.py           创新②：Gradio 交互 Demo
├── tools/                mef_io.py（中文路径安全读写）、make_grid.py、plot_train_log.py、watch_train.py
├── results/metrics/      所有指标结果（CSV + Markdown 汇总表）
├── docs/figures/         对比图、训练曲线
└── AGENTS.md             详细命令速查 & 踩坑记录
```

`datasets/`、`results/` 下的图片、`.venv/` 不入库，按下面步骤自行准备。

---

## 1. 环境

Windows + Python 3.10 + CUDA 12.x 显卡（Linux 同理）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # 应为 True
```

> 路径含中文时 `cv2.imread` 会失败，项目里统一用 `tools/mef_io.py` 的 `imread_rgb / imwrite_rgb`。

## 2. 数据准备（都放在 `datasets/` 下）

| 数据 | 获取方式 | 放置位置 |
|---|---|---|
| MEFB 测试集（100 对） | `git clone --depth 1 --filter=blob:none --sparse https://github.com/xingchenzhang/MEFB.git datasets/MEFB` 然后 `cd datasets/MEFB; git sparse-checkout set input metrics util output/evaluation_metrics` | `datasets/MEFB/input/` |
| MEFB 21 种方法官方融合结果（可选，用于 25 方法对比 & 指标校验） | `gdown 1qB3UwFDWWe1Uq5L7aJPj2Hzkdat94Mu0`，解压 `fused_images.zip` | `datasets/MEFB/output/fused_images/` |
| SICE Part1（360 序列，~3.9GB） | [Google Drive](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view) / [百度网盘](https://pan.baidu.com/s/1kXotehL) | `datasets/SICE/Dataset_Part1/` |
| TransMEF 权重（76MB，可选） | [Google Drive](https://drive.google.com/file/d/1a-i_M7i-rns9pyu-PxkOKuL3RWoza8em/view) / [百度网盘](https://pan.baidu.com/s/1PDUkL_z6DLnHa6mIQy-HPA?pwd=jcx3) | `TransMEF-main/best_model.pth` |

SICE 预处理（长边缩到 1024、按亮度排序、切 128×128 块、留出 30 个测试场景）：

```powershell
.\.venv\Scripts\python.exe Retinex-MEF\data\prepare_sice.py      # → datasets/SICE/{sorted_L1024, split.json, patches_L1024_35k}
```

## 3. 常用命令（在仓库根目录执行）

```powershell
# 交互 Demo（不需要任何数据集，自带权重即可用）→ http://127.0.0.1:7860
.\.venv\Scripts\python.exe demo\app.py

# MEFB 推理
.\.venv\Scripts\python.exe Retinex-MEF\run_retinex_mefb.py                                   # 官方权重
.\.venv\Scripts\python.exe Retinex-MEF\run_retinex_mefb.py --ckpt Retinex-MEF\exp\10_18_13_59_b8_bf16\model\ckpt_40.pth --output results\retinex_mef_self\MEFB
.\.venv\Scripts\python.exe baselines\run_mertens_mefb.py
.\.venv\Scripts\python.exe MEFNet-master\run_mefnet_mefb.py
.\.venv\Scripts\python.exe TransMEF-main\run_transmef_mefb.py

# 指标（MEFB 指标 Python 版）
.\.venv\Scripts\python.exe eval\eval_mefb.py --workers 4 --methods mertens mefnet transmef retinex_mef retinex_mef_self --mefb-official GFF MEFNet U2Fusion
.\.venv\Scripts\python.exe eval\validate_metrics.py --methods GFF MEFNet        # 与官方 MATLAB 结果对比

# 训练（约 15 小时 / 40 epoch @ RTX 4070 Laptop）
cd Retinex-MEF
..\.venv\Scripts\python.exe -u train_local.py --data ..\datasets\SICE\patches_L1024_35k
..\.venv\Scripts\python.exe -u train_local.py --resume exp\<name>\state_last.pth    # 断点续训
cd ..
.\.venv\Scripts\python.exe tools\watch_train.py                                      # 实时进度监控

# 创新① N 帧消融（SICE 留出 30 场景）
.\.venv\Scripts\python.exe innovation\run_nframe_sice.py --workers 4
```

> ⚠️ 16GB 内存的机器上指标评测 `--workers` 不要超过 4（曾因进程过多导致蓝屏）。

## 4. 主要结果

### 4.1 复现：MEFB 100 对（完整 25 方法 × 13 指标表见 `results/metrics/MEFB_summary.md`）

| 方法 | MEF-SSIM | Qabf | VIF | Qcb | NMI | SD |
|---|---|---|---|---|---|---|
| Mertens (OpenCV) | **0.9767** | **0.7370** | 0.7746 | 0.4315 | 0.7397 | 56.96 |
| MEF-Net（移植 torch2） | 0.9155 | 0.7172 | **0.8597** | **0.4952** | 0.6493 | 55.83 |
| TransMEF | 0.9270 | 0.5761 | 0.5373 | 0.3984 | **0.9456** | 50.32 |
| Retinex-MEF 官方权重 | 0.8917 | 0.4680 | 0.4603 | 0.4433 | 0.7427 | 44.48 |
| **Retinex-MEF 自训练（ours）** | 0.9001 | 0.4805 | 0.4905 | 0.4593 | 0.7342 | 45.33 |

- 自训练权重 MEF-SSIM 比官方高 0.96%（100 张中 91 张更好），复现成功。
- 论文只在 30 对 MEFB 上用 Qcb/NMI/Qncie/SSIM/PSNR/CC 报告结果；在完整 100 对、传统梯度/结构类指标下 Retinex-MEF 并不占优，它的优势在去眩光、色彩保真和曝光可控。
- 与论文配置差异：官方代码为 40 epoch、StepLR(10, 0.1)，论文写 80 epoch、每 20 epoch 减半；我们按代码跑，另因显存用 bf16 autocast，训练集为 SICE Part1 的 330 个场景（34,848 个 patch）。

训练曲线：`docs/figures/train_curves_10_18_13_59_b8_bf16.png`

### 4.2 创新①：N 帧融合（SICE 留出 30 场景，每场景 4~19 帧，自训练权重）

Retinex-MEF 推理时输出 `F = L̂ · R̂`，其中 `R̂ = SRE(I_欠曝, I_过曝)`。我们免训练地扩展到 N 帧：

| 策略 | 说明 | MEF-SSIM（对全部帧） | PSNR（对 GT） | SSIM（对 GT） |
|---|---|---|---|---|
| pair2 | 官方：只用最暗+最亮两帧 | 0.8248 | 16.44 | 0.8316 |
| meanR | 反射率在对称帧对 (k, N+1−k) 上平均 | 0.8393 | 16.65 | **0.8592** |
| meanLR | 全部帧照度平均 + meanR | **0.8749** | **17.69** | 0.8195 |
| wL | 按 well-exposedness 加权照度（反例：整体偏暗） | 0.6880 | 16.17 | 0.6954 |

- 端帧极端（全黑/全白）时 2 帧版本信息丢失严重，N 帧版本可恢复（见 `docs/figures/nframe_compare.jpg` 第三行）。
- 完整消融（含官方权重、Mertens 2 帧/N 帧参照）见 `results/metrics/SICE_nframe_summary.md`。

## 5. 后续计划

- [ ] 手机实拍多曝光数据集（Open Camera 曝光包围，`datasets/ours_phone/<场景>/`）→ Demo 与泛化测试
- [ ] 夜景噪声感知（反射率分支去噪）
- [ ] 轻量化（MEFLUT 式 1D LUT 蒸馏）→ AX7020 / Ti60 FPGA 实时融合

## 致谢 / 引用

- Retinex-MEF: <https://github.com/HaowenBai/Retinex-MEF> （Bai et al., ICCV 2025）
- MEF-Net: Ma et al., TIP 2020；TransMEF: <https://github.com/miccaiif/TransMEF> （Qu et al., AAAI 2022）
- MEFB: <https://github.com/xingchenzhang/MEFB> （Zhang, Information Fusion 2021）
- SICE: <https://github.com/csjcai/SICE> （Cai et al., TIP 2018）
