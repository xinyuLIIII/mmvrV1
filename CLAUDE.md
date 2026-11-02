# CLAUDE.md

本文件帮助 Claude Code 快速理解本仓库。原型论文为 `2501.13805v2.pdf`，
本仓库在其基础上重写了工程实现，并对 **stage2（时序解码器）做了改进**。

## 项目概述

基于 **毫米波雷达 + IMU** 的 **3D 手部关键点估计 + 手势分类** 多任务模型（DETR 风格的集合预测）。

- 输入：每个样本统一采样为 **30 帧**
  - mmWave：距离-多普勒/角度热图序列，形状 `(30, H, W)`（原始 `.mat`，或离线 CFAR 后的 `.npy`）
  - IMU：`(30, 6)`（加速度 3 + 角速度 3）
- 输出：每帧每个 query 预测
  - `pred_kpt`：`21*3 = 63`（21 个手部关节的 xyz）
  - `pred_logits`：是否为有效手（`num_logitsclass+1 = 2`）
- 监督：每帧用 **匈牙利匹配** 把 100 个 query 对到 GT 手（1 或 2 只手），再回归关键点 + 分类。
- 指标：`MPJPE`（主指标，mm）、`MPJDLE`(_h/_v/_d) 及五指（thumb/index/middle/ring/pinky）分项；分类 `accuracy` + 混淆矩阵。

## 整体架构（两阶段）

模型主体 `mmVR_AttnBiLSTM`（[models/mmVR_AttnBiLSTM.py](models/mmVR_AttnBiLSTM.py)），前向 `forward(src, imu)`：

**Stage 1 — 逐帧空间编码（原型部分，未改）**
1. mmWave：`mmwave_backbone`（5 层 Conv3d+BN+ReLU+MaxPool）→ `fc_mmwave` → `(30, B, 512)`
2. IMU：`imu_backbone`（2 层单向 LSTM，hidden 512）→ `(30, B, 512)`
3. 位置编码后分别过 `encoder`（mmWave）与 `encoder_imu`（IMU），按 `0.9*mmwave + 0.1*imu` 融合得到 `memory`
4. `posedecoder`（`TransformerPoseDecoder`，6 层，30 个 query=每帧一个）→ `pose_memory`，形状 `(30, B, 512)`，即逐帧姿态特征序列

**Stage 2 — 时序解码（已改进，重点）**
- `AttentionBiLSTMTemporalDecoder`（[models/mmVR_AttnBiLSTM.py:72](models/mmVR_AttnBiLSTM.py#L72)）替换了原型/废弃版本的时序模块。
- 输入：100 个时序 query（`tgt_tem` / `temporalquery_embed`）+ `pose_memory` + `memory`
- 流程：
  1. `temporal = (pose_memory + memory)` →（沿 30 帧）多层 `ResidualSelfAttentionBlock` 做帧间自注意力
  2. 过 **BiLSTM**（`temporal_bilstm`，双向）→ `temporal_proj` 投回 d_model → 显式时序上下文
  3. **门控融合**：把 100 个 query 与每帧时序上下文外积广播，`context_gate` 产生 sigmoid 门 → `fused = frame_queries + gate * frame_context`
  4. 再过多层 `ResidualSelfAttentionBlock` 在每帧内对 100 个 query 做自注意力
- 输出 `(30, 100, B, d)` → `prediction_head` 得到 `pred_logits` / `pred_kpt`

> 设计要点：用 **自注意力 + BiLSTM 双向时序建模 + 门控融合** 替代原时序解码器，
> 增强跨帧时间一致性。层数等由 `--attn_temporal_layers / --query_refine_layers / --bilstm_layers / --bilstm_hidden_dim` 控制。

## 训练入口与配置

- 入口：[train_kpt_attn_bilstm.py](train_kpt_attn_bilstm.py)（`model_name = 'train_attn_bilstm'`）
- 配置：[config_attn_bilstm.py](config_attn_bilstm.py)（argparse，`config.args`）
- 训练循环逻辑全部抽到 [utils/train_runtime.py](utils/train_runtime.py)：
  - `run_pose_epoch`：单 epoch 训练/验证（含 AMP、梯度累积 `--grad_accum_steps`、梯度裁剪、耗时统计）
  - lr 调度默认 `plateau`（监控 `MPJPE`），含 early stopping（`--early_stop_patience`）
  - `effective_batch_size != 32` 且 lr 为默认 1e-4 时会按 batch 自动线性缩放 lr
  - `torch.compile`（`--compile`）、AMP（`--amp`，仅 CUDA）、可选显存统计（`--mem_stats`）
- 最优权重保存：`./experiments/param/train_attn_bilstm/best_test_mpjpe.pth`（按验证 MPJPE）
- 日志：`./logs/<model_name>/{train,test}`（TensorBoard）+ `./experiments/...`（log/混淆矩阵/csv）

运行：`python train_kpt_attn_bilstm.py --dataset_root ./data [--amp --batch_size 32 ...]`

## 数据布局

`dataset_root`（默认 `./data`）下：

```
train_list.txt / eval_list.txt   # 每行: name,person_id,scene,label  (label 从 1 起，代码内 -1)
mmwave/<name>.mat                 # 原始 mmWave (h5: 键 'data')
mmwave_cfar/<signature>/<name>.npy   # 离线 CFAR 结果 (--cfar_mode os2d 时使用)
imu/<name>.npy                   # (T, 6)
kpt_gt/<name>.npy                # (T, 21or42, ...)  -> reshape 成 (30, 1或2手, 63)
```

- 数据集：[dataset/datasets.py](dataset/datasets.py) `HM_Dataset` / `build_dataset`
- 所有模态统一线性采样到 `TARGET_NUM_FRAMES=30`（`get_uniform_frame_indices`）
- `collate_fn`（[utils/misc.py](utils/misc.py)）：stack mmwave/imu，targets 保持 list（变长手数）

## CFAR 预处理（可选）

- 算法：2D OS-CFAR，[utils/cfar.py](utils/cfar.py)（`os_cfar_2d` / `apply_mmwave_cfar`，支持上下半幅分别处理 `--cfar_split_halves`）
- 离线缓存机制：[utils/mmwave_cfar_dataset.py](utils/mmwave_cfar_dataset.py)，按参数生成 signature 目录名
- 预计算脚本：`python scripts/precompute_mmwave_cfar.py --dataset_root ./data --cfar_mode os2d ...`
- 训练时 `--cfar_mode os2d`（配 `--cfar_guard/--cfar_train/--cfar_rank_ratio/--cfar_pfa/--cfar_soft_mode`）会读取匹配 signature 的离线 `.npy`，**找不到会报错**——CFAR 参数必须与预计算时一致。

## 损失与匹配

- [utils/matcher.py](utils/matcher.py) `HungarianMatcher`：逐帧匹配，cost = `set_cost_kpt * L1(kpt) + set_cost_class * (-logits)`
- [utils/loss.py](utils/loss.py) `SetCriterion`：
  - `loss_kpt`：MSE（除以匹配关键点数）；`loss_cls`：交叉熵
  - 权重 `kpt_loss_coef=5, cls_loss_coef=1`
  - 指标累计在 criterion 内（`reset_epoch_stats` / `get_epoch_metrics`），由 `calc_mpjpe` 计算 MPJPE/MPJDLE 及五指分项

## 关键文件地图

| 文件 | 作用 |
|---|---|
| [train_kpt_attn_bilstm.py](train_kpt_attn_bilstm.py) | 训练入口（stage2 改进版） |
| [config_attn_bilstm.py](config_attn_bilstm.py) | 超参/命令行配置 |
| [models/mmVR_AttnBiLSTM.py](models/mmVR_AttnBiLSTM.py) | 主模型 + stage2 时序解码器 + 预测头 + `build_model` |
| [models/Transformer_layers.py](models/Transformer_layers.py) | Encoder / PoseDecoder（stage1）层定义 |
| [dataset/datasets.py](dataset/datasets.py) | 数据集加载 |
| [utils/train_runtime.py](utils/train_runtime.py) | 训练循环/调度/AMP/early-stop |
| [utils/loss.py](utils/loss.py) | SetCriterion + 指标 |
| [utils/matcher.py](utils/matcher.py) | 匈牙利匹配 |
| [utils/cfar.py](utils/cfar.py) / [utils/mmwave_cfar_dataset.py](utils/mmwave_cfar_dataset.py) | CFAR 算法与离线缓存 |
| [utils/misc.py](utils/misc.py) | collate_fn / 日志 / 混淆矩阵 / 分布式工具 |

## 注意事项 / 易踩坑

- **mamba 相关代码为废弃部分，无需阅读、不要参考**（已被本 stage2 AttnBiLSTM 方案取代）。
- `build_backbone`（resnet）虽在 `build_model` 中构建，但 `forward` 实际只用 `mmwave_backbone`（Conv3d）与 `imu_backbone`（LSTM），resnet 路径基本闲置。
- 帧数硬编码为 30、关键点硬编码为 21（`prediction_head` 输出 `21*3`、`calc_mpjpe` reshape 到 `(-1,21,3)`），改帧数/关键点数需多处同步。
- `loss.py:save_as_npy` 内有写死的绝对路径（`/home/lvyizhe/...`），默认未调用，勿误触发。
- 类别融合权重 `0.9/0.1`（mmwave/imu）写死在 `forward` 中。
- CFAR 训练前必须先跑预计算脚本，且参数与训练命令完全一致，否则 signature 目录对不上会 `FileNotFoundError`。
