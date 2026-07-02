# Project Truth

## Research Topic

EgoHand studies ego-centric hand pose estimation and gesture recognition using a head-mounted millimeter-wave radar and IMUs. The local repository implements a two-stage pipeline:

- Stage 1: estimate hand joint coordinates from mmWave radar and IMU input.
- Stage 2: classify gestures from keypoint sequences.

## Repository Context

- Working repository: `/root/mmvrV1`
- Local paper file: `2501.13805v2.pdf`
- Main entry points: `train_kpt.py`, `train_kpt_attn_bilstm.py`, `train_kpt_mamba.py`, `train_cls.py`
- Core model code: `models/`
- Dataset loaders and preprocessing: `dataset/`
- Shared helpers: `utils/`
- Tests: `tests/`
- Generated outputs: `experiments/`, `logs/`

## Confirmed Environment

- Node.js is installed: `v22.22.2`
- Codex CLI is installed: `codex-cli 0.125.0`
- Oh My Paper Codex plugin is enabled locally.
- Frontend, Tauri, and GUI dependencies are intentionally not installed because this server is command-line only.

## Known Results

- README reports 90.8% gesture recognition accuracy for EgoHand.
- Latest local stage-2 classification run: `logs/stage2_2026-04-24_150742.log`
  - Final epoch: 200
  - Final test accuracy: 88.72037914691943
  - Best tracked test accuracy: 89.28909952606635
- Stage 1 has been modified and is considered locally complete by the user.
  - Current stage-1 script: `train_kpt_attn_bilstm.py`
  - Current stage-1 model: `models/mmVR_AttnBiLSTM.py`
  - Current stage-1 config: `config_attn_bilstm.py`
  - Primary result logs: `experiments/train_attn_bilstm_train.log`, `experiments/train_attn_bilstm_test.log`, and `logs/full_train_2026-03-17_123955_cfar.log2`
  - Last formal Attention-BiLSTM segment spans epoch 1 through epoch 90.
  - Train/test sizes in the full log: 4151 / 1055.
  - Best test MPJPE: 21.0356 at epoch 87.
  - Test MPJDLE at best MPJPE: 10.2480.
  - Final epoch 90 test MPJPE: 21.2411.
  - Best test classification-style accuracy in the pose log: 8.4421 at epoch 83; this is not the main stage-1 success metric.
  - Best checkpoint: `experiments/param/train_attn_bilstm/best_test_mpjpe.pth`.
  - `data/mmwave_cfar` is a symlink to `/root/autodl-tmp/mmVR/mmwave_cfar`, with an `os2d_frames30_guard1_train4_rank0.75_pfa0.001_subtract_split` cache present.
  - The exact command line for the March 17 run is not captured in the available logs; the log filename indicates CFAR, but command-line provenance should be re-recorded before paper-ready reporting.

## Confirmed Decisions

- Initialize Oh My Paper from the `experiment` stage because this repository already contains runnable training code, data references, logs, and generated experiment outputs.
- Preserve the existing repository `AGENTS.md`; do not overwrite it with the Oh My Paper template.
- Keep Oh My Paper initialization CLI-only.
- Treat `train_kpt_attn_bilstm.py` as the current completed stage-1 improvement path unless the user revises this project description.

## Progress Log

- 2026-04-25: Initialized `.pipeline/` for Oh My Paper in `/root/mmvrV1`.
- 2026-04-25: Recorded user context that stage 1 is already modified and completed through `train_kpt_attn_bilstm.py`; analyzed and stored Attention-BiLSTM stage-1 metrics.
