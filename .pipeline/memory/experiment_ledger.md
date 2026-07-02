# Experiment Ledger

## Recorded Runs

| Date | Stage | Command or Source | Key Output | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-04-24 | stage2 classification | `logs/stage2_2026-04-24_150742.log` | best test accuracy 89.28909952606635%; final test accuracy 88.72037914691943% | completed | 200 epochs, model name `train` |
| 2026-04-24 | stage2 classification | `experiments/train_test.log` | best tracked test accuracy 89.28909952606635% | completed | structured logger output confirms the stage2 max |
| 2026-03-17 | stage1 keypoint, Attention-BiLSTM | `train_kpt_attn_bilstm.py`; `experiments/train_attn_bilstm_*.log`; `logs/full_train_2026-03-17_123955_cfar.log2` | best test MPJPE 21.0356 at epoch 87; final test MPJPE 21.2411 at epoch 90 | completed | user says stage1 modification is complete; exact original command line not captured in logs |

## Stage-1 Attention-BiLSTM Analysis

- Current stage-1 entry point: `train_kpt_attn_bilstm.py`
- Model implementation: `models/mmVR_AttnBiLSTM.py`
- Config: `config_attn_bilstm.py`
- Architecture change: keeps the mmWave 3D CNN, IMU LSTM, separate Transformer encoders, and pose decoder, then adds an `AttentionBiLSTMTemporalDecoder` with temporal self-attention blocks, a bidirectional LSTM, gated temporal context fusion, and query refinement blocks.
- Runtime signal from full log: `AMP enabled: False`, `train_size: 4151`, `test_size: 1055`, initial LR `0.00010000`.
- Last clean appended log segment: epoch 1 through epoch 90.
- Best test MPJPE: 21.0356 at epoch 87.
- Test metrics at best MPJPE: MPJDLE 10.2480, loss_kpt 0.0158, class_error 94.4532, pose-log accuracy 5.3364.
- Final test metrics at epoch 90: MPJPE 21.2411, MPJDLE 10.3329, loss_kpt 0.0161, pose-log accuracy 7.3983.
- Best test pose-log accuracy: 8.4421 at epoch 83, with MPJPE 21.2142.
- Best train MPJPE in the last segment: 16.9170 at epoch 89; this leaves a generalization gap of about 4.3241 MPJPE versus final test MPJPE at epoch 90.
- Finger MPJPE at best test epoch 87: thumb 20.8537, index 22.2933, middle 20.7352, ring 20.5132, pinky 21.0946.
- Best checkpoint: `experiments/param/train_attn_bilstm/best_test_mpjpe.pth` (about 454 MB).
- Accuracy-triggered artifacts exist for epoch 82 / 8.4421: `experiments/weights/train_attn_bilstm/train_attn_bilstm_82_8.4421.csv` and `experiments/conf_matrix/train_attn_bilstm/train_attn_bilstm_82_8.4421.jpg`.
- CFAR cache present through symlink `data/mmwave_cfar -> /root/autodl-tmp/mmVR/mmwave_cfar`, including `os2d_frames30_guard1_train4_rank0.75_pfa0.001_subtract_split`.
- Caveat: `config_attn_bilstm.py` defaults `cfar_mode` to `none`; the March 17 log filename includes `cfar`, but the exact launched command is not recorded. Re-run or explicitly document the command before using the CFAR setting as a paper-ready claim.

## Next Experiments

- Re-run stage2 with the documented Python 3.10 environment and record exact command, commit, data state, and metrics.
- Treat Attention-BiLSTM as the current stage-1 path and feed its best checkpoint or generated keypoints into stage2 as needed.
- Recover or re-run the exact stage1 command with explicit `--cfar_mode` arguments before writing final reproducibility text.
- Compare stage2 results using keypoints from the completed Attention-BiLSTM stage1 path.
