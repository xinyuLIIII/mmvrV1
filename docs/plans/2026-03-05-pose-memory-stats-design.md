# Pose Memory Stats Design

## Goal
Add lightweight instrumentation to Stage1 (`train_kpt.py`) to compare train/test `pose_memory` distributions and judge overfitting in the pose stage.

## Scope
- Record `pose_memory` stats per epoch for train and test.
- Output to TensorBoard and JSON.
- Minimal runtime overhead; off by default.

## Design
- Expose `pose_memory` from `mmVR_Transformer.forward` when enabled via args.
- Add a small accumulator to compute mean/std/min/max/l1/l2/abs_mean/sparsity over all batches.
- In `train_kpt.py`, collect stats during each epoch for train/test and write:
  - `logs/train` + `logs/test` TensorBoard scalars.
  - `experiments/mem_stats/pose_memory_{train|test}_epoch_{i}.json`.

## Config
New args in `config.py`:
- `--mem_stats` (default False)
- `--mem_stats_save` (default `./experiments/mem_stats/`)
- `--mem_stats_every` (default 1)

## Validation
- Run 1-2 epochs with `--mem_stats` and confirm JSON/TensorBoard outputs exist.
- Ensure training is unchanged when `--mem_stats` is not set.
