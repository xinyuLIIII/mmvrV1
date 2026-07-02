# Orchestrator State

## Current Stage

experiment

## Current Focus

Continue from the completed stage-1 Attention-BiLSTM improvement and focus on stage2 classification, reproducibility, and paper-ready result attribution.

## Immediate Priorities

- Treat `train_kpt_attn_bilstm.py` and `models/mmVR_AttnBiLSTM.py` as the current completed stage-1 improvement path.
- Use the recorded stage-1 best test MPJPE 21.0356 at epoch 87 as the current stage-1 reference result.
- Verify whether stage2 classification used keypoints generated from the completed Attention-BiLSTM stage1 path.
- Reproduce stage2 classification with `train_cls.py` and compare against the latest local best of 89.28909952606635%.
- Re-record the exact stage1 command line if paper-ready reproducibility text is needed.

## Open Risks

- Existing logs indicate stage-2 accuracy is close to, but below, the README-reported 90.8%.
- `data/mmwave_cfar` is a symlink to `/root/autodl-tmp/mmVR/mmwave_cfar` and is currently untracked by git.
- The stage1 full log filename indicates CFAR, but the exact command line is missing from the captured logs.
- Existing root `AGENTS.md` is repository guidance, not the generated Oh My Paper harness text.
