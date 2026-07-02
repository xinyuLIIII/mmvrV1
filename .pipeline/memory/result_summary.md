# Result Summary

## Current Best Local Signals

- Stage-2 gesture classification local best: 89.28909952606635% test accuracy.
- README target/reference: 90.8% gesture recognition accuracy.
- Stage-1 Attention-BiLSTM local best: 21.0356 test MPJPE at epoch 87, from the last clean epoch 1-90 segment in `experiments/train_attn_bilstm_test.log`.
- Stage-1 final epoch 90 test MPJPE: 21.2411; final test MPJDLE: 10.3329.
- Stage-1 best checkpoint: `experiments/param/train_attn_bilstm/best_test_mpjpe.pth`.

## Interpretation

The user states that stage 1 has already been modified and completed through the Attention-BiLSTM path. The stage-1 metric to carry forward is best test MPJPE 21.0356, not the low pose-log classification accuracy. The stage-2 result is close to the README-reported number but does not yet match it, so the next experimental question is whether stage2 was run on the intended keypoint source from the completed stage1 path.

## Missing for Paper-Ready Reporting

- Exact reproducible commands for each reported metric, especially the March 17 Attention-BiLSTM run.
- Dataset split and preprocessing provenance, including whether the stage1 result was launched with explicit `--cfar_mode os2d`.
- Checkpoint paths for the best reported metrics.
- Confidence intervals or repeated-run stability if the paper/report needs stronger empirical support.
