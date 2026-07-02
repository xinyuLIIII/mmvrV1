# Agent Handoff

## Current Position

Oh My Paper has been initialized for `/root/mmvrV1` without replacing the existing root `AGENTS.md`. The user clarified that stage 1 has already been modified and completed via the Attention-BiLSTM path.

## Next Agent Should

- Read `.pipeline/docs/research_brief.json`.
- Read `.pipeline/memory/project_truth.md`.
- Treat `train_kpt_attn_bilstm.py` as the current stage-1 entry point.
- Use `experiments/train_attn_bilstm_test.log` and `experiments/param/train_attn_bilstm/best_test_mpjpe.pth` as the primary stage-1 result references.
- Check `.pipeline/tasks/tasks.json` before starting work.
- Preserve generated data, logs, and experiments unless the user explicitly asks to clean them.

## Important Constraints

- Server is CLI-only. Do not install or start frontend/Tauri/GUI components unless the user explicitly changes direction.
- Root `AGENTS.md` contains repository development rules and should not be overwritten.
- `data/mmwave_cfar` is untracked at initialization time.
- Stage1 exact command line was not recovered from logs; do not overstate CFAR command provenance without a rerun or explicit user confirmation.
