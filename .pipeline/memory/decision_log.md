# Decision Log

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-04-25 | Initialize Oh My Paper at `experiment` stage. | The repository already has training entry points, data references, generated logs, and previous experiment outputs. |
| 2026-04-25 | Preserve the existing root `AGENTS.md`. | It contains repository-specific development rules supplied for `/root/mmvrV1`; overwriting it would be unsafe. |
| 2026-04-25 | Keep setup command-line only. | The user stated this server has no visual environment and frontend features are not needed. |
| 2026-04-25 | Treat Attention-BiLSTM as the completed stage-1 improvement path. | The user stated stage1 has already been modified; local analysis confirms `train_kpt_attn_bilstm.py`, result logs, and best MPJPE checkpoint are present. |
