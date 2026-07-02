# Execution Context

## Machine Context

- Repository: `/root/mmvrV1`
- Shell: `bash`
- Server mode: command-line only, no GUI/frontend setup expected
- Current date at initialization: 2026-04-25

## Recommended Environment

Use the repository-provided Python 3.10 setup:

```bash
bash scripts/setup_mmvr_py310.sh
conda activate mmvr-py310
```

Alternative:

```bash
conda env create -f environment_py310.yaml
conda activate mmvr-py310
```

## Training Commands

Stage-1 keypoint baseline:

```bash
python train_kpt.py
```

Stage-1 Mamba variant:

```bash
python train_kpt_mamba.py
```

Stage-2 classification:

```bash
python train_cls.py
```

Observed recent stage-2 command pattern:

```bash
python -u train_cls.py --dataset_root_kpt ./data --device cuda --epoch 200 --batch_size 32
```

## Tests

Full test suite:

```bash
pytest -q
```

Environment file validation:

```bash
python -m unittest tests.test_env_files -v
```

## CLI Tooling

- Node.js: `v22.22.2`
- npm: `10.9.7`
- Codex CLI: `codex-cli 0.125.0`
- Oh My Paper plugin script check: `on-session-start.mjs` exits successfully when run manually.
