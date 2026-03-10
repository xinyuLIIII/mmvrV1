# Repository Guidelines

## Project Structure & Module Organization
- Training entry points live at the repo root: `train_kpt.py`, `train_kpt_mamba.py`, and `train_cls.py`.
- Core model code is in `models/`; dataset loaders and preprocessing live in `dataset/`; reusable losses, matching, and visualization helpers belong in `utils/`.
- Automated checks live in `tests/`, with focused files for environment definitions, memory logging, visualization, and Mamba model coverage.
- Keep raw or downloaded data under `data/`. Treat `experiments/` and `logs/` as generated outputs, and store planning notes in `docs/plans/`.

## Build, Test, and Development Commands
- `bash scripts/setup_mmvr_py310.sh` creates the recommended Python 3.10 environment with the current PyTorch and Mamba wheel setup.
- `conda env create -f environment_py310.yaml` installs the unified Conda environment from the checked-in spec.
- `python train_kpt.py` runs stage-1 keypoint training; `python train_kpt_mamba.py` runs the Mamba variant; `python train_cls.py` runs stage-2 gesture classification.
- `pytest -q` runs the full test suite. Use `python -m unittest tests.test_env_files -v` for environment-file validation only.

## Coding Style & Naming Conventions
- Follow the existing Python style: 4-space indentation, `snake_case` for functions, variables, and module names, and `PascalCase` for classes.
- Keep configuration changes explicit in `config.py` or `config_mamba.py`, and mirror environment-related changes across `environment_*.yaml`, `requirements_*.txt`, and `scripts/setup_mmvr_py310.sh`.
- Prefer small, focused helpers in `utils/` and keep new dataset or model logic near the related module instead of adding new top-level folders.
- No formatter or linter config is committed; match the surrounding file style and keep imports tidy.

## Testing Guidelines
- Add tests in `tests/` using `test_*.py` filenames.
- Start with the narrowest relevant check, then expand if needed. Example: run `pytest tests/test_mamba_model.py -q` before broader suite runs.
- Extend `tests/test_env_files.py` for environment/setup changes, and follow the memory/model test patterns already used in `tests/`.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative commit subjects; many commits use prefixes such as `feat:` and `chore:`.
- Prefer messages like `fix: align py310 setup defaults` or `test: cover mamba import path`.
- Pull requests should summarize the training or environment impact, list changed files/commands, link related issues, and include logs or screenshots when outputs or metrics change.

## Configuration Tips
- Do not commit dataset contents, model weights, or generated artifacts from `experiments/` and `logs/`.
- When dependency pins change, update `README.md` and the relevant environment/setup files in the same patch.
