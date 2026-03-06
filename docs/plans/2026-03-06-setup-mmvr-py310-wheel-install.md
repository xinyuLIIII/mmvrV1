# setup_mmvr_py310 Wheel Install Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `scripts/setup_mmvr_py310.sh` to install the known-compatible `causal-conv1d` and `mamba-ssm` wheels by downloading fixed release assets and installing them locally.

**Architecture:** Keep the existing environment bootstrap flow, but replace the source-based Mamba dependency installation with deterministic wheel download URLs and local `pip install` steps. Expose the wheel URLs as environment-variable overrides so the script stays usable if the upstream assets change later.

**Tech Stack:** Bash, conda, pip, GitHub release wheel assets, `unittest`

---

### Task 1: Lock the expected script behavior in tests

**Files:**
- Modify: `tests/test_env_files.py`

**Step 1: Write the failing test**
- Assert that `scripts/setup_mmvr_py310.sh` contains fixed wheel URL variables and local wheel installation commands.

**Step 2: Run test to verify it fails**
- Run: `python -m unittest tests.test_env_files -v`
- Expected: failure because the current script still installs `causal-conv1d` and `mamba-ssm` directly from pip.

### Task 2: Patch the setup script

**Files:**
- Modify: `scripts/setup_mmvr_py310.sh`

**Step 1: Add wheel URL configuration**
- Define default URLs for `causal-conv1d` and `mamba-ssm` compatible with `torch==1.13.1+cu116`.

**Step 2: Add download helper**
- Use `wget` or `curl` to download release assets into a temporary wheel directory.

**Step 3: Install wheels locally**
- Replace direct pip package install with local wheel installation.

### Task 3: Verify behavior

**Files:**
- Verify: `tests/test_env_files.py`
- Verify: `scripts/setup_mmvr_py310.sh`

**Step 1: Run targeted tests**
- Run: `python -m unittest tests.test_env_files -v`
- Expected: pass.

**Step 2: Run syntax verification**
- Run: `bash -n scripts/setup_mmvr_py310.sh`
- Expected: pass.
