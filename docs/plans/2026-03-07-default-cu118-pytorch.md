# Default `cu118` PyTorch Installation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `scripts/setup_mmvr_py310.sh` default to official `cu118` PyTorch wheels while keeping docs and tests aligned.

**Architecture:** Keep the existing two-mode installer structure, but switch the default mode from `mirror` to `official-cu118`. This preserves the current escape hatch for mirror-based installs while ensuring the default path matches the repository's CUDA 11.8-compatible wheel assumptions.

**Tech Stack:** Bash, unittest, Markdown documentation

---

### Task 1: Update the failing expectation

**Files:**
- Modify: `tests/test_env_files.py`
- Test: `tests/test_env_files.py`

**Step 1: Write the failing test**

Update the setup-script assertion to expect:

```python
self.assertIn('PYTORCH_INDEX_MODE="${PYTORCH_INDEX_MODE:-official-cu118}"', script_text)
```

**Step 2: Run test to verify it fails**

Run: `cd .. && PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_env_files.EnvironmentDefinitionTests.test_unified_py310_environment_files_are_portable`
Expected: FAIL because the script still defaults to `mirror`.

**Step 3: Write minimal implementation**

Change the default `PYTORCH_INDEX_MODE` in `scripts/setup_mmvr_py310.sh` to `official-cu118`.

**Step 4: Run test to verify it passes**

Run: `cd .. && PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_env_files.EnvironmentDefinitionTests.test_unified_py310_environment_files_are_portable`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/setup_mmvr_py310.sh tests/test_env_files.py README.md docs/plans/2026-03-07-default-cu118-pytorch.md
git commit -m "fix: default py310 setup to cu118"
```

### Task 2: Update user-facing docs

**Files:**
- Modify: `README.md`
- Test: `tests/test_env_files.py`

**Step 1: Update the install note**

Replace the README language that says the default mode is `mirror` with wording that says the default mode is `official-cu118`, while noting that users can still override back to `mirror` if they want their local pip mirror to resolve PyTorch.

**Step 2: Verify the docs remain accurate**

Confirm the README examples still match the script's supported modes: `official-cu118` and `mirror`.

**Step 3: Commit**

```bash
git add README.md scripts/setup_mmvr_py310.sh tests/test_env_files.py docs/plans/2026-03-07-default-cu118-pytorch.md
git commit -m "docs: align setup defaults with cu118"
```
