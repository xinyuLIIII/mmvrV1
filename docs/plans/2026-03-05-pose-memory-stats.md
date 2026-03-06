# Pose Memory Stats Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional pose-memory distribution logging in Stage1 to compare train/test overfitting.

**Architecture:** Expose `pose_memory` from the model when enabled, accumulate lightweight stats per epoch, and write TensorBoard + JSON outputs. Add args to toggle logging and configure output path.

**Tech Stack:** PyTorch, TensorBoardX, JSON, Python stdlib.

---

### Task 1: Add `MemStatsAccumulator`

**Files:**
- Modify: `utils/misc.py`
- Create: `tests/test_mem_stats.py`

**Step 1: Write the failing test**

```python
# tests/test_mem_stats.py
import torch
from utils.misc import MemStatsAccumulator

def test_mem_stats_basic():
    stats = MemStatsAccumulator()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stats.update(x)
    summary = stats.summary()
    assert summary["count"] == 4
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mem_stats.py::test_mem_stats_basic -v`
Expected: FAIL with `ImportError` or `AttributeError` for `MemStatsAccumulator`.

**Step 3: Write minimal implementation**

```python
class MemStatsAccumulator:
    def __init__(self, eps=1e-6, sparsity_thresh=1e-3):
        self.eps = eps
        self.sparsity_thresh = sparsity_thresh
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None
        self.abs_sum = 0.0
        self.l1 = 0.0
        self.l2_sq = 0.0
        self.sparse_count = 0

    def update(self, tensor):
        x = tensor.detach().float().view(-1)
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.sum += x.sum().item()
        self.sum_sq += (x * x).sum().item()
        self.abs_sum += x.abs().sum().item()
        self.l1 += x.abs().sum().item()
        self.l2_sq += (x * x).sum().item()
        self.sparse_count += (x.abs() < self.sparsity_thresh).sum().item()
        cur_min = x.min().item()
        cur_max = x.max().item()
        self.min = cur_min if self.min is None else min(self.min, cur_min)
        self.max = cur_max if self.max is None else max(self.max, cur_max)

    def summary(self):
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "l1": 0.0,
                "l2": 0.0,
                "abs_mean": 0.0,
                "sparsity": 0.0,
            }
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        std = var ** 0.5
        l2 = (self.l2_sq ** 0.5)
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "min": self.min,
            "max": self.max,
            "l1": self.l1,
            "l2": l2,
            "abs_mean": self.abs_sum / self.count,
            "sparsity": float(self.sparse_count) / float(self.count),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mem_stats.py::test_mem_stats_basic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/misc.py tests/test_mem_stats.py
git commit -m "feat: add memory stats accumulator"
```

---

### Task 2: Add config flags for mem stats

**Files:**
- Modify: `config.py`
- Create: `tests/test_config_mem_stats.py`

**Step 1: Write the failing test**

```python
# tests/test_config_mem_stats.py
import importlib
import sys


def test_config_mem_stats_defaults():
    sys.argv = ["test"]
    config = importlib.import_module("config")
    importlib.reload(config)
    assert hasattr(config.args, "mem_stats")
    assert config.args.mem_stats is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_mem_stats.py::test_config_mem_stats_defaults -v`
Expected: FAIL because `mem_stats` is missing.

**Step 3: Write minimal implementation**

Add in `config.py`:
```python
parser.add_argument('--mem_stats', action='store_true', default=False)
parser.add_argument('--mem_stats_save', default='./experiments/mem_stats/')
parser.add_argument('--mem_stats_every', default=1, type=int)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_mem_stats.py::test_config_mem_stats_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config.py tests/test_config_mem_stats.py
git commit -m "feat: add mem stats config flags"
```

---

### Task 3: Wire pose memory logging in Stage1

**Files:**
- Modify: `models/mmVR_Transformer.py`
- Modify: `train_kpt.py`

**Step 1: Write the failing test**

```python
# tests/test_pose_memory_logging.py
import torch
from utils.misc import MemStatsAccumulator

def test_mem_stats_update_from_out():
    stats = MemStatsAccumulator()
    out = {"pose_memory": torch.randn(2, 3, 4)}
    stats.update(out["pose_memory"])
    summary = stats.summary()
    assert summary["count"] == 24
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pose_memory_logging.py::test_mem_stats_update_from_out -v`
Expected: FAIL because `MemStatsAccumulator` is not wired into training yet (test still validates accumulator).

**Step 3: Write minimal implementation**

- In `models/mmVR_Transformer.forward`, if `args.mem_stats` is True, add:
  ```python
  out['pose_memory'] = pose_memory.detach()
  ```
- In `train_kpt.py`:
  - Initialize `train_mem_stats` and `test_mem_stats` when `args.mem_stats`.
  - In each batch, call `train_mem_stats.update(out['pose_memory'])`.
  - At epoch end, write JSON to `args.mem_stats_save` and TB scalars under `mem/pose/*`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pose_memory_logging.py::test_mem_stats_update_from_out -v`
Expected: PASS

**Step 5: Commit**

```bash
git add models/mmVR_Transformer.py train_kpt.py tests/test_pose_memory_logging.py
git commit -m "feat: log pose memory stats in stage1"
```

---

### Task 4: Smoke check

**Files:**
- None (runtime check)

**Step 1: Run a short training**

Run:
```bash
python train_kpt.py --epoch 1 --batch_size 2 --num_workers 0 --mem_stats
```
Expected: JSON files in `./experiments/mem_stats/` and TensorBoard scalars under `mem/pose/*`.

**Step 2: Commit (optional)**

If any adjustments were needed, commit with a small message.
