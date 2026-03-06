import torch
from utils.misc import MemStatsAccumulator


def test_mem_stats_update_from_out():
    stats = MemStatsAccumulator()
    out = {"pose_memory": torch.randn(2, 3, 4)}
    stats.update(out["pose_memory"])
    summary = stats.summary()
    assert summary["count"] == 24
