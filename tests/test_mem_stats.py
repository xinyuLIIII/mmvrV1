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
