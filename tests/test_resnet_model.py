from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ResNet import resnet50  # noqa: E402


def test_resnet50_forward_preserves_batch_dim_for_batch_size_one():
    model = resnet50(num_classes=8)
    x = torch.randn(1, 30, 42, 3)

    out = model(x)

    assert out.shape == (1, 8)


def test_resnet50_forward_handles_multi_sample_batches():
    model = resnet50(num_classes=8)
    x = torch.randn(2, 30, 42, 3)

    out = model(x)

    assert out.shape == (2, 8)
