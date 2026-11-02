from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_gesture_task_only_has_gesture_head():
    from models.multitask_resnet import MultiTaskResNet
    model = MultiTaskResNet(task='gesture')
    out = model(torch.randn(2, 30, 42, 3))
    assert out['gesture'].shape == (2, 8)
    assert out['identity'] is None


def test_identity_task_only_has_identity_head():
    from models.multitask_resnet import MultiTaskResNet
    model = MultiTaskResNet(task='identity')
    out = model(torch.randn(2, 30, 42, 3))
    assert out['identity'].shape == (2, 8)
    assert out['gesture'] is None


def test_dual_task_has_both_heads():
    from models.multitask_resnet import MultiTaskResNet
    model = MultiTaskResNet(task='dual')
    out = model(torch.randn(2, 30, 42, 3))
    assert out['gesture'].shape == (2, 8)
    assert out['identity'].shape == (2, 8)
