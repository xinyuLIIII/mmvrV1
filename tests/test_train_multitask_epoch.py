from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _TinyDual(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.randn(30, 42, 3), i % 8, (i + 1) % 8


def test_dual_epoch_trains_and_counts_both_tasks():
    from train_multitask import run_multitask_epoch
    from models.multitask_resnet import build_multitask_model

    device = torch.device('cpu')
    model = build_multitask_model('dual').to(device)
    loader = torch.utils.data.DataLoader(_TinyDual(), batch_size=2)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    stats = run_multitask_epoch(
        model, loader, device, 'dual', ce, 1.0,
        amp_device_type='cpu', amp_enabled=False,
        optimizer=opt, scaler=None,
    )
    assert stats['count'] == 4
    assert stats['loss_sum'] > 0
    assert 0 <= stats['g_correct'] <= 4
    assert 0 <= stats['id_correct'] <= 4


def test_gesture_epoch_ignores_identity():
    from train_multitask import run_multitask_epoch
    from models.multitask_resnet import build_multitask_model

    device = torch.device('cpu')
    model = build_multitask_model('gesture').to(device)
    loader = torch.utils.data.DataLoader(_TinyDual(), batch_size=2)
    ce = torch.nn.CrossEntropyLoss()

    stats = run_multitask_epoch(
        model, loader, device, 'gesture', ce, 1.0,
        amp_device_type='cpu', amp_enabled=False,
        optimizer=None, scaler=None,
    )
    assert stats['count'] == 4
    assert stats['id_correct'] == 0
