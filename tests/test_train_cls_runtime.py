import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_train_cls_module(monkeypatch):
    sys.modules.pop('train_cls', None)
    sys.modules.pop('config', None)
    monkeypatch.setattr(sys, 'argv', ['test_train_cls_runtime.py'])
    return importlib.import_module('train_cls')


class _TinyDataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.zeros(30, 42, 3), 0


class _DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_build_classification_dataloaders_uses_expected_drop_last(monkeypatch):
    train_cls = _load_train_cls_module(monkeypatch)

    def fake_build_dataset(_, is_train):
        return _TinyDataset(5 if is_train else 3)

    monkeypatch.setattr(train_cls, 'build_dataset', fake_build_dataset)
    args = SimpleNamespace(
        dataset_root_kpt='./data',
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )

    _, _, train_loader, test_loader, train_loader_kwargs, test_loader_kwargs = train_cls.build_classification_dataloaders(args)

    assert train_loader.drop_last is True
    assert test_loader.drop_last is False
    assert train_loader_kwargs['drop_last'] is True
    assert test_loader_kwargs['drop_last'] is False
    assert sum(batch[1].shape[0] for batch in train_loader) == 4
    assert sum(batch[1].shape[0] for batch in test_loader) == 3


def test_build_optimizer_and_scheduler_uses_config_values(monkeypatch):
    train_cls = _load_train_cls_module(monkeypatch)
    model = nn.Linear(4, 2)
    args = SimpleNamespace(lr=5e-4, weight_decay=0.03, lr_drop=7)

    optimizer, scheduler = train_cls.build_optimizer_and_scheduler(model, args)

    assert optimizer.param_groups[0]['lr'] == args.lr
    assert optimizer.param_groups[0]['weight_decay'] == args.weight_decay
    assert scheduler.step_size == args.lr_drop


def test_build_classification_model_selects_skeleton_gcn(monkeypatch):
    train_cls = _load_train_cls_module(monkeypatch)
    args = SimpleNamespace(
        cls_model='skeleton_gcn',
        skeleton_gcn_hidden_dim=16,
        skeleton_gcn_layers=1,
        skeleton_gcn_dropout=0.0,
        skeleton_gcn_branch='fusion',
        skeleton_gcn_normalize=True,
    )

    model = train_cls.build_classification_model(args, num_class=8)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 30, 42, 3))

    assert out.shape == (2, 8)


def test_log_run_start_emits_append_friendly_banner(monkeypatch, capsys):
    train_cls = _load_train_cls_module(monkeypatch)
    train_logger = _DummyLogger()
    test_logger = _DummyLogger()
    args = SimpleNamespace(
        dataset_root_kpt='./data',
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        compile=False,
        matmul_precision='high',
    )
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    run_record = train_cls.create_run_start_record(
        model_name='train',
        args=args,
        device=torch.device('cpu'),
        train_size=10,
        test_size=4,
        train_loader_kwargs={'shuffle': True, 'drop_last': True},
        test_loader_kwargs={'shuffle': False, 'drop_last': False},
        optimizer=optimizer,
        scheduler=scheduler,
        amp_enabled=False,
    )
    train_cls.log_run_start(train_logger, test_logger, run_record)

    assert run_record['run_event'] == 'start'
    assert run_record['test_loader']['drop_last'] is False
    assert train_logger.messages[0]['run_event'] == 'start'
    assert test_logger.messages[0]['run_event'] == 'start'
    assert 'run_start:' in capsys.readouterr().out


def test_run_classification_epoch_counts_processed_samples(monkeypatch):
    train_cls = _load_train_cls_module(monkeypatch)

    def fake_build_dataset(_, is_train):
        return _TinyDataset(5 if is_train else 3)

    monkeypatch.setattr(train_cls, 'build_dataset', fake_build_dataset)
    args = SimpleNamespace(
        dataset_root_kpt='./data',
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    _, _, train_loader, _, _, _ = train_cls.build_classification_dataloaders(args)

    class PerfectClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, 8, dtype=x.dtype, device=x.device)
            logits[:, 0] = 1.0 + self.bias
            return logits

    model = PerfectClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = train_cls.create_grad_scaler('cpu', enabled=False)
    loss_fn = nn.CrossEntropyLoss()

    loss_sum, correct, sample_count, _ = train_cls.run_classification_epoch(
        model,
        train_loader,
        torch.device('cpu'),
        loss_fn,
        'cpu',
        False,
        num_class=8,
        optimizer=optimizer,
        scaler=scaler,
    )

    assert sample_count == 4
    assert correct == 4
    assert loss_sum >= 0.0
