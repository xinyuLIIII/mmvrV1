import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_mamba_module(monkeypatch, install_stub):
    sys.modules.pop('models.mmVR_Mamba', None)
    monkeypatch.setattr(sys, 'argv', ['test_mamba_model.py'])
    if install_stub:
        stub = types.ModuleType('mamba_ssm')

        class DummyMamba(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand):
                super().__init__()
                self.proj = nn.Linear(d_model, d_model)

            def forward(self, x):
                return self.proj(x)

        stub.Mamba = DummyMamba
        monkeypatch.setitem(sys.modules, 'mamba_ssm', stub)
    else:
        monkeypatch.setitem(sys.modules, 'mamba_ssm', None)
    return importlib.import_module('models.mmVR_Mamba')


def test_mamba_temporal_decoder_shape(monkeypatch):
    module = _load_mamba_module(monkeypatch, install_stub=True)
    decoder = module.MambaTemporalDecoder(
        d_model=32,
        num_frames=5,
        num_queries=7,
        num_layers=2,
        d_state=8,
        d_conv=2,
        expand=2,
        dropout=0.0,
    )
    tgt = torch.zeros(7, 3, 32)
    pose_memory = torch.randn(5, 3, 32)
    memory = torch.randn(5, 3, 32)
    pos = torch.randn(5, 3, 32)
    query_pos = torch.randn(7, 3, 32)

    out = decoder(tgt, pose_memory, memory, pos=pos, query_pos=query_pos)

    assert out.shape == (5, 7, 3, 32)


def test_mamba_build_model_forward(monkeypatch):
    module = _load_mamba_module(monkeypatch, install_stub=True)
    args = SimpleNamespace(
        device='cpu',
        backbone_name='resnet18',
        backbone_pretrain=False,
        position_embedding='learned',
        dilation=False,
        enc_layers=2,
        dec_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        nheads=8,
        num_queries=100,
        cls_loss_coef=1.0,
        kpt_loss_coef=5.0,
        set_cost_class=5.0,
        set_cost_kpt=25.0,
        mem_stats=True,
        mamba_layers=2,
        mamba_d_state=8,
        mamba_d_conv=2,
        mamba_expand=2,
    )

    model, criterion = module.build_model(args)
    model.eval()
    criterion.eval()

    samples = torch.randn(1, 30, 256, 128)
    imu = torch.randn(1, 30, 6)
    with torch.no_grad():
        out = model(samples, imu)

    assert out['pred_logits'].shape == (30, 1, 100, 2)
    assert out['pred_kpt'].shape == (30, 1, 100, 63)
    assert out['pose_memory'].shape == (30, 1, 512)


def test_mamba_dependency_error_is_clear(monkeypatch):
    module = _load_mamba_module(monkeypatch, install_stub=False)

    try:
        module.ResidualMambaBlock(d_model=32)
    except module.MissingMambaDependencyError as exc:
        message = str(exc)
    else:
        raise AssertionError('Expected MissingMambaDependencyError to be raised')

    assert 'mamba-ssm' in message
    assert 'causal-conv1d' in message
