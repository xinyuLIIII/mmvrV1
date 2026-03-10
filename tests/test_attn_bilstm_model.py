import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_attn_bilstm_module(monkeypatch):
    sys.modules.pop('models.mmVR_AttnBiLSTM', None)
    monkeypatch.setattr(sys, 'argv', ['test_attn_bilstm_model.py'])
    return importlib.import_module('models.mmVR_AttnBiLSTM')


def test_attn_bilstm_temporal_decoder_shape(monkeypatch):
    module = _load_attn_bilstm_module(monkeypatch)
    decoder = module.AttentionBiLSTMTemporalDecoder(
        d_model=32,
        num_frames=5,
        num_queries=7,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        attn_temporal_layers=2,
        query_refine_layers=2,
        bilstm_layers=1,
        bilstm_hidden_dim=16,
    )
    tgt = torch.zeros(7, 3, 32)
    pose_memory = torch.randn(5, 3, 32)
    memory = torch.randn(5, 3, 32)
    pos = torch.randn(5, 3, 32)
    query_pos = torch.randn(7, 3, 32)

    out = decoder(tgt, pose_memory, memory, pos=pos, query_pos=query_pos)

    assert out.shape == (5, 7, 3, 32)


def test_attn_bilstm_build_model_forward(monkeypatch):
    module = _load_attn_bilstm_module(monkeypatch)
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
        attn_temporal_layers=2,
        query_refine_layers=2,
        bilstm_layers=1,
        bilstm_hidden_dim=256,
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
