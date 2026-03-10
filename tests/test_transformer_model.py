import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

module = importlib.import_module('models.mmVR_Transformer')


def test_transformer_build_model_forward_cpu():
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


def test_transformer_criterion_handles_single_hand_targets():
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
        mem_stats=False,
    )

    model, criterion = module.build_model(args)
    model.eval()
    criterion.eval()

    samples = torch.randn(1, 30, 256, 128)
    imu = torch.randn(1, 30, 6)
    with torch.no_grad():
        out = model(samples, imu)

    target = [{
        'kpt': torch.randn(30, 1, 63),
        'kpt_cls': torch.ones(30, 1, 1, dtype=torch.long),
        'label': torch.tensor(0),
        'filename': torch.tensor([1, 2, 3, 4]),
    }]

    loss_dict = criterion(out, target)

    assert 'loss_ce' in loss_dict
    assert 'loss_kpt' in loss_dict
