from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.disentangle import (  # noqa: E402
    DisentConfig, dann_anneal, floor_gate, grad_reverse, scheduled_lambda,
)
from models.multitask_resnet_disent import build_disent_model  # noqa: E402


class _TinyDual(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.randn(30, 42, 3), i % 8, (i + 1) % 8


def test_grl_forward_identity_backward_negated():
    x = torch.randn(3, 5, requires_grad=True)
    y = grad_reverse(x, 0.25)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.25))


def test_none_method_matches_plain_shapes():
    m = build_disent_model('dual', cfg=DisentConfig(method='none'))
    out = m(torch.randn(2, 30, 42, 3))
    assert out['gesture'].shape == (2, 8)
    assert out['identity'].shape == (2, 8)
    assert out['adv_identity'] is None
    assert m.head_gesture.in_features == 2048


def test_bigrl_builds_subspaces_and_adversaries():
    cfg = DisentConfig(method='bigrl')
    m = build_disent_model('dual', cfg=cfg)
    out = m(torch.randn(2, 30, 42, 3))
    assert out['gesture'].shape == (2, 8)
    assert out['identity'].shape == (2, 8)
    assert out['f_g'].shape == (2, cfg.dim_g)
    assert out['f_i'].shape == (2, cfg.dim_i)
    assert out['f_s'].shape == (2, cfg.dim_s)
    assert out['adv_identity'].shape == (2, 8)
    assert out['adv_gesture'].shape == (2, 8)
    assert out['recon'].shape == (2, 2048)
    assert m.head_gesture.in_features == cfg.dim_g + cfg.dim_s
    assert m.head_identity.in_features == cfg.dim_i + cfg.dim_s


def test_split_method_has_no_adversaries():
    m = build_disent_model('dual', cfg=DisentConfig(method='split'))
    out = m(torch.randn(2, 30, 42, 3))
    assert out['adv_identity'] is None and out['adv_gesture'] is None
    assert out['recon'] is not None
    assert len(list(m.discriminator_parameters())) == 0


def test_detach_backbone_blocks_adversarial_gradient_into_backbone():
    """The load-bearing safety property: with adv_detach_backbone=True the
    reversed gradient never reaches the shared ResNet."""
    for detach, expect_zero in ((True, True), (False, False)):
        cfg = DisentConfig(method='bigrl', recon_weight=0.0,
                           adv_detach_backbone=detach)
        m = build_disent_model('dual', cfg=cfg)
        m.set_lambda(1.0, 1.0)
        out = m(torch.randn(2, 30, 42, 3))
        loss = (torch.nn.functional.cross_entropy(out['adv_identity'], torch.tensor([0, 1]))
                + torch.nn.functional.cross_entropy(out['adv_gesture'], torch.tensor([2, 3])))
        m.zero_grad()
        loss.backward()
        w = m.backbone.conv1.weight
        got_zero = (w.grad is None) or bool(torch.all(w.grad == 0))
        assert got_zero == expect_zero


def test_lambda_never_exceeds_max_and_starts_at_zero():
    cfg = DisentConfig(method='bigrl', lambda_max=0.3,
                       adv_start_epoch=40, adv_ramp_epochs=40)
    assert scheduled_lambda(0, cfg) == 0.0
    assert scheduled_lambda(39, cfg) == 0.0
    assert scheduled_lambda(40, cfg) == 0.0
    assert 0.0 < scheduled_lambda(60, cfg) < 0.3
    assert scheduled_lambda(200, cfg) <= 0.3
    assert abs(dann_anneal(1.0) - (2 / (1 + 2.718281828 ** -10) - 1)) < 1e-6


def test_floor_gate_closes_at_chance():
    hi = floor_gate(0.90, 8, margin=0.05, temp=0.02)
    lo = floor_gate(0.125, 8, margin=0.05, temp=0.02)
    assert hi > 0.99
    assert lo < 0.10


def test_disent_epoch_runs_and_reports_probes():
    from train_multitask_disent import run_disent_epoch
    cfg = DisentConfig(method='bigrl')
    device = torch.device('cpu')
    model = build_disent_model('dual', cfg=cfg).to(device)
    model.set_lambda(0.1, 0.1)
    loader = torch.utils.data.DataLoader(_TinyDual(), batch_size=2)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    stats = run_disent_epoch(model, loader, device, 'dual', ce, 1.0,
                             amp_device_type='cpu', amp_enabled=False,
                             optimizer=opt, scaler=None, cfg=cfg,
                             recon_weight=1.0)
    assert stats['count'] == 4
    assert stats['loss_sum'] > 0
    assert stats['loss_rec'] > 0
    assert 0 <= stats['adv_id_correct'] <= 4
    assert 0 <= stats['adv_g_correct'] <= 4
