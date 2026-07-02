from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.SkeletonGCN import SkeletonGCNClassifier, build_static_hand_adjacency  # noqa: E402


def test_static_hand_adjacency_has_expected_shape():
    adjacency = build_static_hand_adjacency()

    assert adjacency.shape == (42, 42)
    assert torch.isfinite(adjacency).all()
    assert torch.allclose(adjacency.sum(dim=-1), torch.ones(42))


def test_skeleton_gcn_forward_matches_stage2_contract():
    model = SkeletonGCNClassifier(num_classes=8, hidden_dim=16, num_layers=1)
    model.eval()
    x = torch.randn(2, 30, 42, 3)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_skeleton_gcn_masks_zero_padded_hand_in_dynamic_graph():
    model = SkeletonGCNClassifier(num_classes=8, hidden_dim=16, num_layers=1)
    model.eval()
    x = torch.randn(2, 30, 42, 3)
    x[:, :, 21:, :] = 0.0
    node_mask = model.compute_node_mask(x)
    normalized = model.normalize_input(x, node_mask)

    dynamic_adjacency = model.build_dynamic_adjacency(normalized, node_mask)
    with torch.no_grad():
        out = model(x)

    assert torch.isfinite(out).all()
    assert dynamic_adjacency.shape == (2, 42, 42)
    assert dynamic_adjacency[:, 21:, :].abs().max().item() == 0.0
    assert dynamic_adjacency[:, :, 21:].abs().max().item() < 1e-6


def test_skeleton_gcn_supports_static_and_dynamic_ablation_modes():
    x = torch.randn(2, 30, 42, 3)
    for branch in ('static', 'dynamic'):
        model = SkeletonGCNClassifier(num_classes=8, hidden_dim=16, num_layers=1, branch=branch)
        model.eval()

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 8)
