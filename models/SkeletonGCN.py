import math

import torch
import torch.nn as nn
import torch.nn.functional as F


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _row_normalize(adjacency, eps=1e-6):
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adjacency / degree


def build_static_hand_adjacency(num_hands=2, joints_per_hand=21, connect_hands=True):
    """Build a 42-node hand skeleton graph with self loops and palm/finger links."""
    num_nodes = num_hands * joints_per_hand
    adjacency = torch.eye(num_nodes, dtype=torch.float32)

    for hand_idx in range(num_hands):
        offset = hand_idx * joints_per_hand
        for start, end in HAND_EDGES:
            adjacency[offset + start, offset + end] = 1.0
            adjacency[offset + end, offset + start] = 1.0

    if connect_hands and num_hands == 2:
        for joint_idx in range(joints_per_hand):
            left = joint_idx
            right = joints_per_hand + joint_idx
            adjacency[left, right] = 1.0
            adjacency[right, left] = 1.0

    return _row_normalize(adjacency)


class SkeletonGraphBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.graph_proj = nn.Linear(in_channels, out_channels)
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )
        self.temporal_proj = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adjacency, node_mask):
        residual = self.residual(x)
        x = aggregate_graph(x, adjacency)
        x = self.graph_proj(x)
        x = self.activation(x)

        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.temporal_proj(x)

        x = self.dropout(x)
        x = self.norm(x + residual)
        x = self.activation(x)
        return apply_node_mask(x, node_mask)


def aggregate_graph(x, adjacency):
    if adjacency.dim() == 2:
        return torch.einsum('vw,btwc->btvc', adjacency, x)
    if adjacency.dim() == 3:
        return torch.einsum('bvw,btwc->btvc', adjacency, x)
    raise ValueError(f'Adjacency must be 2D or 3D, got shape {tuple(adjacency.shape)}')


def apply_node_mask(x, node_mask):
    return x * node_mask[:, None, :, None].to(dtype=x.dtype)


class SkeletonGCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes=8,
        num_frames=30,
        num_nodes=42,
        in_channels=3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        branch='fusion',
        normalize_input=True,
        mask_eps=1e-6,
    ):
        super().__init__()
        if branch not in ('static', 'dynamic', 'fusion'):
            raise ValueError("branch must be one of: 'static', 'dynamic', 'fusion'")
        if num_nodes != 42:
            raise ValueError('SkeletonGCNClassifier currently expects 42 nodes.')

        self.num_frames = num_frames
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.branch = branch
        self.normalize_input_enabled = normalize_input
        self.mask_eps = mask_eps
        self.dynamic_temperature = math.sqrt(float(max(num_frames - 1, 1) * in_channels))

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.static_blocks = self._make_branch(hidden_dim, num_layers, dropout)
        self.dynamic_blocks = self._make_branch(hidden_dim, num_layers, dropout)
        self.register_buffer('static_adjacency', build_static_hand_adjacency(), persistent=False)

        if branch == 'fusion':
            self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            classifier_in = hidden_dim * 3
        else:
            self.fusion_gate = None
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _make_branch(self, hidden_dim, num_layers, dropout):
        return nn.ModuleList(
            SkeletonGraphBlock(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f'SkeletonGCN expects a 3D or 4D tensor, got shape {tuple(x.shape)}')
        if x.shape[-2:] != (self.num_nodes, self.in_channels):
            raise ValueError(
                f'SkeletonGCN expects input shape (B, T, {self.num_nodes}, {self.in_channels}), '
                f'got {tuple(x.shape)}'
            )

        node_mask = self.compute_node_mask(x)
        x = self.normalize_input(x, node_mask)
        dynamic_adjacency = None
        if self.branch in ('dynamic', 'fusion'):
            dynamic_adjacency = self.build_dynamic_adjacency(x, node_mask)
        x = self.input_proj(x)
        x = apply_node_mask(x, node_mask)

        features = []
        if self.branch in ('static', 'fusion'):
            static_adjacency = self.mask_static_adjacency(node_mask, dtype=x.dtype, device=x.device)
            features.append(self.run_branch(x, static_adjacency, node_mask, self.static_blocks))
        if self.branch in ('dynamic', 'fusion'):
            features.append(self.run_branch(x, dynamic_adjacency, node_mask, self.dynamic_blocks))

        if self.branch == 'fusion':
            static_feature, dynamic_feature = features
            concat_feature = torch.cat((static_feature, dynamic_feature), dim=-1)
            gate = torch.sigmoid(self.fusion_gate(concat_feature))
            mixed_feature = gate * static_feature + (1.0 - gate) * dynamic_feature
            feature = torch.cat((mixed_feature, concat_feature), dim=-1)
        else:
            feature = features[0]

        return self.classifier(feature)

    def run_branch(self, x, adjacency, node_mask, blocks):
        for block in blocks:
            x = block(x, adjacency, node_mask)
        return self.masked_pool(x, node_mask)

    def compute_node_mask(self, x):
        return x.detach().abs().sum(dim=(1, 3)) > self.mask_eps

    def normalize_input(self, x, node_mask):
        if not self.normalize_input_enabled:
            return x

        hands = []
        for start in (0, 21):
            end = start + 21
            hand = x[:, :, start:end, :]
            hand_mask = node_mask[:, start:end]
            wrist = hand[:, :, :1, :]
            centered = hand - wrist
            distances = centered.norm(dim=-1)
            valid_distances = distances * hand_mask[:, None, :].to(dtype=x.dtype)
            valid_count = hand_mask.sum(dim=1).clamp_min(1).to(dtype=x.dtype)
            scale = valid_distances.sum(dim=(1, 2)) / (valid_count * hand.shape[1])
            scale = scale.clamp_min(self.mask_eps).view(-1, 1, 1, 1)
            centered = centered / scale
            hands.append(centered)

        normalized = torch.cat(hands, dim=2)
        return apply_node_mask(normalized, node_mask)

    def mask_static_adjacency(self, node_mask, dtype, device):
        adjacency = self.static_adjacency.to(dtype=dtype, device=device)
        adjacency = adjacency.unsqueeze(0) * node_mask[:, :, None].to(dtype) * node_mask[:, None, :].to(dtype)
        return _row_normalize(adjacency)

    def build_dynamic_adjacency(self, x, node_mask=None):
        if x.dim() != 4:
            raise ValueError(f'Dynamic adjacency expects a 4D tensor, got shape {tuple(x.shape)}')
        if node_mask is None:
            node_mask = self.compute_node_mask(x)

        if x.shape[1] > 1:
            velocity = x[:, 1:] - x[:, :-1]
        else:
            velocity = torch.zeros_like(x)

        signature = velocity.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        signature = signature - signature.mean(dim=-1, keepdim=True)
        signature = F.normalize(signature, p=2, dim=-1, eps=self.mask_eps)
        logits = torch.bmm(signature, signature.transpose(1, 2)) / self.dynamic_temperature

        min_value = torch.finfo(logits.dtype).min / 2
        logits = logits.masked_fill(~node_mask[:, None, :], min_value)
        adjacency = F.softmax(logits, dim=-1)
        adjacency = adjacency * node_mask[:, :, None].to(dtype=adjacency.dtype)
        adjacency = adjacency * node_mask[:, None, :].to(dtype=adjacency.dtype)
        return adjacency

    def masked_pool(self, x, node_mask):
        x = apply_node_mask(x, node_mask)
        denominator = node_mask.sum(dim=1).clamp_min(1).to(dtype=x.dtype).unsqueeze(-1)
        denominator = denominator * x.shape[1]
        return x.sum(dim=(1, 2)) / denominator


def skeleton_gcn(num_classes=8, **kwargs):
    return SkeletonGCNClassifier(num_classes=num_classes, **kwargs)
