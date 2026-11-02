"""Route A adaptation: bounded bidirectional GRL on private subspaces.

This module is additive: it does not import or modify any existing model file.
It provides
  * ``grad_reverse``           -- DANN gradient reversal layer (arXiv:1505.07818)
  * ``Discriminator``          -- 2-layer MLP adversary (H_y subset H_d guaranteed)
  * ``DisentangleNeck``        -- 2048 -> (f_G | f_I | f_S) split + recon anchor
  * ``DisentConfig``           -- plain dataclass carrying all knobs
  * ``add_disent_args``        -- argparse registration helper for config.py
  * schedule helpers           -- DANN lambda anneal + reversed Heaviside floor gate
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Gradient Reversal Layer (DANN, arXiv:1505.07818 Eq.16/17)
# forward: identity.  backward: grad * (-lambda).
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return _GradReverse.apply(x, lambda_)


class Discriminator(nn.Module):
    """2-layer MLP.  Strictly more expressive than the linear task heads, so
    DANN's H_y subset H_d condition holds, but ~10x smaller than DANN's
    1024-1024 default (we only have 2029 training samples)."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DisentConfig:
    method: str = 'bigrl'          # none | split | bigrl | bigrl_g | bigrl_i
    dim_g: int = 320
    dim_i: int = 320
    dim_s: int = 128
    disc_hidden: int = 256
    disc_dropout: float = 0.5
    head_dropout: float = 0.3
    recon_weight: float = 1.0
    adv_detach_backbone: bool = True
    lambda_max: float = 0.3
    anneal_gamma: float = 10.0
    adv_start_epoch: int = 40
    adv_ramp_epochs: int = 40
    gate_margin: float = 0.05
    gate_temp: float = 0.02
    gate_ema: float = 0.9
    disc_lr_mult: float = 10.0
    id_floor: float = 94.0
    backoff_patience: int = 5
    backoff_factor: float = 0.5
    backoff_enabled: bool = True

    @property
    def adv_on_fg(self):
        return self.method in ('bigrl', 'bigrl_g')

    @property
    def adv_on_fi(self):
        return self.method in ('bigrl', 'bigrl_i')

    @property
    def enabled(self):
        return self.method != 'none'


def add_disent_args(parser):
    """Register every new flag.  Call this from config.py (2-line edit)."""
    parser.add_argument('--disent_method', default='none',
                        choices=('none', 'split', 'bigrl', 'bigrl_g', 'bigrl_i'))
    parser.add_argument('--disent_dim_g', default=320, type=int)
    parser.add_argument('--disent_dim_i', default=320, type=int)
    parser.add_argument('--disent_dim_s', default=128, type=int)
    parser.add_argument('--disent_disc_hidden', default=256, type=int)
    parser.add_argument('--disent_disc_dropout', default=0.5, type=float)
    parser.add_argument('--disent_head_dropout', default=0.3, type=float)
    parser.add_argument('--disent_recon_weight', default=1.0, type=float)
    parser.add_argument('--disent_adv_detach_backbone',
                        dest='disent_adv_detach_backbone', action='store_true')
    parser.add_argument('--no_disent_adv_detach_backbone',
                        dest='disent_adv_detach_backbone', action='store_false')
    parser.set_defaults(disent_adv_detach_backbone=True)
    parser.add_argument('--disent_lambda_max', default=0.3, type=float)
    parser.add_argument('--disent_anneal_gamma', default=10.0, type=float)
    parser.add_argument('--disent_adv_start_epoch', default=40, type=int)
    parser.add_argument('--disent_adv_ramp_epochs', default=40, type=int)
    parser.add_argument('--disent_gate_margin', default=0.05, type=float)
    parser.add_argument('--disent_gate_temp', default=0.02, type=float)
    parser.add_argument('--disent_gate_ema', default=0.9, type=float)
    parser.add_argument('--disent_disc_lr_mult', default=10.0, type=float)
    parser.add_argument('--disent_id_floor', default=94.0, type=float)
    parser.add_argument('--disent_backoff_patience', default=5, type=int)
    parser.add_argument('--disent_backoff_factor', default=0.5, type=float)
    parser.add_argument('--disent_backoff', dest='disent_backoff', action='store_true')
    parser.add_argument('--no_disent_backoff', dest='disent_backoff', action='store_false')
    parser.set_defaults(disent_backoff=True)
    parser.add_argument('--disent_run_suffix', default='')
    return parser


def config_from_args(args):
    return DisentConfig(
        method=getattr(args, 'disent_method', 'none'),
        dim_g=getattr(args, 'disent_dim_g', 320),
        dim_i=getattr(args, 'disent_dim_i', 320),
        dim_s=getattr(args, 'disent_dim_s', 128),
        disc_hidden=getattr(args, 'disent_disc_hidden', 256),
        disc_dropout=getattr(args, 'disent_disc_dropout', 0.5),
        head_dropout=getattr(args, 'disent_head_dropout', 0.3),
        recon_weight=getattr(args, 'disent_recon_weight', 1.0),
        adv_detach_backbone=getattr(args, 'disent_adv_detach_backbone', True),
        lambda_max=getattr(args, 'disent_lambda_max', 0.3),
        anneal_gamma=getattr(args, 'disent_anneal_gamma', 10.0),
        adv_start_epoch=getattr(args, 'disent_adv_start_epoch', 40),
        adv_ramp_epochs=getattr(args, 'disent_adv_ramp_epochs', 40),
        gate_margin=getattr(args, 'disent_gate_margin', 0.05),
        gate_temp=getattr(args, 'disent_gate_temp', 0.02),
        gate_ema=getattr(args, 'disent_gate_ema', 0.9),
        disc_lr_mult=getattr(args, 'disent_disc_lr_mult', 10.0),
        id_floor=getattr(args, 'disent_id_floor', 94.0),
        backoff_patience=getattr(args, 'disent_backoff_patience', 5),
        backoff_factor=getattr(args, 'disent_backoff_factor', 0.5),
        backoff_enabled=getattr(args, 'disent_backoff', True),
    )


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def dann_anneal(p, gamma=10.0):
    """DANN Eq: 2/(1+exp(-gamma*p)) - 1, p in [0,1]."""
    p = min(max(float(p), 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def scheduled_lambda(epoch, cfg):
    if not cfg.enabled or epoch < cfg.adv_start_epoch:
        return 0.0
    span = max(1, cfg.adv_ramp_epochs)
    p = (epoch - cfg.adv_start_epoch) / span
    return cfg.lambda_max * dann_anneal(p, cfg.anneal_gamma)


def floor_gate(acc, num_classes, margin, temp):
    """Reversed Heaviside gate (ImmCOGNITO ceiling gate, flipped).

    ~1 while the adversary can still read the nuisance out of the private
    block, ~0 once it has been pushed to the 1/num_classes chance floor."""
    floor = 1.0 / float(num_classes) + margin
    return 1.0 / (1.0 + math.exp(-(acc - floor) / max(temp, 1e-6)))


class LambdaController:
    """Closed-loop guard on the delivered identity accuracy.

    Not from any of the four Route-A papers -- this is an addition (see design
    notes).  lambda_scale only ever decreases, so the controller cannot inject
    oscillation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.scale = 1.0
        self.bad = 0

    def step(self, identity_acc):
        if not (self.cfg.enabled and self.cfg.backoff_enabled):
            return self.scale
        if identity_acc < self.cfg.id_floor:
            self.bad += 1
            if self.bad >= self.cfg.backoff_patience:
                self.scale *= self.cfg.backoff_factor
                self.bad = 0
        else:
            self.bad = 0
        return self.scale


# --------------------------------------------------------------------------- #
# The neck
# --------------------------------------------------------------------------- #
class DisentangleNeck(nn.Module):
    """2048 -> LayerNorm-ReLU bottleneck -> [f_G | f_I | f_S].

    gesture head reads  cat(f_G, f_S)
    identity head reads cat(f_I, f_S)
    D_I (identity adversary) reads GRL(f_G)   -> pushes identity OUT of f_G
    D_G (gesture  adversary) reads GRL(f_I)   -> pushes gesture  OUT of f_I
    recon head reconstructs the frozen 2048-d pooled feature from the full z,
    which forbids deleting information (only redistributing it).
    """

    def __init__(self, cfg, feat_dim, num_gesture, num_identity,
                 has_gesture, has_identity):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = feat_dim
        self.dim_g, self.dim_i, self.dim_s = cfg.dim_g, cfg.dim_i, cfg.dim_s
        self.z_dim = self.dim_g + self.dim_i + self.dim_s

        self.proj = nn.Linear(feat_dim, self.z_dim)
        self.norm = nn.LayerNorm(self.z_dim)
        self.act = nn.ReLU(inplace=True)
        self.head_drop = nn.Dropout(cfg.head_dropout)

        self.recon = nn.Linear(self.z_dim, feat_dim) if cfg.recon_weight > 0 else None

        both = has_gesture and has_identity
        self.disc_identity = (
            Discriminator(self.dim_g, cfg.disc_hidden, num_identity, cfg.disc_dropout)
            if (both and cfg.adv_on_fg) else None)
        self.disc_gesture = (
            Discriminator(self.dim_i, cfg.disc_hidden, num_gesture, cfg.disc_dropout)
            if (both and cfg.adv_on_fi) else None)

        # set from the training loop every epoch
        self.lambda_fg = 0.0
        self.lambda_fi = 0.0

    # -- dims the task heads must be built with -------------------------- #
    @property
    def gesture_in_dim(self):
        return self.dim_g + self.dim_s

    @property
    def identity_in_dim(self):
        return self.dim_i + self.dim_s

    def set_lambda(self, lambda_fg, lambda_fi):
        self.lambda_fg = float(lambda_fg)
        self.lambda_fi = float(lambda_fi)

    def encode(self, feat):
        return self.act(self.norm(self.proj(feat)))

    def split(self, z):
        a = self.dim_g
        b = self.dim_g + self.dim_i
        return z[:, :a], z[:, a:b], z[:, b:]

    def discriminator_parameters(self):
        mods = [m for m in (self.disc_identity, self.disc_gesture) if m is not None]
        for m in mods:
            for p in m.parameters():
                yield p

    def forward(self, feat):
        z = self.encode(feat)
        f_g, f_i, f_s = self.split(z)
        out = {
            'z': z, 'f_g': f_g, 'f_i': f_i, 'f_s': f_s,
            'in_gesture': self.head_drop(torch.cat([f_g, f_s], dim=1)),
            'in_identity': self.head_drop(torch.cat([f_i, f_s], dim=1)),
            'adv_identity': None, 'adv_gesture': None,
            'recon': None, 'recon_target': None,
        }
        if self.recon is not None:
            out['recon'] = self.recon(z)
            out['recon_target'] = feat.detach()

        if self.disc_identity is not None or self.disc_gesture is not None:
            if self.cfg.adv_detach_backbone:
                # LayerNorm/ReLU/Linear only -> no running stats, safe to call
                # twice.  This keeps every adversarial gradient inside the
                # bottleneck projection; the backbone never sees it.
                z_adv = self.encode(feat.detach())
                g_adv, i_adv, _ = self.split(z_adv)
            else:
                g_adv, i_adv = f_g, f_i
            if self.disc_identity is not None:
                out['adv_identity'] = self.disc_identity(
                    grad_reverse(g_adv, self.lambda_fg))
            if self.disc_gesture is not None:
                out['adv_gesture'] = self.disc_gesture(
                    grad_reverse(i_adv, self.lambda_fi))
        return out
