"""Disentangled dual-task model.

Additive: does NOT touch models/multitask_resnet.py, models/ResNet.py,
train_cls.py or dataset/dataset_kpt.py.  When cfg.method == 'none' this class
is numerically identical to MultiTaskResNet (same modules, same shapes), so the
existing baselines stay reproducible from this file too.
"""

import torch
import torch.nn as nn

from models.ResNet import resnet50
from models.disentangle import DisentangleNeck, DisentConfig


class DisentMultiTaskResNet(nn.Module):
    def __init__(self, task, num_gesture=8, num_identity=8, cfg=None):
        super().__init__()
        if task not in ('gesture', 'identity', 'dual'):
            raise ValueError(f'unknown task: {task}')
        self.task = task
        self.cfg = cfg if cfg is not None else DisentConfig(method='none')
        self.backbone = resnet50(num_classes=num_gesture, include_top=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = 2048

        has_g = task in ('gesture', 'dual')
        has_i = task in ('identity', 'dual')

        if self.cfg.enabled:
            self.neck = DisentangleNeck(self.cfg, self.feat_dim,
                                        num_gesture, num_identity, has_g, has_i)
            g_in, i_in = self.neck.gesture_in_dim, self.neck.identity_in_dim
        else:
            self.neck = None
            g_in = i_in = self.feat_dim

        self.head_gesture = nn.Linear(g_in, num_gesture) if has_g else None
        self.head_identity = nn.Linear(i_in, num_identity) if has_i else None

    # ------------------------------------------------------------------ #
    def backbone_parameters(self):
        return self.backbone.parameters()

    def discriminator_parameters(self):
        if self.neck is None:
            return iter(())
        return self.neck.discriminator_parameters()

    def main_parameters(self):
        disc = {id(p) for p in self.discriminator_parameters()}
        for p in self.parameters():
            if id(p) not in disc:
                yield p

    def set_lambda(self, lambda_fg, lambda_fi):
        if self.neck is not None:
            self.neck.set_lambda(lambda_fg, lambda_fi)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        feat = self.backbone(x)              # (B, 2048, 2, 1)
        feat = self.avgpool(feat)            # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)        # (B, 2048)

        out = {'gesture': None, 'identity': None, 'feat': feat,
               'adv_identity': None, 'adv_gesture': None,
               'recon': None, 'recon_target': None,
               'f_g': None, 'f_i': None, 'f_s': None}

        if self.neck is None:
            g_in = i_in = feat
        else:
            nk = self.neck(feat)
            g_in, i_in = nk['in_gesture'], nk['in_identity']
            out.update({k: nk[k] for k in
                        ('adv_identity', 'adv_gesture', 'recon',
                         'recon_target', 'f_g', 'f_i', 'f_s')})

        if self.head_gesture is not None:
            out['gesture'] = self.head_gesture(g_in)
        if self.head_identity is not None:
            out['identity'] = self.head_identity(i_in)
        return out


def build_disent_model(task, num_gesture=8, num_identity=8, cfg=None):
    return DisentMultiTaskResNet(task, num_gesture=num_gesture,
                                 num_identity=num_identity, cfg=cfg)
