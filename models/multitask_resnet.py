import torch
import torch.nn as nn

from models.ResNet import resnet50


class MultiTaskResNet(nn.Module):
    def __init__(self, task, num_gesture=8, num_identity=8):
        super().__init__()
        if task not in ('gesture', 'identity', 'dual'):
            raise ValueError(f'unknown task: {task}')
        self.task = task
        self.backbone = resnet50(num_classes=num_gesture, include_top=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 2048  # resnet50 layer4 输出通道
        self.head_gesture = nn.Linear(feat_dim, num_gesture) if task in ('gesture', 'dual') else None
        self.head_identity = nn.Linear(feat_dim, num_identity) if task in ('identity', 'dual') else None

    def forward(self, x):
        feat = self.backbone(x)              # (B, 2048, h, w)
        feat = self.avgpool(feat)            # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)        # (B, 2048)
        return {
            'gesture': self.head_gesture(feat) if self.head_gesture is not None else None,
            'identity': self.head_identity(feat) if self.head_identity is not None else None,
        }


def build_multitask_model(task, num_gesture=8, num_identity=8):
    return MultiTaskResNet(task, num_gesture=num_gesture, num_identity=num_identity)
