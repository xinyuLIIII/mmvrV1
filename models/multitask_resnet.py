import torch
import torch.nn as nn

from models.ResNet import resnet50

VALID_TASKS = ('gesture', 'identity', 'dual')


class MultiTaskResNet(nn.Module):
    """共享 ResNet50 backbone + 手势头/身份头（硬参数共享）。

    task='gesture'/'identity' 只建对应头；'dual' 两个头都建。
    forward 返回 {'gesture': logits|None, 'identity': logits|None}。
    """

    def __init__(self, task, num_gesture=8, num_identity=8):
        super().__init__()
        if task not in VALID_TASKS:
            raise ValueError(f'task must be one of {VALID_TASKS}, got {task!r}')
        self.task = task
        self.backbone = resnet50(include_top=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 512 * 4  # Bottleneck.expansion=4 -> 2048

        self.head_gesture = None
        self.head_identity = None
        if task in ('gesture', 'dual'):
            self.head_gesture = nn.Linear(feat_dim, num_gesture)
        if task in ('identity', 'dual'):
            self.head_identity = nn.Linear(feat_dim, num_identity)

    def forward(self, x):
        feat = self.backbone(x)               # (B, 2048, h, w)
        feat = self.avgpool(feat)             # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)         # (B, 2048)
        out = {'gesture': None, 'identity': None}
        if self.head_gesture is not None:
            out['gesture'] = self.head_gesture(feat)
        if self.head_identity is not None:
            out['identity'] = self.head_identity(feat)
        return out


def build_multitask_model(task, num_gesture=8, num_identity=8):
    return MultiTaskResNet(task, num_gesture=num_gesture, num_identity=num_identity)

