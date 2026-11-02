import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

NUM_GESTURE = 8
NUM_IDENTITY = 8
DROP_PERSONS = {'02', '04', '05', '12'}
TRAIN_TRIALS = {'01', '02', '03', '04', '05', '06', '15', '16', '17', '18', '19', '20'}
GAP_TRIALS = {'07', '08', '13', '14'}
TEST_TRIALS = {'09', '10', '11', '12'}
ALL_TRIALS = TRAIN_TRIALS | GAP_TRIALS | TEST_TRIALS
# 8 人排序后映射到连续 0-7
IDENTITY_MAP = {p: i for i, p in enumerate(sorted(
    {'01', '03', '06', '07', '08', '09', '10', '11'}))}
# 单手手势 (gesture label 0-5) 需补零；双手 (6,7) 原样
SINGLE_HAND_LABELS = {0, 1, 2, 3, 4, 5}


class MultiTaskDataset(Dataset):
    def __init__(self, dataset_root, mode, persons=None, trials=None):
        super().__init__()
        self.kpt_root = os.path.join(dataset_root, 'kpt_output')
        # list 目录：去掉末级 'mmVR'（若存在），否则用 dataset_root 本身
        head, tail = os.path.split(os.path.normpath(dataset_root))
        list_dir = head if tail == 'mmVR' else dataset_root
        if trials is not None:
            self.keep_trials = set(trials)
        else:
            self.keep_trials = TRAIN_TRIALS if mode else TEST_TRIALS
        # persons=None 时保留全部 8 人（原闭集行为）
        self.keep_persons = set(persons) if persons is not None else None

        self.names = []
        self.gesture_labels = []
        self.identity_labels = []
        for list_name in ('train_list.txt', 'eval_list.txt'):
            path = os.path.join(list_dir, list_name)
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cols = line.split(',')
                    name, person_id, trial_id = cols[0], cols[1], cols[4]
                    if person_id in DROP_PERSONS:
                        continue
                    if self.keep_persons is not None and person_id not in self.keep_persons:
                        continue
                    if trial_id not in self.keep_trials:
                        continue
                    self.names.append(name)
                    self.gesture_labels.append(int(cols[3]) - 1)
                    self.identity_labels.append(IDENTITY_MAP[person_id])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        g_label = self.gesture_labels[index]
        id_label = self.identity_labels[index]
        arr = np.load(os.path.join(self.kpt_root, name + '.npy'))
        kpt = torch.from_numpy(arr).float()
        if g_label in SINGLE_HAND_LABELS:
            padding = torch.zeros(30, 21, 3)
            kpt = torch.cat((padding, kpt), dim=1)  # 固定零补左侧
        return kpt, g_label, id_label


def build_dataset(dataset_root, mode):
    return MultiTaskDataset(dataset_root, mode)


def build_loso_datasets(dataset_root, heldout_person):
    """留一人划分：返回 (train, val, heldout)。

    train:   其余 7 人的 TRAIN_TRIALS
    val:     其余 7 人的 TEST_TRIALS（与 train 之间隔着 GAP_TRIALS），只用于选模型 + 闭集身份
    heldout: 留出者的全部 trial，只用于报告跨用户手势准确率，绝不参与选模型
    身份标签仍用 8 类 IDENTITY_MAP，留出者对应的那一类在训练中不出现。
    """
    if heldout_person not in IDENTITY_MAP:
        raise ValueError(f'heldout_person must be one of {sorted(IDENTITY_MAP)}, got {heldout_person!r}')
    seen = set(IDENTITY_MAP) - {heldout_person}
    train_ds = MultiTaskDataset(dataset_root, True, persons=seen)
    val_ds = MultiTaskDataset(dataset_root, False, persons=seen)
    heldout_ds = MultiTaskDataset(dataset_root, False, persons={heldout_person}, trials=ALL_TRIALS)
    return train_ds, val_ds, heldout_ds
