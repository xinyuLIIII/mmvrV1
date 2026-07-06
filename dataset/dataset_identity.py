import os

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data.dataset import Dataset

NUM_GESTURE = 8
NUM_IDENTITY = 8
EXCLUDED_PERSONS = ('02', '04', '05', '12')


def _load_all_samples(dataset_root):
    """合并 train_list + eval_list 全部样本，过滤 EXCLUDED_PERSONS。

    name = person_scene_gesture_repeat（如 11_03_07_11）。
    返回 list[(name, person_id, scene, gesture_label)]，gesture_label 0-7。
    """
    raw = []
    for list_name in ('train_list.txt', 'eval_list.txt'):
        path = os.path.join(dataset_root, list_name)
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                person_id = parts[1]
                if person_id in EXCLUDED_PERSONS:
                    continue
                name = parts[0]
                scene = parts[2]
                gesture_label = int(parts[3]) - 1
                raw.append((name, person_id, scene, gesture_label))
    return raw


def _split_indices(raw, test_size, seed):
    """按 (person, scene, gesture) 整组切分，同组 repeat 不跨 train/test。

    GroupShuffleSplit 单次切分（不分折）。切完断言 train 覆盖全部身份与手势、
    且 train/test 无 group 交集，否则 raise 让调用方换 seed。
    """
    groups = [f'{p}_{s}_{g}' for _, p, s, g in raw]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    train_idx, test_idx = next(splitter.split(raw, groups=groups))

    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f'group leak: {len(overlap)} groups appear in both splits')

    persons = {p for _, p, _, _ in raw}
    gestures = {g for _, _, _, g in raw}
    train_persons = {raw[i][1] for i in train_idx}
    train_gestures = {raw[i][3] for i in train_idx}
    if train_persons != persons:
        raise RuntimeError(
            f'train missing identities {persons - train_persons}; '
            f'change --seed or --test_size')
    if train_gestures != gestures:
        raise RuntimeError(
            f'train missing gestures {gestures - train_gestures}; '
            f'change --seed or --test_size')
    return train_idx, test_idx


class MultiTaskDataset(Dataset):
    """骨架关键点 -> (kpt[30,42,3], gesture_label, identity_label).

    - 过滤 EXCLUDED_PERSONS（消除补零泄漏），剩 8 人。
    - 单手手势(label 0-5)统一左侧补零，不看 people_id。
    - 按 (person,scene,gesture) 整组切分 train/test，消除 repeat 近邻泄漏。
    - 同时返回手势与身份两个标签，训练脚本按 --task 取用。
    """

    def __init__(self, dataset_root, mode, test_size=0.2, seed=42):
        super().__init__()
        self.path = dataset_root
        self.is_train = bool(mode)

        raw = _load_all_samples(dataset_root)

        # 身份映射在过滤后、切分前基于全量 8 人构建，train/test 保持一致。
        persons = sorted({person_id for _, person_id, _, _ in raw})
        self.id_map = {person_id: idx for idx, person_id in enumerate(persons)}

        train_idx, test_idx = _split_indices(raw, test_size, seed)
        keep = train_idx if self.is_train else test_idx

        self.data = {'file_name': [], 'gesture': [], 'identity': []}
        for i in keep:
            name, person_id, _, gesture_label = raw[i]
            self.data['file_name'].append(name)
            self.data['gesture'].append(gesture_label)
            self.data['identity'].append(self.id_map[person_id])

    def __len__(self):
        return len(self.data['file_name'])

    def load_keypoint(self, filename):
        arr = np.load(os.path.join(self.path, 'kpt_output', filename + '.npy'))
        return torch.from_numpy(arr).float()

    def __getitem__(self, index):
        name = self.data['file_name'][index]
        gesture_label = self.data['gesture'][index]
        identity_label = self.data['identity'][index]
        keypoint_data = self.load_keypoint(name)
        if gesture_label in (0, 1, 2, 3, 4, 5):
            padding = torch.zeros(30, 21, 3)
            keypoint_data = torch.cat((padding, keypoint_data), dim=1)
        return keypoint_data, gesture_label, identity_label


def build_dataset(dataset_root, mode, test_size=0.2, seed=42):
    return MultiTaskDataset(dataset_root, mode, test_size=test_size, seed=seed)

