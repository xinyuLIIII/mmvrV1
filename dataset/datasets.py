import logging
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.mmwave_cfar_dataset import (
    TARGET_NUM_FRAMES,
    get_cfar_cache_dir,
    get_uniform_frame_indices,
    load_raw_mmwave_sample,
)

log = logging.getLogger(__name__)

class HM_Dataset(Dataset):
    def __init__(self, dataset_root, mode, mmwave_source=None):
        super(HM_Dataset, self).__init__()
        self.data_root = dataset_root
        self.is_train = mode
        if self.is_train:
            self.file = os.path.join(self.data_root, 'train_list.txt')
        else:
            self.file = os.path.join(self.data_root, 'eval_list.txt')
        self.path = self.data_root
        self.imu_dir = os.path.join(self.path, 'imu')
        self.kpt_dir = os.path.join(self.path, 'kpt_gt')
        self.target_num_frames = TARGET_NUM_FRAMES
        self.data = {
            'file_name': list(),
            'person_id': list(),
            'scene': list(),
            'label': list(),
            'filename_tensor': list(),
        }
        self._kpt_cls_cache = {}
        self._frame_index_cache = {}
        self.mmwave_source = mmwave_source or {'mode': 'none'}
        self.mmwave_mode = self.mmwave_source.get('mode', 'none')
        if self.mmwave_mode == 'none':
            self.mmwave_dir = os.path.join(self.path, 'mmwave')
        elif self.mmwave_mode == 'os2d':
            self.mmwave_dir = get_cfar_cache_dir(
                self.path,
                mode=self.mmwave_mode,
                guard_cells=self.mmwave_source.get('guard_cells', 1),
                training_cells=self.mmwave_source.get('training_cells', 4),
                rank_ratio=self.mmwave_source.get('rank_ratio', 0.75),
                pfa=self.mmwave_source.get('pfa', 1e-3),
                soft_mode=self.mmwave_source.get('soft_mode', 'subtract'),
                split_halves=self.mmwave_source.get('split_halves', True),
                target_num_frames=self.target_num_frames,
            )
            if not os.path.isdir(self.mmwave_dir):
                raise FileNotFoundError(
                    f'Offline CFAR dataset not found: {self.mmwave_dir}. '
                    'Run scripts/precompute_mmwave_cfar.py with matching CFAR arguments first.'
                )
        else:
            raise ValueError(f'Unsupported mmwave source mode: {self.mmwave_mode}')
        with open(self.file, encoding='utf-8') as file:
            data_list = file.readlines()
        for string in data_list:
            name, person_id, scene, raw_label = string.strip().split(',')[:4]
            label = int(raw_label) - 1
            self.data['file_name'].append(name)
            self.data['person_id'].append(person_id)
            self.data['scene'].append(scene)
            self.data['label'].append(label)
            self.data['filename_tensor'].append(
                torch.tensor([int(num) for num in name.split('_')], dtype=torch.int64)
            )

        log.info("加载完毕HM数据集")

    def __getitem__(self, index):
        data_name = self.data['file_name'][index]
        mmwave_data = self.load_mmwave(data_name)
        imu_data = self.load_imu(data_name)
        keypoint_data = self.load_keypoint(data_name)
        targets = self.get_item_single_frame(keypoint_data, index)
        return mmwave_data, imu_data, targets

    def __len__(self):
        return len(self.data['file_name'])

    def load_mmwave(self, filename):
        if self.mmwave_mode == 'none':
            mmwave_path = os.path.join(self.mmwave_dir, filename + '.mat')
            mmwave_data = load_raw_mmwave_sample(
                mmwave_path,
                target_num_frames=self.target_num_frames,
            )
            return torch.from_numpy(mmwave_data)

        mmwave_path = os.path.join(self.mmwave_dir, filename + '.npy')
        if not os.path.exists(mmwave_path):
            raise FileNotFoundError(
                f'Offline CFAR sample not found: {mmwave_path}. '
                'Regenerate the CFAR dataset with scripts/precompute_mmwave_cfar.py.'
            )
        mmwave_data = np.load(mmwave_path, mmap_mode='r', allow_pickle=False)
        if mmwave_data.ndim != 3 or mmwave_data.shape[0] != self.target_num_frames:
            raise ValueError(
                f'Offline CFAR sample {mmwave_path} has shape {mmwave_data.shape}, '
                f'expected ({self.target_num_frames}, H, W).'
            )
        mmwave_data = np.array(mmwave_data, dtype=np.float32, copy=True, order='C')
        return torch.from_numpy(mmwave_data)

    def load_imu(self, filename):
        imu_data = np.load(
            os.path.join(self.imu_dir, filename + '.npy'),
            mmap_mode='r',
            allow_pickle=False,
        )
        frame_indices = self.get_frame_indices(imu_data.shape[0])
        imu_data = np.ascontiguousarray(imu_data[frame_indices], dtype=np.float32)
        return torch.from_numpy(imu_data)

    def load_keypoint(self, filename):
        keypoint_data = np.load(
            os.path.join(self.kpt_dir, filename + '.npy'),
            mmap_mode='r',
            allow_pickle=False,
        )
        frame_indices = self.get_frame_indices(keypoint_data.shape[0])
        keypoint_data = np.ascontiguousarray(keypoint_data[frame_indices], dtype=np.float32)
        if keypoint_data.shape[1] == 21:
            keypoint_data = keypoint_data.reshape(30, 1, -1)
        else:
            keypoint_data = keypoint_data.reshape(30, 2, -1)
        return torch.from_numpy(keypoint_data)

    def get_item_single_frame(self, keypoint, index):
        NumOfHand = keypoint.shape[1]
        kpt_cls = self.get_cls(NumOfHand)
        target = {'kpt': keypoint, 'kpt_cls': kpt_cls,
                  'label': torch.tensor(self.data['label'][index], dtype=torch.int64),
                  'filename': self.data['filename_tensor'][index]
                }
        return target

    def get_cls(self, numofperson):
        if numofperson not in self._kpt_cls_cache:
            cls_tensor = torch.ones((self.target_num_frames, numofperson, 1), dtype=torch.int64)
            self._kpt_cls_cache[numofperson] = cls_tensor
        return self._kpt_cls_cache[numofperson]

    def get_frame_indices(self, frame_count):
        if frame_count not in self._frame_index_cache:
            self._frame_index_cache[frame_count] = get_uniform_frame_indices(
                frame_count,
                target_num_frames=self.target_num_frames,
            )
        return self._frame_index_cache[frame_count]

def build_dataset(dataset_root, mode, args=None):
    mmwave_source = {
        'mode': getattr(args, 'cfar_mode', 'none'),
        'guard_cells': getattr(args, 'cfar_guard', 1),
        'training_cells': getattr(args, 'cfar_train', 4),
        'rank_ratio': getattr(args, 'cfar_rank_ratio', 0.75),
        'pfa': getattr(args, 'cfar_pfa', 1e-3),
        'soft_mode': getattr(args, 'cfar_soft_mode', 'subtract'),
        'split_halves': getattr(args, 'cfar_split_halves', True),
    }
    return HM_Dataset(dataset_root, mode, mmwave_source=mmwave_source)
