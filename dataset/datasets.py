import logging
import os

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)

class HM_Dataset(Dataset):
    def __init__(self, dataset_root, mode):
        super(HM_Dataset, self).__init__()
        self.data_root = dataset_root
        self.is_train = mode
        if self.is_train:
            self.file = os.path.join(self.data_root, 'train_list.txt')
        else:
            self.file = os.path.join(self.data_root, 'eval_list.txt')
        self.path = self.data_root
        self.mmwave_dir = os.path.join(self.path, 'mmwave')
        self.imu_dir = os.path.join(self.path, 'imu')
        self.kpt_dir = os.path.join(self.path, 'kpt_gt')
        self.target_num_frames = 30
        self.data = {
            'file_name': list(),
            'person_id': list(),
            'scene': list(),
            'label': list(),
            'filename_tensor': list(),
        }
        self._kpt_cls_cache = {}
        self._frame_index_cache = {}
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
        with h5py.File(os.path.join(self.mmwave_dir, filename + '.mat'), "r") as handle:
            frame_indices = self.get_frame_indices(handle["data"].shape[-1])
            mmwave_data = np.asarray(handle["data"][:, :, frame_indices], dtype=np.float32)
        mmwave_data = np.ascontiguousarray(mmwave_data.transpose((2, 1, 0)))
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
            if frame_count <= 0:
                raise ValueError('frame_count must be positive.')
            self._frame_index_cache[frame_count] = np.linspace(
                0,
                frame_count - 1,
                self.target_num_frames,
                dtype=np.int64,
            )
        return self._frame_index_cache[frame_count]

def build_dataset(dataset_root, mode):
    return HM_Dataset(dataset_root, mode)
