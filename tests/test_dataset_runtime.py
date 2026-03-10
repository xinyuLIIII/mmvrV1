import pathlib
import sys

import h5py
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _create_dataset_fixture(root):
    for dirname in ('mmwave', 'imu', 'kpt_gt'):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    (root / 'train_list.txt').write_text('01_02_03,person,scene,2\n', encoding='utf-8')
    (root / 'eval_list.txt').write_text('01_02_03,person,scene,2\n', encoding='utf-8')

    mmwave = np.arange(4 * 5 * 40, dtype=np.float32).reshape(4, 5, 40)
    with h5py.File(root / 'mmwave' / '01_02_03.mat', 'w') as handle:
        handle.create_dataset('data', data=mmwave)

    imu = np.arange(40 * 6, dtype=np.float32).reshape(40, 6)
    np.save(root / 'imu' / '01_02_03.npy', imu)

    kpt = np.arange(40 * 21 * 3, dtype=np.float32).reshape(40, 21, 3)
    np.save(root / 'kpt_gt' / '01_02_03.npy', kpt)


def test_hm_dataset_loads_preprocessed_sample(tmp_path):
    from dataset.datasets import HM_Dataset

    _create_dataset_fixture(tmp_path)
    dataset = HM_Dataset(str(tmp_path), True)

    mmwave, imu, target = dataset[0]

    expected_frames = dataset.get_frame_indices(40)
    assert mmwave.shape == (30, 5, 4)
    assert imu.shape == (30, 6)
    assert target['kpt'].shape == (30, 1, 63)
    assert target['kpt_cls'].shape == (30, 1, 1)
    assert target['filename'].tolist() == [1, 2, 3]
    assert target['label'].item() == 1
    assert mmwave.dtype == torch.float32
    assert torch.equal(imu[:, 0], torch.from_numpy(np.arange(40, dtype=np.float32)[expected_frames] * 6))
