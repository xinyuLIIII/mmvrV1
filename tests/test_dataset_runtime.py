import pathlib
import runpy
import sys

import h5py
import numpy as np
import torch
from types import SimpleNamespace

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


def test_build_dataset_loads_precomputed_cfar_sample(tmp_path):
    from dataset.datasets import build_dataset
    from utils.mmwave_cfar_dataset import get_cfar_cache_dir, precompute_mmwave_cfar_dataset

    _create_dataset_fixture(tmp_path)
    base_mmwave = np.ones((8, 8, 40), dtype=np.float32)
    base_mmwave[3, 3, :] = 12.0
    with h5py.File(tmp_path / 'mmwave' / '01_02_03.mat', 'w') as handle:
        handle.create_dataset('data', data=base_mmwave)

    precompute_mmwave_cfar_dataset(
        str(tmp_path),
        mode='os2d',
        guard_cells=0,
        training_cells=1,
        rank_ratio=0.75,
        pfa=0.2,
        soft_mode='subtract',
        split_halves=False,
    )
    args = SimpleNamespace(
        cfar_mode='os2d',
        cfar_guard=0,
        cfar_train=1,
        cfar_rank_ratio=0.75,
        cfar_pfa=0.2,
        cfar_soft_mode='subtract',
        cfar_split_halves=False,
    )

    dataset = build_dataset(str(tmp_path), True, args=args)
    mmwave, _, _ = dataset[0]
    cache_dir = pathlib.Path(
        get_cfar_cache_dir(
            str(tmp_path),
            mode='os2d',
            guard_cells=0,
            training_cells=1,
            rank_ratio=0.75,
            pfa=0.2,
            soft_mode='subtract',
            split_halves=False,
        )
    )
    cached_mmwave = np.load(cache_dir / '01_02_03.npy', allow_pickle=False)

    assert mmwave.shape == (30, 8, 8)
    assert mmwave.dtype == torch.float32
    assert torch.equal(mmwave, torch.from_numpy(cached_mmwave))


def test_build_dataset_raises_when_precomputed_cfar_dataset_missing(tmp_path):
    from dataset.datasets import build_dataset

    _create_dataset_fixture(tmp_path)
    args = SimpleNamespace(
        cfar_mode='os2d',
        cfar_guard=0,
        cfar_train=1,
        cfar_rank_ratio=0.75,
        cfar_pfa=0.2,
        cfar_soft_mode='subtract',
        cfar_split_halves=False,
    )

    try:
        build_dataset(str(tmp_path), True, args=args)
    except FileNotFoundError as exc:
        assert 'scripts/precompute_mmwave_cfar.py' in str(exc)
    else:
        raise AssertionError('Expected missing offline CFAR dataset to raise FileNotFoundError.')


def test_precompute_script_processes_entire_mmwave_directory(tmp_path, monkeypatch):
    from utils.mmwave_cfar_dataset import build_cfar_cache_signature

    _create_dataset_fixture(tmp_path)
    extra_mmwave = np.ones((4, 5, 40), dtype=np.float32)
    with h5py.File(tmp_path / 'mmwave' / '99_99_99.mat', 'w') as handle:
        handle.create_dataset('data', data=extra_mmwave)

    script_path = ROOT / 'scripts' / 'precompute_mmwave_cfar.py'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            str(script_path),
            '--dataset_root', str(tmp_path),
            '--cfar_guard', '0',
            '--cfar_train', '1',
            '--cfar_rank_ratio', '0.75',
            '--cfar_pfa', '0.2',
            '--cfar_soft_mode', 'subtract',
            '--no_cfar_split_halves',
        ],
    )

    runpy.run_path(str(script_path), run_name='__main__')

    signature = build_cfar_cache_signature(
        mode='os2d',
        guard_cells=0,
        training_cells=1,
        rank_ratio=0.75,
        pfa=0.2,
        soft_mode='subtract',
        split_halves=False,
    )
    cache_dir = tmp_path / 'mmwave_cfar' / signature
    assert (cache_dir / '01_02_03.npy').exists()
    assert (cache_dir / '99_99_99.npy').exists()
