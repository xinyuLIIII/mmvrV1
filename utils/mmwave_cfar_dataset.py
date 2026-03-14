import os
from pathlib import Path

import h5py
import numpy as np

from utils.cfar import apply_mmwave_cfar


TARGET_NUM_FRAMES = 30


def get_uniform_frame_indices(frame_count, target_num_frames=TARGET_NUM_FRAMES):
    if frame_count <= 0:
        raise ValueError('frame_count must be positive.')
    return np.linspace(0, frame_count - 1, target_num_frames, dtype=np.int64)


def load_raw_mmwave_sample(mmwave_path, target_num_frames=TARGET_NUM_FRAMES):
    with h5py.File(mmwave_path, 'r') as handle:
        frame_indices = get_uniform_frame_indices(handle['data'].shape[-1], target_num_frames)
        mmwave_data = np.asarray(handle['data'][:, :, frame_indices], dtype=np.float32)
    return np.ascontiguousarray(mmwave_data.transpose((2, 1, 0)))


def build_cfar_cache_signature(
    mode='os2d',
    guard_cells=1,
    training_cells=4,
    rank_ratio=0.75,
    pfa=1e-3,
    soft_mode='subtract',
    split_halves=True,
    target_num_frames=TARGET_NUM_FRAMES,
):
    if mode != 'os2d':
        raise ValueError(f'Unsupported CFAR cache mode: {mode}')
    split_tag = 'split' if split_halves else 'nosplit'
    return (
        f'{mode}_frames{target_num_frames}_guard{guard_cells}_train{training_cells}'
        f'_rank{rank_ratio:g}_pfa{pfa:g}_{soft_mode}_{split_tag}'
    )


def get_cfar_cache_dir(
    dataset_root,
    mode='os2d',
    guard_cells=1,
    training_cells=4,
    rank_ratio=0.75,
    pfa=1e-3,
    soft_mode='subtract',
    split_halves=True,
    target_num_frames=TARGET_NUM_FRAMES,
):
    signature = build_cfar_cache_signature(
        mode=mode,
        guard_cells=guard_cells,
        training_cells=training_cells,
        rank_ratio=rank_ratio,
        pfa=pfa,
        soft_mode=soft_mode,
        split_halves=split_halves,
        target_num_frames=target_num_frames,
    )
    return os.path.join(dataset_root, 'mmwave_cfar', signature)


def precompute_mmwave_cfar_dataset(
    dataset_root,
    mode='os2d',
    guard_cells=1,
    training_cells=4,
    rank_ratio=0.75,
    pfa=1e-3,
    soft_mode='subtract',
    split_halves=True,
    target_num_frames=TARGET_NUM_FRAMES,
):
    source_dir = Path(dataset_root) / 'mmwave'
    if not source_dir.is_dir():
        raise FileNotFoundError(f'Raw mmwave directory not found: {source_dir}')

    output_dir = Path(
        get_cfar_cache_dir(
            dataset_root,
            mode=mode,
            guard_cells=guard_cells,
            training_cells=training_cells,
            rank_ratio=rank_ratio,
            pfa=pfa,
            soft_mode=soft_mode,
            split_halves=split_halves,
            target_num_frames=target_num_frames,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mmwave_paths = sorted(source_dir.glob('*.mat'))
    if not mmwave_paths:
        raise FileNotFoundError(f'No mmwave .mat files found under {source_dir}')

    for mmwave_path in mmwave_paths:
        sampled = load_raw_mmwave_sample(mmwave_path, target_num_frames=target_num_frames)
        filtered = apply_mmwave_cfar(
            sampled,
            mode=mode,
            guard_cells=guard_cells,
            training_cells=training_cells,
            rank_ratio=rank_ratio,
            pfa=pfa,
            soft_mode=soft_mode,
            split_halves=split_halves,
        )
        np.save(
            output_dir / f'{mmwave_path.stem}.npy',
            np.asarray(filtered, dtype=np.float32),
            allow_pickle=False,
        )

    return output_dir, len(mmwave_paths)
