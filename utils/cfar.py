import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _validate_window(guard_cells, training_cells):
    if guard_cells < 0:
        raise ValueError('guard_cells must be non-negative.')
    if training_cells < 1:
        raise ValueError('training_cells must be at least 1.')


def _build_training_mask(guard_cells, training_cells):
    radius = guard_cells + training_cells
    window_size = radius * 2 + 1
    mask = np.ones((window_size, window_size), dtype=bool)
    center = radius
    mask[
        center - guard_cells:center + guard_cells + 1,
        center - guard_cells:center + guard_cells + 1,
    ] = False
    if not mask.any():
        raise ValueError('CFAR window must include at least one training cell.')
    return mask


def _resolve_rank_index(num_training_cells, rank_ratio):
    if not 0.0 < rank_ratio <= 1.0:
        raise ValueError('rank_ratio must be within (0, 1].')
    return max(0, min(num_training_cells - 1, int(math.ceil(num_training_cells * rank_ratio)) - 1))


def _resolve_threshold_scale(num_training_cells, pfa):
    if not 0.0 < pfa < 1.0:
        raise ValueError('pfa must be within (0, 1).')
    return num_training_cells * (pfa ** (-1.0 / num_training_cells) - 1.0)


def os_cfar_2d(frame, guard_cells=1, training_cells=4, rank_ratio=0.75, pfa=1e-3, soft_mode='subtract'):
    _validate_window(guard_cells, training_cells)
    if soft_mode not in ('subtract', 'mask'):
        raise ValueError("soft_mode must be 'subtract' or 'mask'.")

    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError('os_cfar_2d expects a 2D frame.')

    training_mask = _build_training_mask(guard_cells, training_cells)
    num_training_cells = int(training_mask.sum())
    rank_index = _resolve_rank_index(num_training_cells, rank_ratio)
    threshold_scale = _resolve_threshold_scale(num_training_cells, pfa)

    radius = guard_cells + training_cells
    padded = np.pad(frame, ((radius, radius), (radius, radius)), mode='edge')
    windows = sliding_window_view(padded, training_mask.shape)
    training_values = windows[..., training_mask]
    ordered = np.partition(training_values, rank_index, axis=-1)
    noise_floor = ordered[..., rank_index]
    threshold = noise_floor * threshold_scale

    if soft_mode == 'mask':
        return frame * (frame > threshold)
    return np.maximum(frame - threshold, 0.0).astype(np.float32, copy=False)


def apply_mmwave_cfar(frame_sequence, mode='none', guard_cells=1, training_cells=4, rank_ratio=0.75, pfa=1e-3,
                      soft_mode='subtract', split_halves=True):
    frame_sequence = np.asarray(frame_sequence, dtype=np.float32)
    if frame_sequence.ndim != 3:
        raise ValueError('apply_mmwave_cfar expects frames shaped as (frames, height, width).')
    if mode == 'none':
        return frame_sequence
    if mode != 'os2d':
        raise ValueError(f'Unsupported CFAR mode: {mode}')

    filtered = np.empty_like(frame_sequence)
    for index, frame in enumerate(frame_sequence):
        if split_halves and frame.shape[0] % 2 == 0:
            midpoint = frame.shape[0] // 2
            top = os_cfar_2d(
                frame[:midpoint],
                guard_cells=guard_cells,
                training_cells=training_cells,
                rank_ratio=rank_ratio,
                pfa=pfa,
                soft_mode=soft_mode,
            )
            bottom = os_cfar_2d(
                frame[midpoint:],
                guard_cells=guard_cells,
                training_cells=training_cells,
                rank_ratio=rank_ratio,
                pfa=pfa,
                soft_mode=soft_mode,
            )
            filtered[index] = np.concatenate((top, bottom), axis=0)
        else:
            filtered[index] = os_cfar_2d(
                frame,
                guard_cells=guard_cells,
                training_cells=training_cells,
                rank_ratio=rank_ratio,
                pfa=pfa,
                soft_mode=soft_mode,
            )
    return filtered
