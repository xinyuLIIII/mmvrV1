import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_os_cfar_soft_subtract_suppresses_background_and_keeps_peak():
    from utils.cfar import apply_mmwave_cfar

    frames = np.ones((1, 8, 8), dtype=np.float32)
    frames[0, 3, 3] = 12.0

    filtered = apply_mmwave_cfar(
        frames,
        mode='os2d',
        guard_cells=0,
        training_cells=1,
        rank_ratio=0.75,
        pfa=0.2,
        soft_mode='subtract',
        split_halves=False,
    )

    assert filtered.shape == frames.shape
    assert filtered.dtype == np.float32
    assert filtered[0, 3, 3] > 8.0
    assert filtered[0, 0, 0] == 0.0


def test_os_cfar_split_halves_avoids_cross_half_contamination():
    from utils.cfar import apply_mmwave_cfar

    frames = np.ones((1, 8, 8), dtype=np.float32)
    frames[0, 3, 3] = 12.0
    frames[0, 4:, :] = 20.0

    split_filtered = apply_mmwave_cfar(
        frames,
        mode='os2d',
        guard_cells=0,
        training_cells=1,
        rank_ratio=0.75,
        pfa=0.2,
        soft_mode='subtract',
        split_halves=True,
    )
    unsplit_filtered = apply_mmwave_cfar(
        frames,
        mode='os2d',
        guard_cells=0,
        training_cells=1,
        rank_ratio=0.75,
        pfa=0.2,
        soft_mode='subtract',
        split_halves=False,
    )

    assert split_filtered[0, 3, 3] > 8.0
    assert unsplit_filtered[0, 3, 3] < split_filtered[0, 3, 3]
