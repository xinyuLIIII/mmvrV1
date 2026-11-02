from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_fixture(root: Path):
    # root 模拟 data/；kpt 在 root/mmVR/kpt_output
    kpt_dir = root / 'mmVR' / 'kpt_output'
    kpt_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        # name, person, scene, gesture, trial
        ('01_01_01_03', '01', '01', '01', '03'),  # 8人保留, train区块, 单手
        ('01_01_01_10', '01', '01', '01', '10'),  # test区块, 单手
        ('01_01_01_07', '01', '01', '01', '07'),  # gap, 应丢弃
        ('03_01_08_04', '03', '01', '08', '04'),  # train区块, 双手手势(08)
        ('02_01_01_03', '02', '01', '01', '03'),  # 删4人之一, 应丢弃
    ]
    lines = [','.join(r) + '\n' for r in rows]
    (root / 'train_list.txt').write_text(''.join(lines), encoding='utf-8')
    (root / 'eval_list.txt').write_text('', encoding='utf-8')
    for name, _p, _s, g, _t in rows:
        arr = np.zeros((30, 42, 3), np.float32) if g == '08' else np.zeros((30, 21, 3), np.float32)
        np.save(kpt_dir / (name + '.npy'), arr)


def test_train_mode_keeps_only_train_trials_of_8_persons(tmp_path):
    from dataset.dataset_identity import MultiTaskDataset
    _make_fixture(tmp_path)
    ds = MultiTaskDataset(str(tmp_path / 'mmVR'), True)
    # 保留: 01_01_01_03, 03_01_08_04 (train区块 & 非删除人)
    # 丢弃: _10(test), _07(gap), 02_(删4人)
    assert len(ds) == 2


def test_test_mode_keeps_only_test_trials(tmp_path):
    from dataset.dataset_identity import MultiTaskDataset
    _make_fixture(tmp_path)
    ds = MultiTaskDataset(str(tmp_path / 'mmVR'), False)
    assert len(ds) == 1  # 只有 01_01_01_10


def test_getitem_returns_padded_shape_and_two_labels(tmp_path):
    from dataset.dataset_identity import MultiTaskDataset
    _make_fixture(tmp_path)
    ds = MultiTaskDataset(str(tmp_path / 'mmVR'), True)
    kpt, g_label, id_label = ds[0]
    assert kpt.shape == (30, 42, 3)
    assert isinstance(g_label, int) and 0 <= g_label <= 7
    assert isinstance(id_label, int) and 0 <= id_label <= 7


def test_single_hand_pads_on_left_regardless_of_person(tmp_path):
    from dataset.dataset_identity import MultiTaskDataset
    _make_fixture(tmp_path)
    ds = MultiTaskDataset(str(tmp_path / 'mmVR'), True)
    # 找到单手样本 01_01_01_03 (gesture 01 -> label 0, 单手)
    for i in range(len(ds)):
        kpt, g, _id = ds[i]
        if g == 0:
            # 左侧应为补零（前21个关节全 0），右侧为真实手
            assert torch.all(kpt[:, :21, :] == 0)
            break
    else:
        raise AssertionError('未找到单手样本')


def test_loso_split_holds_out_person_and_uses_seen_val_block(tmp_path):
    from dataset.dataset_identity import build_loso_datasets
    _make_fixture(tmp_path)
    train_ds, val_ds, heldout_ds = build_loso_datasets(str(tmp_path / 'mmVR'), '01')
    # 留出者 01 的全部 trial（含 gap 区块 07）都进 heldout，且只在 heldout 里
    assert sorted(heldout_ds.names) == ['01_01_01_03', '01_01_01_07', '01_01_01_10']
    assert train_ds.names == ['03_01_08_04']
    assert val_ds.names == []
    assert not set(heldout_ds.names) & (set(train_ds.names) | set(val_ds.names))
    # 删 4 人仍被丢弃，不能作为留出者
    import pytest
    with pytest.raises(ValueError):
        build_loso_datasets(str(tmp_path / 'mmVR'), '02')
