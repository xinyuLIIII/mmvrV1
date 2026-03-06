from pathlib import Path

from utils.kpt_overfitting_viz import (
    build_overfitting_summary,
    load_log_metrics,
    load_mem_stats,
    plot_overfitting_dashboard,
)


def test_loaders_and_summary(tmp_path):
    mem_dir = tmp_path / "mem_stats"
    mem_dir.mkdir()

    (mem_dir / "pose_memory_train_epoch_1.json").write_text(
        '{"count": 4, "mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "abs_mean": 0.5, "l1": 2.0, "l2": 2.0, "sparsity": 0.1}'
    )
    (mem_dir / "pose_memory_test_epoch_1.json").write_text(
        '{"count": 4, "mean": 0.1, "std": 1.1, "min": -1.0, "max": 1.0, "abs_mean": 0.55, "l1": 2.2, "l2": 2.2, "sparsity": 0.2}'
    )
    (mem_dir / "pose_memory_train_epoch_2.json").write_text(
        '{"count": 4, "mean": 0.0, "std": 0.9, "min": -1.0, "max": 1.0, "abs_mean": 0.45, "l1": 1.8, "l2": 1.8, "sparsity": 0.15}'
    )
    (mem_dir / "pose_memory_test_epoch_2.json").write_text(
        '{"count": 4, "mean": 0.2, "std": 1.2, "min": -1.0, "max": 1.0, "abs_mean": 0.60, "l1": 2.4, "l2": 2.4, "sparsity": 0.25}'
    )

    train_log = tmp_path / "train_train.log"
    train_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 0.5, 'MPJPE': 100.0, 'MPJDLE': 50.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 2, 'loss_kpt': 0.2, 'MPJPE': 60.0, 'MPJDLE': 30.0}\n"
    )
    test_log = tmp_path / "train_test.log"
    test_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 0.6, 'MPJPE': 110.0, 'MPJDLE': 55.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 2, 'loss_kpt': 0.55, 'MPJPE': 108.0, 'MPJDLE': 54.0}\n"
    )

    mem_df = load_mem_stats(mem_dir)
    log_df = load_log_metrics(train_log, test_log)
    summary = build_overfitting_summary(log_df)

    assert list(mem_df["epoch"]) == [1, 1, 2, 2]
    assert set(mem_df["split"]) == {"train", "test"}
    assert "l1_per_elem" in mem_df.columns
    assert "l2_per_elem" in mem_df.columns
    assert list(log_df["epoch"]) == [1, 2, 1, 2]
    assert summary["final_epoch"] == 2
    assert summary["best_test_mpjpe_epoch"] == 2
    assert abs(summary["final_mpjpe_gap"] - 48.0) < 1e-9
    assert abs(summary["final_loss_kpt_gap"] - 0.35) < 1e-9


def test_plot_overfitting_dashboard_writes_png(tmp_path):
    mem_dir = Path("experiments/mem_stats")
    train_log = Path("experiments/train_train.log")
    test_log = Path("experiments/train_test.log")

    mem_df = load_mem_stats(mem_dir)
    log_df = load_log_metrics(train_log, test_log)
    summary = build_overfitting_summary(log_df)
    output_path = tmp_path / "dashboard.png"

    plot_overfitting_dashboard(mem_df, log_df, output_path, summary)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
