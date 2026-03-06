import ast
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MEM_PATTERN = re.compile(r"pose_memory_(train|test)_epoch_(\d+)\.json$")
LOG_PATTERN = re.compile(r"INFO - (\{.*\})")


def load_mem_stats(mem_stats_dir):
    mem_stats_dir = Path(mem_stats_dir)
    rows = []
    for path in sorted(mem_stats_dir.glob("pose_memory_*_epoch_*.json")):
        match = MEM_PATTERN.match(path.name)
        if not match:
            continue
        split, epoch_text = match.groups()
        payload = json.loads(path.read_text())
        count = payload.get("count", 0) or 0
        row = {
            "epoch": int(epoch_text),
            "split": split,
            **payload,
        }
        row["l1_per_elem"] = payload.get("l1", 0.0) / count if count else 0.0
        row["l2_per_elem"] = payload.get("l2", 0.0) / math.sqrt(count) if count else 0.0
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No pose memory stats found in {mem_stats_dir}")

    frame = pd.DataFrame(rows)
    frame["split"] = pd.Categorical(frame["split"], categories=["train", "test"], ordered=True)
    return frame.sort_values(["epoch", "split"]).reset_index(drop=True)


def _parse_log_file(log_path, split):
    log_path = Path(log_path)
    rows = []
    for line in log_path.read_text().splitlines():
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        payload = ast.literal_eval(match.group(1))
        payload["split"] = split
        rows.append(payload)

    if not rows:
        raise FileNotFoundError(f"No epoch metrics found in {log_path}")

    frame = pd.DataFrame(rows)
    return frame.sort_values("epoch").reset_index(drop=True)


def load_log_metrics(train_log_path, test_log_path):
    train_df = _parse_log_file(train_log_path, "train")
    test_df = _parse_log_file(test_log_path, "test")
    return pd.concat([train_df, test_df], ignore_index=True)


def build_overfitting_summary(log_df):
    metrics = ["loss_kpt", "MPJPE", "MPJDLE"]
    pivot = log_df.pivot(index="epoch", columns="split", values=metrics)
    pivot = pivot.sort_index()
    common_epochs = [int(epoch) for epoch in pivot.index.tolist()]
    if not common_epochs:
        raise ValueError("No overlapping epochs between train and test logs")

    final_epoch = common_epochs[-1]
    first_epoch = common_epochs[0]
    summary = {
        "first_epoch": first_epoch,
        "final_epoch": final_epoch,
    }

    for metric in metrics:
        train_first = float(pivot[(metric, "train")].loc[first_epoch])
        train_final = float(pivot[(metric, "train")].loc[final_epoch])
        test_first = float(pivot[(metric, "test")].loc[first_epoch])
        test_final = float(pivot[(metric, "test")].loc[final_epoch])
        summary[f"final_{metric.lower()}_gap"] = test_final - train_final
        summary[f"train_{metric.lower()}_drop"] = train_first - train_final
        summary[f"test_{metric.lower()}_drop"] = test_first - test_final

    test_only = log_df[log_df["split"] == "test"].sort_values("epoch")
    best_test_row = test_only.loc[test_only["MPJPE"].idxmin()]
    summary["best_test_mpjpe_epoch"] = int(best_test_row["epoch"])
    summary["best_test_mpjpe"] = float(best_test_row["MPJPE"])
    summary["best_test_mpjdle"] = float(best_test_row["MPJDLE"])
    return summary


def _plot_metric(ax, frame, metric, title, ylabel):
    for split, color in (("train", "tab:blue"), ("test", "tab:orange")):
        split_df = frame[frame["split"] == split].sort_values("epoch")
        ax.plot(split_df["epoch"], split_df[metric], marker="o", linewidth=2, label=split, color=color)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--")


def _build_interpretation(summary):
    return (
        "Interpretation: pose-memory summary stats stay relatively close between train/test, "
        "but KPT optimization metrics separate clearly. "
        f"By epoch {summary['final_epoch']}, loss_kpt gap = {summary['final_loss_kpt_gap']:.3f}, "
        f"MPJPE gap = {summary['final_mpjpe_gap']:.2f}, MPJDLE gap = {summary['final_mpjdle_gap']:.2f}. "
        f"Best test MPJPE appears at epoch {summary['best_test_mpjpe_epoch']} "
        f"({summary['best_test_mpjpe']:.2f})."
    )


def plot_overfitting_dashboard(mem_df, log_df, output_path, summary=None):
    if summary is None:
        summary = build_overfitting_summary(log_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=False)

    mem_metrics = [
        ("abs_mean", "Pose Memory Absolute Mean", "abs_mean"),
        ("std", "Pose Memory Std", "std"),
        ("sparsity", "Pose Memory Sparsity", "sparsity"),
    ]
    log_metrics = [
        ("loss_kpt", "KPT Loss", "loss_kpt"),
        ("MPJPE", "MPJPE", "MPJPE"),
        ("MPJDLE", "MPJDLE", "MPJDLE"),
    ]

    for row_index, (metric, title, ylabel) in enumerate(mem_metrics):
        _plot_metric(axes[row_index, 0], mem_df, metric, title, ylabel)
        train_last = float(mem_df[mem_df["split"] == "train"].sort_values("epoch")[metric].iloc[-1])
        test_last = float(mem_df[mem_df["split"] == "test"].sort_values("epoch")[metric].iloc[-1])
        axes[row_index, 0].text(
            0.02,
            0.95,
            f"final gap = {abs(test_last - train_last):.4f}",
            transform=axes[row_index, 0].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

    for row_index, (metric, title, ylabel) in enumerate(log_metrics):
        _plot_metric(axes[row_index, 1], log_df, metric, title, ylabel)
        pivot = log_df.pivot(index="epoch", columns="split", values=metric).sort_index()
        final_gap = float(pivot["test"].iloc[-1] - pivot["train"].iloc[-1])
        axes[row_index, 1].text(
            0.02,
            0.95,
            f"final gap = {final_gap:.4f}",
            transform=axes[row_index, 1].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")

    fig.suptitle("KPT Overfitting Dashboard: Pose Memory vs. Train/Test Performance", fontsize=16, y=0.995)
    fig.text(0.5, 0.01, _build_interpretation(summary), ha="center", va="bottom", wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "build_overfitting_summary",
    "load_log_metrics",
    "load_mem_stats",
    "plot_overfitting_dashboard",
]
