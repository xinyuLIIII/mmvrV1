import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LOG_PATTERN = re.compile(r"INFO - (\{.*\})")
FIGURE_FILENAMES = {
    "fig1": "fig1_baseline_overfitting.png",
    "fig2": "fig2_model_improvement_summary.png",
    "fig3": "fig3_attn_bilstm_training_dynamics.png",
    "figS1": "figS1_per_finger_breakdown.png",
    "manifest": "figure_manifest.json",
    "prompt": "fig4_autofigure_prompt.txt",
}
TRAIN_COLOR = "#1f77b4"
TEST_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2a9d8f"
WARNING_COLOR = "#d95f02"


def _read_json(path):
    return json.loads(Path(path).read_text())


def _parse_log_file(log_path, split):
    rows = []
    for line in Path(log_path).read_text().splitlines():
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        payload = ast.literal_eval(match.group(1))
        payload["split"] = split
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No epoch metrics found in {log_path}")
    return rows


def _last_monotonic_segment(rows):
    start = 0
    prev_epoch = None
    for index, row in enumerate(rows):
        epoch = row["epoch"]
        if prev_epoch is not None and epoch <= prev_epoch:
            start = index
        prev_epoch = epoch
    return rows[start:]


def _rows_to_frame(rows):
    return pd.DataFrame(rows).sort_values(["split", "epoch"]).reset_index(drop=True)


def load_last_run_log_metrics(train_log_path, test_log_path):
    train_rows = _last_monotonic_segment(_parse_log_file(train_log_path, "train"))
    test_rows = _last_monotonic_segment(_parse_log_file(test_log_path, "test"))
    return _rows_to_frame(train_rows + test_rows)


def load_log_metrics(train_log_path, test_log_path):
    return _rows_to_frame(
        _parse_log_file(train_log_path, "train") + _parse_log_file(test_log_path, "test")
    )


def _apply_axes_style(ax, title, ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_train_test_lines(ax, frame, metric, title, ylabel):
    for split, color in (("train", TRAIN_COLOR), ("test", TEST_COLOR)):
        split_df = frame[frame["split"] == split].sort_values("epoch")
        ax.plot(split_df["epoch"], split_df[metric], marker="o", linewidth=2, color=color, label=split)
    _apply_axes_style(ax, title, ylabel)


def _metric_at_epoch(frame, split, epoch, metric):
    split_df = frame[frame["split"] == split].set_index("epoch")
    return float(split_df.loc[epoch, metric])


def plot_baseline_overfitting(baseline_log_df, baseline_summary, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [
        ("loss_kpt", "Baseline KPT Loss", "loss_kpt", "final_loss_kpt_gap"),
        ("MPJPE", "Baseline MPJPE", "MPJPE", "final_mpjpe_gap"),
        ("MPJDLE", "Baseline MPJDLE", "MPJDLE", "final_mpjdle_gap"),
    ]
    for ax, (metric, title, ylabel, gap_key) in zip(axes, specs):
        _plot_train_test_lines(ax, baseline_log_df, metric, title, ylabel)
        ax.text(
            0.03,
            0.95,
            f"final gap = {baseline_summary[gap_key]:.4f}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    axes[1].text(
        0.98,
        0.08,
        f"best test MPJPE = {baseline_summary['best_test_mpjpe']:.4f}\n@ epoch {baseline_summary['best_test_mpjpe_epoch']}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Baseline KPT Overfitting Evidence", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_improvement_summary(current_log_df, run_summary, output_path):
    baseline = run_summary["baseline_comparison"]
    current = run_summary["current_run"]
    best_epoch = current["best_test_mpjpe_epoch"]
    final_epoch = current["final_epoch"]
    states = ["Baseline", f"Best@{best_epoch}", f"Final@{final_epoch}"]
    best_gap = _metric_at_epoch(current_log_df, "test", best_epoch, "MPJPE") - _metric_at_epoch(
        current_log_df, "train", best_epoch, "MPJPE"
    )
    final_gap = _metric_at_epoch(current_log_df, "test", final_epoch, "MPJPE") - _metric_at_epoch(
        current_log_df, "train", final_epoch, "MPJPE"
    )
    best_loss_gap = _metric_at_epoch(current_log_df, "test", best_epoch, "loss_kpt") - _metric_at_epoch(
        current_log_df, "train", best_epoch, "loss_kpt"
    )
    final_loss_gap = _metric_at_epoch(current_log_df, "test", final_epoch, "loss_kpt") - _metric_at_epoch(
        current_log_df, "train", final_epoch, "loss_kpt"
    )
    metric_specs = [
        (
            "Best Test MPJPE",
            [
                baseline["baseline_best_test_mpjpe"],
                _metric_at_epoch(current_log_df, "test", best_epoch, "MPJPE"),
                _metric_at_epoch(current_log_df, "test", final_epoch, "MPJPE"),
            ],
            baseline["best_test_mpjpe_improvement"],
        ),
        (
            "MPJPE Generalization Gap",
            [baseline["baseline_final_mpjpe_gap"], best_gap, final_gap],
            baseline["final_mpjpe_gap_reduction"],
        ),
        (
            "loss_kpt Generalization Gap",
            [baseline["baseline_final_loss_kpt_gap"], best_loss_gap, final_loss_gap],
            baseline["final_loss_kpt_gap_reduction"],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bar_colors = [WARNING_COLOR, ACCENT_COLOR, TEST_COLOR]
    for ax, (title, values, improvement) in zip(axes, metric_specs):
        ax.bar(states, values, color=bar_colors, width=0.65)
        _apply_axes_style(ax, title, title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.text(
            0.5,
            0.95,
            f"improvement = {improvement:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    fig.suptitle("Baseline vs. Attention-BiLSTM Improvement Summary", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_dynamics(current_log_df, run_summary, output_path):
    late_signal = run_summary["late_overfitting_signal"]
    current = run_summary["current_run"]
    ref_epoch = late_signal["reference_epoch"]
    final_epoch = current["final_epoch"]
    best_train_mpjpe = _metric_at_epoch(current_log_df, "train", ref_epoch, "MPJPE")
    best_test_mpjpe = _metric_at_epoch(current_log_df, "test", ref_epoch, "MPJPE")
    final_train_mpjpe = _metric_at_epoch(current_log_df, "train", final_epoch, "MPJPE")
    final_test_mpjpe = _metric_at_epoch(current_log_df, "test", final_epoch, "MPJPE")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _plot_train_test_lines(axes[0], current_log_df, "MPJPE", "Attention-BiLSTM MPJPE", "MPJPE")
    _plot_train_test_lines(axes[1], current_log_df, "loss_kpt", "Attention-BiLSTM loss_kpt", "loss_kpt")

    for ax in axes:
        ax.axvline(ref_epoch, color=ACCENT_COLOR, linestyle="--", linewidth=2)
        ax.axvspan(ref_epoch, final_epoch, color=WARNING_COLOR, alpha=0.12)

    axes[0].text(
        0.98,
        0.95,
        (
            f"best checkpoint: epoch {ref_epoch}\n"
            f"test MPJPE {best_test_mpjpe:.4f} -> {final_test_mpjpe:.4f}\n"
            f"train MPJPE {best_train_mpjpe:.4f} -> {final_train_mpjpe:.4f}"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axes[1].text(
        0.98,
        0.95,
        "late overfitting region",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Attention-BiLSTM Training Dynamics and Late Overfitting", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_finger_breakdown(current_log_df, run_summary, output_path):
    ref_epoch = run_summary["late_overfitting_signal"]["reference_epoch"]
    final_epoch = run_summary["current_run"]["final_epoch"]
    test_df = current_log_df[current_log_df["split"] == "test"].set_index("epoch")
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    ref_values = [float(test_df.loc[ref_epoch, f"mpjpe_{finger}"]) for finger in fingers]
    final_values = [float(test_df.loc[final_epoch, f"mpjpe_{finger}"]) for finger in fingers]

    x = range(len(fingers))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([value - width / 2 for value in x], ref_values, width=width, color=ACCENT_COLOR, label=f"epoch {ref_epoch}")
    ax.bar([value + width / 2 for value in x], final_values, width=width, color=TEST_COLOR, label=f"epoch {final_epoch}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fingers)
    _apply_axes_style(ax, "Per-finger Test MPJPE Breakdown", "MPJPE")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_autofigure_prompt():
    return """Create a clean academic model-overview figure for a group meeting slide.

Topic: improved mmWave-IMU hand pose estimation pipeline with an Attention-BiLSTM temporal decoder.

Style requirements:
- publication-style scientific diagram
- white background
- clean vector look
- blue/orange/teal accent colors
- minimal text, no decorative elements
- left-to-right information flow
- readable at slide scale
- emphasize the modified temporal module with a highlighted box

Content to show:
1. Two inputs on the left:
   - mmWave frames
   - IMU sequence
2. Feature extraction stage:
   - mmWave backbone
   - IMU backbone
3. Dual encoder fusion stage:
   - mmWave encoder
   - IMU encoder
   - fused memory representation
4. Pose decoding stage:
   - pose decoder
   - output pose_memory
5. Highlighted improved temporal decoder block:
   - temporal attention blocks
   - BiLSTM temporal modeling
   - gating fusion between frame queries and temporal context
   - query refinement blocks
6. Prediction head on the right:
   - hand keypoint prediction
   - gesture/logit prediction
7. Add one small annotation near the highlighted block:
   - "Main modification for reducing KPT overfitting"

Layout guidance:
- use rectangular module boxes and arrows
- make the highlighted temporal decoder visually distinct
- show that pose_memory and fused memory both feed into the temporal decoder
- final outputs should be clearly separated on the far right
- overall figure should feel suitable for a lab presentation, not a commercial infographic
"""


def create_report_figures(
    baseline_train_log_path,
    baseline_test_log_path,
    baseline_summary_path,
    run_summary_path,
    train_log_path,
    test_log_path,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = _read_json(baseline_summary_path)
    run_summary = _read_json(run_summary_path)
    baseline_log_df = load_log_metrics(baseline_train_log_path, baseline_test_log_path)
    current_log_df = load_last_run_log_metrics(train_log_path, test_log_path)

    current_run = run_summary.setdefault("current_run", {})
    current_run.setdefault("final_epoch", int(current_log_df["epoch"].max()))

    fig1_path = output_dir / FIGURE_FILENAMES["fig1"]
    fig2_path = output_dir / FIGURE_FILENAMES["fig2"]
    fig3_path = output_dir / FIGURE_FILENAMES["fig3"]
    fig_s1_path = output_dir / FIGURE_FILENAMES["figS1"]
    prompt_path = output_dir / FIGURE_FILENAMES["prompt"]
    manifest_path = output_dir / FIGURE_FILENAMES["manifest"]

    plot_baseline_overfitting(baseline_log_df, baseline_summary, fig1_path)
    plot_model_improvement_summary(current_log_df, run_summary, fig2_path)
    plot_training_dynamics(current_log_df, run_summary, fig3_path)
    plot_per_finger_breakdown(current_log_df, run_summary, fig_s1_path)
    prompt_path.write_text(build_autofigure_prompt())

    manifest = {
        "files": {
            "fig1": str(fig1_path),
            "fig2": str(fig2_path),
            "fig3": str(fig3_path),
            "figS1": str(fig_s1_path),
            "fig4_prompt": str(prompt_path),
        },
        "figures": {
            "fig1": {
                "best_test_mpjpe_epoch": baseline_summary["best_test_mpjpe_epoch"],
                "final_mpjpe_gap": baseline_summary["final_mpjpe_gap"],
                "final_loss_kpt_gap": baseline_summary["final_loss_kpt_gap"],
            },
            "fig2": {
                "best_epoch": run_summary["current_run"]["best_test_mpjpe_epoch"],
                "best_test_mpjpe_improvement": run_summary["baseline_comparison"]["best_test_mpjpe_improvement"],
                "final_mpjpe_gap_reduction": run_summary["baseline_comparison"]["final_mpjpe_gap_reduction"],
            },
            "fig3": {
                "reference_epoch": run_summary["late_overfitting_signal"]["reference_epoch"],
                "test_mpjpe_degradation_after_reference": run_summary["late_overfitting_signal"]["test_mpjpe_degradation_after_epoch_43"],
            },
            "figS1": {
                "comparison_epochs": [
                    run_summary["late_overfitting_signal"]["reference_epoch"],
                    run_summary["current_run"]["final_epoch"],
                ]
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


__all__ = [
    "build_autofigure_prompt",
    "create_report_figures",
    "load_last_run_log_metrics",
]
