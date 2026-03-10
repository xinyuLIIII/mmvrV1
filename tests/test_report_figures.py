import json
from pathlib import Path

from utils.report_figures import build_autofigure_prompt, create_report_figures, load_last_run_log_metrics


def test_load_last_run_log_metrics_ignores_appended_runs(tmp_path):
    train_log = tmp_path / "train.log"
    test_log = tmp_path / "test.log"

    train_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 0.9, 'MPJPE': 90.0, 'MPJDLE': 45.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 1, 'loss_kpt': 0.8, 'MPJPE': 80.0, 'MPJDLE': 40.0}\n"
        "2026-03-05 00:00:02,000 - INFO - {'epoch': 18, 'loss_kpt': 0.3, 'MPJPE': 30.0, 'MPJDLE': 15.0, 'mpjpe_thumb': 31.0, 'mpjpe_index': 32.0, 'mpjpe_middle': 33.0, 'mpjpe_ring': 34.0, 'mpjpe_pinky': 35.0}\n"
        "2026-03-05 00:00:03,000 - INFO - {'epoch': 19, 'loss_kpt': 0.2, 'MPJPE': 20.0, 'MPJDLE': 10.0, 'mpjpe_thumb': 21.0, 'mpjpe_index': 22.0, 'mpjpe_middle': 23.0, 'mpjpe_ring': 24.0, 'mpjpe_pinky': 25.0}\n"
    )
    test_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 1.0, 'MPJPE': 100.0, 'MPJDLE': 50.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 1, 'loss_kpt': 0.85, 'MPJPE': 85.0, 'MPJDLE': 42.0}\n"
        "2026-03-05 00:00:02,000 - INFO - {'epoch': 18, 'loss_kpt': 0.33, 'MPJPE': 33.0, 'MPJDLE': 16.5, 'mpjpe_thumb': 34.0, 'mpjpe_index': 35.0, 'mpjpe_middle': 36.0, 'mpjpe_ring': 37.0, 'mpjpe_pinky': 38.0}\n"
        "2026-03-05 00:00:03,000 - INFO - {'epoch': 19, 'loss_kpt': 0.25, 'MPJPE': 25.0, 'MPJDLE': 12.5, 'mpjpe_thumb': 26.0, 'mpjpe_index': 27.0, 'mpjpe_middle': 28.0, 'mpjpe_ring': 29.0, 'mpjpe_pinky': 30.0}\n"
    )

    frame = load_last_run_log_metrics(train_log, test_log)

    assert list(frame[frame["split"] == "train"]["epoch"]) == [1, 18, 19]
    assert list(frame[frame["split"] == "test"]["epoch"]) == [1, 18, 19]
    assert float(frame[(frame["split"] == "test") & (frame["epoch"] == 19)]["MPJPE"].iloc[0]) == 25.0


def test_create_report_figures_writes_outputs_and_manifest(tmp_path):
    baseline_train_log = tmp_path / "baseline_train.log"
    baseline_test_log = tmp_path / "baseline_test.log"
    current_train_log = tmp_path / "current_train.log"
    current_test_log = tmp_path / "current_test.log"
    baseline_summary = tmp_path / "baseline_summary.json"
    run_summary = tmp_path / "run_summary.json"
    output_dir = tmp_path / "report_figures"

    baseline_train_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 1.0, 'MPJPE': 200.0, 'MPJDLE': 100.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 2, 'loss_kpt': 0.5, 'MPJPE': 120.0, 'MPJDLE': 60.0}\n"
    )
    baseline_test_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 1.1, 'MPJPE': 210.0, 'MPJDLE': 110.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 2, 'loss_kpt': 1.0, 'MPJPE': 205.0, 'MPJDLE': 105.0}\n"
    )
    current_train_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 0.9, 'MPJPE': 90.0, 'MPJDLE': 45.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 1, 'loss_kpt': 0.8, 'MPJPE': 80.0, 'MPJDLE': 40.0}\n"
        "2026-03-05 00:00:02,000 - INFO - {'epoch': 18, 'loss_kpt': 0.3, 'MPJPE': 30.0, 'MPJDLE': 15.0, 'mpjpe_thumb': 31.0, 'mpjpe_index': 32.0, 'mpjpe_middle': 33.0, 'mpjpe_ring': 34.0, 'mpjpe_pinky': 35.0}\n"
        "2026-03-05 00:00:03,000 - INFO - {'epoch': 43, 'loss_kpt': 0.15, 'MPJPE': 22.0, 'MPJDLE': 11.0, 'mpjpe_thumb': 23.0, 'mpjpe_index': 24.0, 'mpjpe_middle': 25.0, 'mpjpe_ring': 26.0, 'mpjpe_pinky': 27.0}\n"
        "2026-03-05 00:00:04,000 - INFO - {'epoch': 50, 'loss_kpt': 0.1, 'MPJPE': 18.0, 'MPJDLE': 9.0, 'mpjpe_thumb': 19.0, 'mpjpe_index': 20.0, 'mpjpe_middle': 21.0, 'mpjpe_ring': 22.0, 'mpjpe_pinky': 23.0}\n"
    )
    current_test_log.write_text(
        "2026-03-05 00:00:00,000 - INFO - {'epoch': 1, 'loss_kpt': 1.0, 'MPJPE': 100.0, 'MPJDLE': 50.0}\n"
        "2026-03-05 00:00:01,000 - INFO - {'epoch': 1, 'loss_kpt': 0.85, 'MPJPE': 85.0, 'MPJDLE': 42.0}\n"
        "2026-03-05 00:00:02,000 - INFO - {'epoch': 18, 'loss_kpt': 0.33, 'MPJPE': 33.0, 'MPJDLE': 16.5, 'mpjpe_thumb': 34.0, 'mpjpe_index': 35.0, 'mpjpe_middle': 36.0, 'mpjpe_ring': 37.0, 'mpjpe_pinky': 38.0}\n"
        "2026-03-05 00:00:03,000 - INFO - {'epoch': 43, 'loss_kpt': 0.2, 'MPJPE': 24.0, 'MPJDLE': 12.0, 'mpjpe_thumb': 25.0, 'mpjpe_index': 26.0, 'mpjpe_middle': 27.0, 'mpjpe_ring': 28.0, 'mpjpe_pinky': 29.0}\n"
        "2026-03-05 00:00:04,000 - INFO - {'epoch': 50, 'loss_kpt': 0.4, 'MPJPE': 40.0, 'MPJDLE': 20.0, 'mpjpe_thumb': 41.0, 'mpjpe_index': 42.0, 'mpjpe_middle': 43.0, 'mpjpe_ring': 44.0, 'mpjpe_pinky': 45.0}\n"
    )

    baseline_summary.write_text(
        json.dumps(
            {
                "first_epoch": 1,
                "final_epoch": 2,
                "final_loss_kpt_gap": 0.5,
                "final_mpjpe_gap": 85.0,
                "final_mpjdle_gap": 45.0,
                "best_test_mpjpe_epoch": 2,
                "best_test_mpjpe": 205.0,
            }
        )
    )
    run_summary.write_text(
        json.dumps(
            {
                "current_run": {
                    "best_test_mpjpe_epoch": 43,
                    "best_test_mpjpe": 24.0,
                    "best_test_mpjdle": 12.0,
                    "final_train_mpjpe": 18.0,
                    "final_test_mpjpe": 40.0,
                    "final_train_loss_kpt": 0.1,
                    "final_test_loss_kpt": 0.4,
                    "final_loss_kpt_gap": 0.3,
                    "final_mpjpe_gap": 22.0,
                },
                "baseline_comparison": {
                    "baseline_best_test_mpjpe": 205.0,
                    "baseline_best_test_mpjpe_epoch": 2,
                    "baseline_final_mpjpe_gap": 85.0,
                    "baseline_final_loss_kpt_gap": 0.5,
                    "best_test_mpjpe_improvement": 181.0,
                    "final_mpjpe_gap_reduction": 63.0,
                    "final_loss_kpt_gap_reduction": 0.2,
                },
                "late_overfitting_signal": {
                    "reference_epoch": 43,
                    "train_mpjpe_epoch_43": 22.0,
                    "test_mpjpe_epoch_43": 24.0,
                    "train_mpjpe_epoch_50": 18.0,
                    "test_mpjpe_epoch_50": 40.0,
                    "train_mpjpe_improvement_after_epoch_43": 4.0,
                    "test_mpjpe_degradation_after_epoch_43": 16.0,
                    "test_mpjdle_degradation_after_epoch_43": 8.0,
                    "mpjpe_gap_epoch_43": 2.0,
                    "mpjpe_gap_epoch_50": 22.0,
                },
            }
        )
    )

    manifest = create_report_figures(
        baseline_train_log,
        baseline_test_log,
        baseline_summary,
        run_summary,
        current_train_log,
        current_test_log,
        output_dir,
    )

    expected_files = {
        "fig1_baseline_overfitting.png",
        "fig2_model_improvement_summary.png",
        "fig3_attn_bilstm_training_dynamics.png",
        "figS1_per_finger_breakdown.png",
        "figure_manifest.json",
        "fig4_autofigure_prompt.txt",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    assert manifest["figures"]["fig3"]["reference_epoch"] == 43
    assert manifest["figures"]["fig2"]["best_epoch"] == 43
    assert "Attention-BiLSTM temporal decoder" in build_autofigure_prompt()
