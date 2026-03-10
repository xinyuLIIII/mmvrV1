import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.report_figures import create_report_figures


def main():
    parser = argparse.ArgumentParser(description="Create report figures for baseline and Attention-BiLSTM runs.")
    parser.add_argument("--baseline-train-log", default="experiments/train_train.log")
    parser.add_argument("--baseline-test-log", default="experiments/train_test.log")
    parser.add_argument("--baseline-summary", default="experiments/kpt_overfitting_summary.json")
    parser.add_argument("--run-summary", default="experiments/train_attn_bilstm_last_run_summary.json")
    parser.add_argument("--train-log", default="experiments/train_attn_bilstm_train.log")
    parser.add_argument("--test-log", default="experiments/train_attn_bilstm_test.log")
    parser.add_argument("--output-dir", default="experiments/report_figures")
    args = parser.parse_args()

    manifest = create_report_figures(
        args.baseline_train_log,
        args.baseline_test_log,
        args.baseline_summary,
        args.run_summary,
        args.train_log,
        args.test_log,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
