#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.report_figures import create_experiment_comparison_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison figures for the baseline and Attention-BiLSTM experiment logs."
    )
    parser.add_argument(
        "--experiments-dir",
        default=str(ROOT / "experiments"),
        help="Directory containing the canonical experiment logs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <experiments-dir>/report_figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = create_experiment_comparison_report(args.experiments_dir, args.output_dir)
    print(f"Generated figures in: {report['output_dir']}")
    print(f"Figure manifest: {report['figure_manifest_path']}")
    print(f"Comparison summary: {report['comparison_summary_path']}")


if __name__ == "__main__":
    main()
