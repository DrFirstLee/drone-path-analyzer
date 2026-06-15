from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import analyze_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone-path-analyzer",
        description="Analyze a drone path CSV and export segmented CSV files plus an overview PNG.",
    )
    parser.add_argument("csv_file", help="Path to the drone CSV file to analyze.")
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory. Defaults to ./output.",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=30,
        help="Rolling window size for yaw and speed features. Defaults to 30.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = analyze_csv(
        csv_path=Path(args.csv_file),
        output_dir=Path(args.output),
        sliding_window=args.sliding_window,
    )

    print("Drone path analysis complete.")
    print(f"Output directory: {result.output_dir}")
    print(f"Overview image: {result.overview_png}")
    print(f"Full result CSV: {result.full_result_csv}")
    print(f"Segment CSV files: {len(result.segment_csvs)}")
    return 0
