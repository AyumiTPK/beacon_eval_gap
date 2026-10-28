"""Run operational detection-quality and ranking post-analysis."""

import argparse
from pathlib import Path
from tsad_transfer.detection_quality import analyze_detection_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/main"),
        help="Directory containing model subdirectories and retrospective metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/operational_quality"),
        help="Directory in which to save the six post-analysis CSV files.",
    )
    parser.add_argument(
        "--deadlines-min",
        type=float,
        nargs="+",
        default=[10, 30, 60],
        help="Fault detection deadlines, measured in minutes.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=150000,
        help="Number of score rows to read per CSV chunk.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.990, 0.995, 0.999],
        help="Reference-calibration quantiles for threshold sensitivity.",
    )
    args = parser.parse_args()

    summary = analyze_detection_quality(
        results=args.results,
        output=args.output,
        deadlines=args.deadlines_min,
        chunksize=args.chunksize,
        quantiles=args.quantiles,
    )

    print(
        f"Saved operational-quality results for {len(summary)} models "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
