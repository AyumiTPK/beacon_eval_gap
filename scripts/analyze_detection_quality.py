import argparse
from pathlib import Path
from tsad_transfer.detection_quality import analyze_detection_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/main'))
    parser.add_argument('--output', type=Path, default=Path('results/operational_quality'))
    parser.add_argument('--deadlines-min', type=float, nargs='+', default=[10, 30, 60])
    parser.add_argument('--chunksize', type=int, default=150000)
    args = parser.parse_args()
    summary = analyze_detection_quality(args.results, args.output, args.deadlines_min, args.chunksize)
    print(f'Saved operational-quality results for {len(summary)} models to {args.output}')


if __name__ == '__main__':
    main()
