import argparse
from tsad_transfer.analysis import analyze

parser = argparse.ArgumentParser()
parser.add_argument("results")
args = parser.parse_args()
analyze(args.results)