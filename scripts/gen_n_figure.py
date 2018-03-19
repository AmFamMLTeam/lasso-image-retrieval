import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from lasso_ir.constants import OUTPUT_DIR, EXP_FIGURE
import argparse
import sys

parser = argparse.ArgumentParser(description="A script to generate a plot of number of positive queries vs number of queries.")

parser.add_argument("JSON", help="history json file (output from run_experiment.py)")
parser.add_argument("-o", "--output", help=f"output file. default: timestamp (e.g. '{EXP_FIGURE.split('/')[-1]}')")

args = parser.parse_args(sys.argv[1:])

output = args.output or EXP_FIGURE

with open(args.JSON) as f:
    history = json.load(f)

history = history["n_positive"]


pretty_names = {"random": "Random Sampling",
                "nn": "Nearest Neighbors (all)",
                "nn-lasso": "Nearest Neighbors (LASSO)"}

n_queries = history.pop("n_queries")
for alg, alg_data in history.items():
    plt.plot(n_queries, alg_data, label=pretty_names.get(alg, alg))

plt.legend()

plt.xlabel("# of Queries")
plt.ylabel("# of Positives")

print(f"saving figure to {output.replace('.png', '')}.png")
plt.savefig(output)

