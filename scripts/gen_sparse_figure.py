import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from lasso_ir.constants import EXP_FIGURE
import argparse
import sys

parser = argparse.ArgumentParser(description="A script to generate a plot of sparsity of features vs number of queries.")

parser.add_argument("JSON", help="history json file (output from run_experiment.py)")
parser.add_argument("-o", "--output", help=f"output file. default: timestamp (e.g. '{EXP_FIGURE.split('/')[-1]}')")

args = parser.parse_args(sys.argv[1:])

output = args.output or EXP_FIGURE

with open(args.JSON) as f:
    history = json.load(f)

n_queries = history["n_positive"]["n_queries"]
history = history["alg_meta"]

for alg, alg_meta in history.items():
    if alg == "nn":
        continue
    n_nonzero = map(lambda x: x or 0, map(lambda x: x["n_nonzero"], alg_meta))
    plt.plot(n_queries, list(n_nonzero), label=alg)

plt.legend()

plt.xlabel("# of Queries")
plt.ylabel("# of Nonzero Coefficients")

print(f"saving figure to {output.replace('.png', '')}.png")
plt.savefig(output)

