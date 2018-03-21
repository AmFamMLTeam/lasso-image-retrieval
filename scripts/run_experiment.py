
from lasso_ir.active.experiment import Experiment
from lasso_ir.active.algorithms import NearestNeighbor, LassoNN, BaseAlgorithm, NLassoNN, MargionalNN, MargionalNNPlus
from lasso_ir.constants import OUTPUT_PREFIX
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lasso_ir.features import y
from tqdm import tqdm
import json
import argparse
import sys

parser = argparse.ArgumentParser(description="The main script to run an active learning image retrieval experiment.")

parser.add_argument("-o", "--output", default=OUTPUT_PREFIX, help=f"the prefix for the output file(s), default: timestamp (e.g. '{OUTPUT_PREFIX.split('/')[-1]}')")
parser.add_argument("-f", "--figure", action="store_const", const=True, default=False, help="whether to make/save the figure, default: false")
parser.add_argument("-n", "--nqueries", default=20, type=int, help="the number of queries to run in this experiment. default: 20")

args = parser.parse_args(sys.argv[1:])

MAKEFIG = args.figure
N_QUERIES = args.nqueries
EXP_HISTORY = args.output + "history.json"
EXP_FIGURE = args.output + "plot.png"

seed = random.choice(np.where(y == 1)[0])

# Set up your algorithms here
algs = [BaseAlgorithm(seed), NearestNeighbor(seed), LassoNN(seed), NLassoNN(seed, 2), NLassoNN(seed, 3), MargionalNN(seed, 2), MargionalNN(seed, 3), MargionalNNPlus(seed, 2), MargionalNNPlus(seed, 3)]

exp = Experiment(algs)

for i in tqdm(range(N_QUERIES)):
    exp.get_query()

with open(EXP_HISTORY, "w") as f:
    results = {"n_positive": exp.history,
               "alg_meta": exp.alg_history}
    json.dump(results, f)

print(f"saved history to {EXP_HISTORY}")

if MAKEFIG:
    plt.figure()
    n_queries = exp.history["n_queries"]
    for alg in exp.algorithms:
        n_positive = exp.history[alg.name]
        plt.plot(n_queries, n_positive, label=alg.name)
    plt.legend()
    plt.savefig(EXP_FIGURE)
    print(f"saved figure to {EXP_FIGURE}")
