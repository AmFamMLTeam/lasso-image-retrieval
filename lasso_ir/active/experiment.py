from multiprocessing import Pool
from lasso_ir.features import y
from lasso_ir.constants import MULTIPROCESSING


def get_query_alg(alg):
    query = alg.get_query()
    alg.process_answer(query, y[query])
    return alg


class Experiment:
    def __init__(self, algorithms):
        self.algorithms = algorithms
        self.history = {alg.name: [] for alg in self.algorithms}
        self.alg_history = {alg.name: [] for alg in self.algorithms}
        self.history["n_queries"] = []
        self.pool = Pool(min(len(algorithms), 4))

    def get_query(self):

        if MULTIPROCESSING:
            self.algorithms = self.pool.map(get_query_alg, self.algorithms)
        else:
            self.algorithms = list(map(get_query_alg, self.algorithms))

        for alg in self.algorithms:
            self.history[alg.name].append(alg.n_positive)
            snapshot = alg.get_snapshot()
            if snapshot:
                self.alg_history[alg.name].append(snapshot)

        self.history["n_queries"].append(len(self.history["n_queries"]))
