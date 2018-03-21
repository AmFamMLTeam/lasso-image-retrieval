import random

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from lasso_ir.features import X, y
from lasso_ir.constants import MULTIPROCESSING
from lasso_ir.models import NLogisticRegression, MarginalRegression


class BaseAlgorithm:
    """
    Base algorithm, defines most of the necessary methods. Implements passive/random search.
    """
    name = "random"

    def __init__(self, seed):
        self.n = len(y)
        self.answers = dict()
        self.answers[seed] = 1
        self.n_positive = 0
        self.unlabeled = list(range(self.n))
        self.unlabeled.pop(seed)

    def get_snapshot(self):
        return {}

    def get_query(self):
        return random.choice(self.unlabeled)

    def process_answer(self, index, label):
        self.answers[index] = label
        self.unlabeled.pop(self.unlabeled.index(index))
        self.n_positive += int(label == 1)

    def select_features(self):
        return X

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"'{self.name}'"


class NearestNeighbor(BaseAlgorithm):
    """
    For each query, picks a random positive example seen so far and finds the closest unlabeled example to that.
    The base NN algorithm uses all features, but LassoNN reimplements "select_features"
    """
    name = "nn"

    def get_query(self):
        positives = filter(lambda k: self.answers[k] == 1, self.answers.keys())
        target = random.choice(list(positives))
        _X = self.select_features()
        x = _X[target]
        _X = _X[self.unlabeled]
        dists = np.linalg.norm(_X - x, axis=1)
        best = np.argmin(dists)
        return self.unlabeled[best]


class MargionalNN(NearestNeighbor):
    def __init__(self, seed, N=1):
        self.N = N
        super().__init__(seed)

    def select_features(self):
        labeled = list(self.answers.keys())
        _y = list(self.answers.values())
        if can_fit(_y):
            n_trained = len(_y)
            _X = X[labeled]
            mask = MarginalRegression().fit(_X, _y).topfeatures(int(n_trained / self.N))
            return X[:, mask]
        else:
            return X


class LassoNN(NearestNeighbor):
    """
    Nearest Neighbor but uses LASSO to select the features
    """
    name = "nn-lasso"

    def __init__(self, seed, N=1):
        self.coefs = None
        self.C = .1
        self.N = N
        super().__init__(seed)

    def scoring(self):
        return sparsity_score

    def select_features(self):
        labeled = list(self.answers.keys())
        _y = list(self.answers.values())
        if can_fit(_y):
            _X = X[labeled]
            Cs = [self.C*2**n for n in range(-2, 3)] + [.1*2**n for n in range(-2, 3)]
            Cs = list(set(Cs))
            if MULTIPROCESSING:
                n_jobs = 1
            else:
                n_jobs = -1
            search = GridSearchCV(NLogisticRegression(N=self.N, penalty="l1", class_weight="balanced"), param_grid={"C": Cs}, scoring=self.scoring(), n_jobs=n_jobs, refit=True)
            model = search.fit(_X, _y).best_estimator_
            self.coefs = model.coef_
            self.C = model.get_params()["C"]
            mask = np.ravel(model.coef_.astype(bool))
            return X[:, mask]
        else:
            return X

    def get_snapshot(self):
        if self.coefs is None:
            return {"coefs": None, "n_nonzero": None, "C": None}
        else:
            return{"coefs": self.coefs.tolist(), "n_nonzero": np.count_nonzero(self.coefs), "C": self.C}


class NLassoNN(LassoNN):
    """
    LassoNN but requires number of nonzero features to be no larger than (number of training examples / self.N)
    """
    def scoring(self):
        return maximum_sparsity

    @property
    def name(self):
        return f"nn-lasso-{self.N}"


def can_fit(_y):
    return _y.count(1) > 3 and _y.count(0) > 3


def sparsity_score(est, _X: np.ndarray, _y: np.ndarray):
    sparsity = -np.count_nonzero(est.coef_) / est.coef_.shape[1]
    auc = roc_auc_score(_y, est.decision_function(_X))
    return sparsity + auc


def maximum_sparsity(est, _X: np.ndarray, _y: np.ndarray):
    if np.count_nonzero(est.coef_) > (est.n_trained / est.N):
        return -2
    else:
        return roc_auc_score(_y, est.decision_function(_X))
