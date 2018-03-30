import random

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from lasso_ir.features import X, y
from lasso_ir.constants import MULTIPROCESSING
from lasso_ir.models import NLogisticRegression, MarginalRegression
import time

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
        t = time.time()
        positives = filter(lambda k: self.answers[k] == 1, self.answers.keys())
        target = random.choice(list(positives))
        _X = self.select_features()
        x = _X[target]
        _X = _X[self.unlabeled]
        dists = np.linalg.norm(_X - x, axis=1)
        best = np.argmin(dists)
        self.elapsed = time.time() - t
        return self.unlabeled[best]


class MargionalNN(NearestNeighbor):
    """
    Nearest Neighbor but uses Marginal Regression to select the features.
    Will select n_trained/N features, (N=1 by default)
    """
    def __init__(self, seed, N=1):
        self.N = N
        self.mask = None
        super().__init__(seed)

    def select_features(self):
        labeled = list(self.answers.keys())
        _y = list(self.answers.values())
        if can_fit(_y):
            n_trained = len(_y)
            _X = X[labeled]
            self.mask = MarginalRegression().fit(_X, _y).topfeatures(int(n_trained / self.N))
            return X[:, self.mask]
        else:
            return X

    @property
    def name(self):
        return f"nn-marginal-{self.N}"

    def get_snapshot(self):
        if self.mask is None:
            return {"coefs": None, "n_nonzero": None, "elapsed": self.elapsed}
        else:
            return{"coefs": np.ravel(np.argwhere(self.mask)).tolist(), "n_nonzero": np.count_nonzero(self.mask),  "elapsed": self.elapsed}


class MargionalNNPlus(MargionalNN):
    @property
    def name(self):
        return f"nn-marginalplus-{self.N}"

    def select_features(self):
        labeled = list(self.answers.keys())
        n_trained = len(self.answers)
        n_positive = list(self.answers.values()).count(1)
        n_negative = n_trained - n_positive
        n_sample = n_positive - n_negative
        unlabeled_sample = []
        if 0 < n_sample <= len(self.unlabeled):
            unlabeled_sample = random.sample(self.unlabeled, n_sample)
        sample = labeled + unlabeled_sample
        _y = [self.answers.get(i, 0) for i in sample]  # default to 0
        #_y = [self.answers.get(i, -1) for i in sample]  # default to -1
        if can_fit(_y):
            _X = X[sample]
            self.mask = MarginalRegression().fit(_X, _y).topfeatures(int(n_trained / self.N))
            return X[:, self.mask]
        else:
            return X

class MargionalNNPlusPlus(MargionalNN):
    @property
    def name(self):
        return f"nn-marginalplusplus-{self.N}"
    
    def select_features(self):
        labeled = list(self.answers.keys())
        n_trained = len(self.answers)
        n_positive = list(self.answers.values()).count(1)
        n_negative = n_trained - n_positive
        n_sample = n_positive - n_negative
        unlabeled_sample = []
        if 0 < n_sample <= len(self.unlabeled):
            unlabeled_sample = random.sample(self.unlabeled, n_sample)
            sample = labeled + unlabeled_sample
        else:
            sample = [i for i in labeled if self.answers[i] == 1]
            neg_sample = [i for i in labeled if self.answers[i] != 1]
            sample = sample + random.sample(neg_sample, n_positive)
        _y = [self.answers.get(i, 0) for i in sample]  # default to 0
        #_y = [self.answers.get(i, -1) for i in sample]  # default to -1
        if can_fit(_y):
            _X = X[sample]
            self.mask = MarginalRegression().fit(_X, _y).topfeatures(int(n_trained / self.N))
            return X[:, self.mask]
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
            return {"coefs": None, "n_nonzero": None, "C": None, "elapsed": self.elapsed}
        else:
            return{"coefs": self.coefs.tolist(), "n_nonzero": np.count_nonzero(self.coefs), "C": self.C, "elapsed": self.elapsed}


class LassoNNPlus(NearestNeighbor):
    """
        Nearest Neighbor but uses LASSO to select the features plus samples balanced
        """
    name = "nn-lasso-plus"
    
    def __init__(self, seed, N=1):
        self.coefs = None
        self.C = .1
        self.N = N
        super().__init__(seed)
    
    def scoring(self):
        return sparsity_score
    
    def select_features(self):
        labeled = list(self.answers.keys())
        labeled = list(self.answers.keys())
        n_trained = len(self.answers)
        n_positive = sum(self.answers.values())
        n_negative = n_trained - n_positive
        n_sample = n_positive - n_negative
        unlabeled_sample = []
        if 0 < n_sample <= len(self.unlabeled):
            unlabeled_sample = random.sample(self.unlabeled, n_sample)
        sample = labeled + unlabeled_sample
        _y = [self.answers.get(i, 0) for i in sample]  # default to 0
        #_y = [self.answers.get(i, -1) for i in sample]  # default to -1
        if can_fit(_y):
            _X = X[sample]
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
        return {"coefs": None, "n_nonzero": None, "C": None, "elapsed": self.elapsed}
    else:
        return{"coefs": self.coefs.tolist(), "n_nonzero": np.count_nonzero(self.coefs), "C": self.C, "elapsed": self.elapsed}




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
