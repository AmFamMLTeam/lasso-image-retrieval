from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn import preprocessing
import numpy as np
import warnings


class MarginalRegression(LinearModel, RegressorMixin):
    def fit(self, X, y):
        self.n_trained = len(y)
        # scale gives zero mean and unit variance
        X /= np.max(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            X = preprocessing.scale(X)
        self.coef_ = np.dot(X.T, y)
        return self

    def topfeatures(self, k=None):
        if not hasattr(self, 'coef_'):
            raise NotFittedError
        k = k or self.n_trained
        # sort coefficients highest to lowest
        topk_indices = np.argsort(-np.abs(self.coef_))[:k]
        mask = np.zeros(len(self.coef_), dtype=bool)
        mask[topk_indices] = True
        return mask


class NLogisticRegression(LogisticRegression):
    """
    A version of Logistic Regression that keeps track of an extra parameter N (used for maximum_sparsity scoring) and how many examples were used to train it.
    """
    def __init__(self, N, penalty, class_weight, C=.1):
        self.N = N
        super().__init__(penalty=penalty, class_weight=class_weight, C=C)

    def fit(self, _X, _y, sample_weight=None):
        self.n_trained = len(_y)
        super().fit(_X, _y, sample_weight)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    _X, _y = make_classification(1000, n_features=100, n_informative=10)

    lasso = LogisticRegression(C=.01, penalty="l1").fit(_X, _y)
    marg = MarginalRegression().fit(_X, _y)

    print(np.where(marg.topkfeatures(5))[0])
    print(np.argwhere(np.abs(np.ravel(lasso.coef_)) > 0))
