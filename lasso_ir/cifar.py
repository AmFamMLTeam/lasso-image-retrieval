from lasso_ir.constants import CIFAR_FEATURES, MAX_N_CIFAR_TRAIN, CIFAR_POSITIVE_CLASSES, DOWNSAMPLE_POSITIVE
from functools import reduce
import h5py
import numpy as np
import os
import random

if not os.path.exists(CIFAR_FEATURES):
    raise Exception(f"{CIFAR_FEATURES} doesn't exist")


with h5py.File(CIFAR_FEATURES) as f:
    X_train_fc2 = f["train"]["X_fc2"][:MAX_N_CIFAR_TRAIN]
    X_test_fc2 = f["test"]["X_fc2"].value
    X_train_fc1 = f["train"]["X_fc2"][:MAX_N_CIFAR_TRAIN]
    X_test_fc1 = f["test"]["X_fc2"].value
    y_train = np.ravel(f["train"]["y"][:MAX_N_CIFAR_TRAIN])
    y_test = np.ravel(f["test"]["y"].value)


def set_y(y, classes):
    return reduce(lambda o, t: np.logical_or(o, t), map(lambda c: y == c, classes)).astype(int)


X_train_fc = np.hstack([X_train_fc1, X_train_fc2])
X_test_fc = np.hstack([X_test_fc1, X_test_fc2])
y_train = set_y(y_train, CIFAR_POSITIVE_CLASSES)
y_test = set_y(y_test, CIFAR_POSITIVE_CLASSES)

train_positive_i: list = np.ravel(np.argwhere(y_train == 1)).tolist()
train_negative_i: list = np.ravel(np.argwhere(y_train != 1)).tolist()
n_train_positive = len(train_positive_i)
train_positive_i_sample: list = random.sample(train_positive_i, n_train_positive // DOWNSAMPLE_POSITIVE)

train_i = train_negative_i + train_positive_i_sample

X_train_fc = X_train_fc[train_i]
y_train = y_train[train_i]

from lasso_ir.utils import debug

debug(f'number of examples: {len(y_train)}')
debug(f'number of positive examples: {len(np.argwhere(y_train == 1))}')

# clean up uneccessary ones to save memory
del X_train_fc1
del X_train_fc2
del X_test_fc
del X_test_fc1
del X_test_fc2
del train_i
del train_negative_i
del train_positive_i
del train_positive_i_sample
