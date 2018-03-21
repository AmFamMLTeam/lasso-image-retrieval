import os
import datetime
import logging
from os.path import join

CATS = {3}
MACHINES = {0, 1, 8, 9}

# These are the constants you want to play with
###
MAX_N_CIFAR_TRAIN = 50000
CIFAR_POSITIVE_CLASSES = CATS
MULTIPROCESSING = True
###

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_HOME = os.path.dirname(PACKAGE_DIR)

DATA_DIR = join(PROJECT_HOME, "data")
OUTPUT_DIR = join(DATA_DIR, "output")

LOGS_DIR = join(PROJECT_HOME, "logs")
LOG_FILE = join(LOGS_DIR, f"log-{datetime.date.today()}.txt")

FILE_LOG_LEVEL = logging.DEBUG
STDERR_LOG_LEVEL = logging.DEBUG

CIFAR_DIR = join(DATA_DIR, "cifar10")
CIFAR_DATA = join(CIFAR_DIR, "cifar_resized.hdf5")
CIFAR_FEATURES = join(CIFAR_DIR, "cifar_feats.hdf5")
CIFAR_FC_FEATURES = join(CIFAR_DIR, "cifar_fc_feats.hdf5")

TIMESTAMP = datetime.datetime.now().strftime("%m%d-%H%M")
OUTPUT_PREFIX = join(OUTPUT_DIR, TIMESTAMP)
EXP_FIGURE = OUTPUT_PREFIX + "plot.png"
EXP_HISTORY = OUTPUT_PREFIX + "history.json"

for d in [DATA_DIR, OUTPUT_DIR, LOGS_DIR, CIFAR_DIR]:
    os.makedirs(d, exist_ok=True)
