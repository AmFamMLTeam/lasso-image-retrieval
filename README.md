

## Setup

To set up the environment, install [conda](https://conda.io/docs/user-guide/install/index.html) and run the `./setup_env.sh` script.

Next, `source activate.sh` to activate the conda environment and set your `PYTHONPATH` to include the `lasso_ir` package.

You'll need to `source activate.sh` in any new shell before running the scripts or anything that involves importing the `lasso_ir` package.

## Scripts

#### `python scripts/run_experiment.py`

```
usage: run_experiment.py [-h] [-o OUTPUT] [-f] [-n NQUERIES]

Run an active learning image retrieval experiment.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        the prefix for the output file(s), default: timestamp
                        (e.g. '0319-1337')
  -f, --figure          whether to make/save the figure, default: false
  -n NQUERIES, --nqueries NQUERIES
                        the number of queries to run in this experiment.
                        default: 20
```

#### `python scripts/gen_n_figure.py`

```
usage: gen_n_figure.py [-h] [-o OUTPUT] JSON

A script to generate a plot of number of positive queries vs number of
queries.

positional arguments:
  JSON                  history json file (output from run_experiment.py)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output file. default: timestamp (e.g.
```

#### `python scripts/gen_sparse_figure.py` 

```
usage: gen_sparse_figure.py [-h] [-o OUTPUT] JSON

A script to generate a plot of sparsity of features vs number of queries.

positional arguments:
  JSON                  history json file (output from run_experiment.py)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output file. default: timestamp (e.g.
                        '0319-1343plot.png')
```

## Configure the Scripts

There are a few things you need to manually configure right now.

To define which algorithms the experiment uses, edit the `algs = []` list in `run_experiment.py`

In `lasso_ir/constants.py`, you can edit some constants:

 - MAX_N_CIFAR_TRAIN: The maximum number of examples to load from the data.  Make this small for debugging because it can take a while to load all the data.
 - CIFAR_POSITIVE_CLASSES: A subset of `{0,1,2,3,4,5,6,7,8,9}` to define what are the "positive" classes of the CIFAR dataset.
 - MULTIPROCESSING: If True, each algorithms' `get_query` will be run concurrently. If False, sequentially. If False, the grid searches for selecting the LASSO parameter will be run in parallel. I haven't tested which is fastest!

## Example

Here's a quick example on how to run a small experiment and generate the plots.

```
$ python scripts/run_experiment.py -n 100 -o example
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:51<00:00,  1.94it/s]
saved history to examplehistory.json
$ python scripts/gen_n_figure.py examplehistory.json -o n_positives
saving figure to n_positives.png
$ python scripts/gen_sparse_figure.py examplehistory.json -o n_sparse
saving figure to n_sparse.png
```

