#!/usr/bin/env bash
conda create -n lasso-ir python=3 -y
source activate lasso-ir
yes w | pip install -r requirements.txt
source ./activate.sh
