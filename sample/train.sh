#!/bin/bash

source activate cuda11_env

export CANDLE_DATA_DIR=""

pyScript="${1}/src/train.py"

python -u $pyScript > "train.log"
