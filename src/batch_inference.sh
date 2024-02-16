#!/bin/bash

echo "===== shell argv ====="
echo "config: $1"

python3 preprocess.py --config $1 --predict

python3 batch_inference.py --config $1
