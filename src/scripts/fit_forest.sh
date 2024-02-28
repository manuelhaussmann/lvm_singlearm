#!/usr/bin/env bash

MAX_ID=10

srun -c 4 --time=4:00:00 --mem=10GB \
  python extcontcode/models/causalforest.py \
    --max_id $MAX_ID \
    --npred 25 \
    --shift 0 \
    --flip 0.0 \
    --verbose
