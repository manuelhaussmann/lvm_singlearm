#!/usr/bin/env bash

SUBSET=100
PERCVAL=0.1
BATCH=100
NPRED=25
SHIFT=0
FLIP=0.0

sh=$SHIFT
pf=$FLIP
srun -c 4 --time=1:00:00 --mem=20GB \
python extcontcode/benchdata/ihdp/ihdp.py \
          --subset $SUBSET \
          --perc_val $PERCVAL \
          --batch_size $BATCH \
          --shift_size $sh \
          --prob_flip $pf \
          --n_pred $NPRED \
          --verbose

