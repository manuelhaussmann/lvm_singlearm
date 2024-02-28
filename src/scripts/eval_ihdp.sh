#!/usr/bin/env bash


# Specify shift (sh) and flip probbality (pf)
sh=0
pf=0.0
loader="train_loader"
srun -c 4 --time=4:00:00 --mem=10GB \
python benchruns/eval_ihdp.py --save_name "../runs/ihdp/result" \
  --max_id 100 \
  --n_pred 25 \
  --setting "full" \
  --zdim 5 \
  --shift $sh \
  --flip $pf \
  --tte_partial \
  --save \
  --loader "$loader"
