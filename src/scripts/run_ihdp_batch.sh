#!/usr/bin/env bash
#SBATCH --mem=2GB
#SBATCH --cpus-per-task 2
#SBATCH --time=00:10:00
#SBATCH --array=0-100

CHUNKSIZE=1
n=$SLURM_ARRAY_TASK_ID
min_ind=$((n*CHUNKSIZE))
max_ind=$(((n+1)*CHUNKSIZE))

 EXPSETUP="exp_yamls/full_ihdp.yaml" # The original setting with 25 predictive covariates
# EXPSETUP="exp_yamls/partial_ihdp.yaml"
# EXPSETUP="exp_yamls/tte_ihdp.yaml"

srun \
python benchruns/exp_benchmark.py \
    --exp_setup "$EXPSETUP" \
    --verbose \
    --min_id "$min_ind" \
    --max_id "$max_ind"
