# Estimating treatment effects from single-arm trials via latent-variable modeling

This repo contains a reference implementation for 

**Estimating treatment effects from single-arm trials via latent-variable modeling**  
_Manuel Haussmann, Tran Minh Son Le, Viivi Halla-aho, Samu Kurki, Jussi Leinonen, Miika Koskinen, Samuel Kaski, Harri Lähdesmäki_  
_27th International Conference on Artificial Intelligence and Statistics_  
[Paper](https://proceedings.mlr.press/v238/haussmann24a.html)


## Usage

### Public benchmark example: IHDP

The following steps run through the whole pipeline once from raw files to final predicitons

1. Run `benchmark_scripts/get_ihdp1000.sh` to download and extract the data
2. Run `src/scripts/prep_data_ihdp_local.sh` to preprocess it.
   (`extcontcode/benchdata/ihdp/ihdp.py` contains the python scripts used for preprocessing)
3. Run `src/sripts/run_ihdp.sh` to train models according to the specifications in a
   separate yaml file. (See `benchruns/exp_benchmark.py` for details)
4. Run `src/scripts/eval_ihdp.sh` for an evaluation routine

See `benchruns/exp_benchmark.py` for an example on how to train a model and how to use
`benchruns/eval_ihdp.py` as a second step for evaluation.

