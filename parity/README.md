## Please follow the following steps.

1. Train a teacher model

Run script `./scripts/script_{mlp/gpt}.sh` to train a MLP or GPT-2 model on parity.

Please modify the bash script for necessary changes (e.g. file paths, hyperparameters, etc.).

2. Distil to a student model

Run script `./scripts/script_distil_{mlp/gpt}.sh`

Please modify the bash script for necessary changes (e.g. file paths, hyperparameters, etc.).

3. Compute correlations

Please run `python probing.py` to get correlations to low degree polynomials on the output of a checkpoint.