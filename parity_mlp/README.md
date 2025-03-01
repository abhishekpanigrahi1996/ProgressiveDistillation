## Please follow the following steps.

1. Train a teacher model

Go to "scripts"
Run script ce_train.sh

Please check ce_train.sh to modify necessary arguments as needed

2. Distil to a student model

Run script distil.sh

Please check distil.sh to modify necessary arguments as needed

3. Compute correlations

Please run python probing.py to get correlations to low degree polynomials on the output of a checkpoint