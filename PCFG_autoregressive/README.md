1. Generate data

- Go to "data_generation"
- Run "python create_ce.py"

Important arguments are:
- pcfg_def = Please use one of the files from PCFG_def, e.g. PCFG_def/yuanzhi_cfg3b.pkl
- num_train_examples = Number of training examples
- desired_level = Predictions are restricted to the tokens that are at the right boundaries of spans of non terminals that appear at level 'desired_level'. Setting it as 0 will be the general auto-regressive training (results are reported for 1/2/3 are reported in the main paper).

2. Train a teacher model

- Go to "scripts"
- Run script main_ce.sh

Please check main_ce.sh to modify necessary arguments as needed

3. Distil to a student model

- Go to "scripts"
- Run script distil_ce.sh

Please check distil_ce.sh to modify necessary arguments as needed
