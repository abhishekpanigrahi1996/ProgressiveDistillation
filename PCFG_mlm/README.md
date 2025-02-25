## Please follow the following steps.
1. Generate PCFG training data

Go to "data_generation"
Run "python create_mlm.py"

Important arguments are:
pcfg_def = Please use one of the files from PCFG_def, e.g. PCFG_def/yuanzhi_cfg3b.pkl
num_train_examples = Number of training examples


2. Train a teacher model

Go to "scripts"
Run script train.sh

Please check train.sh to modify necessary arguments as needed

3. Distil to a student model

Run script distil.sh

Please check distil.sh to modify necessary arguments as needed

4. Non-terminal prediction

Go to "data_generation"
Run "nt_classify.py"

Important arguments are:
pcfg_def = Please use one of the files from PCFG_def, e.g. PCFG_def/yuanzhi_cfg3b.pkl
num_train_examples = Number of training examples

Go to "probing/scripts"
Run train_boundary.sh

Please check the script for important arguments to modify

output_pth will contain the non-terminal prediction scores

5. Compute M_close/M_robust

Go to "data_generation"
Run "ngram_gen.py"

Important arguments are:
pcfg_def = Please use one of the files from PCFG_def, e.g. PCFG_def/yuanzhi_cfg3b.pkl
robust_or_close = If True, we generate data for M_robust, else we generate data for M_close


Go to "probing/scripts"
Run ngram_prediction.sh

Please check the script for important arguments to modify

output_pth will contain the M_robust/M_close prediction scores




