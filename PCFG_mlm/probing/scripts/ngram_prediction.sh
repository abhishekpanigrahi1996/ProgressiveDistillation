# Argument definitions (please set them accordingly)
# PROJ_DIR:  absolute path to directory on probing
# DATA_DIR:  absolute path to directory on data_generation
# MODEL_DIR: absolute path to location where you save the checkpoints
# learning_rate: learning rate to train the probe (1e-2 in the paper)
# eval_file: set accordingly, whether you use the evaluation file for M_robust or M_close


PROJ_DIR=PCFG_mlm/probing/
DATA_DIR='PCFG_mlm/data_generation/'
MODEL_DIR="PCFG_mlm/"


RUN_DIR=$PROJ_DIR
seed=0

eval_file=$PROJ_DIR'/data_generation/data/yuanzhi_cfg3b.pkl/eval_seed5_mlm_MI_ngramrobust.pt'
model_name_or_path=$MODEL_DIR"output/test/yuanzhi_cfg3b.pkl_mlm_lr5e-3_hid256_nL4_samples1_024_000_nH32_seed0_bt8_accsteps64_yuanzhi_cfg3b/"

logging_dir='log'
output_pth=$PROJ_DIR'output/test/yuanzhi_cfg3b_ngramrobust.json'

python $RUN_DIR"ngram_prediction.py" ${eval_file} ${model_name_or_path} ${logging_dir} -1 ${seed} ${output_pth}
