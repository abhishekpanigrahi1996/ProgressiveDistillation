# Argument definitions (please set them accordingly)
# PROJ_DIR:  absolute path to directory on probing
# DATA_DIR:  absolute path to directory on data_generation
# MODEL_DIR: absolute path to location where you save the checkpoints
# learning_rate: learning rate to train the probe (1e-2 in the paper)


PROJ_DIR=PCFG_mlm/probing/
DATA_DIR='PCFG_mlm/data_generation/'
MODEL_DIR="PCFG_mlm/"

RUN_DIR=$PROJ_DIR
learning_rate=1e-2
model_name_or_path=$MODEL_DIR"output/test/yuanzhi_cfg3b.pkl_mlm_lr5e-3_hid256_nL4_samples1_024_000_nH32_seed0_bt8_accsteps64_yuanzhi_cfg3b/"

TRAIN_FILE=$DATA_DIR'data/yuanzhi_cfg3b.pkl/train_seed0_boundary.pt' 
EVAL_FILE=$DATA_DIR'data/yuanzhi_cfg3b.pkl/eval_seed0_boundary.pt'

num_predictions=7
num_positions=450
position_embd=256
logging_dir='Run'
num_NTs=30
output_pth=$PROJ_DIR'output/test/NT_prediction.json'
python $RUN_DIR"boundary_nt_prediction.py" ${TRAIN_FILE} ${EVAL_FILE} ${model_name_or_path} ${num_predictions} ${num_positions} ${position_embd} ${learning_rate} ${logging_dir} ${num_NTs}  ${output_pth}
