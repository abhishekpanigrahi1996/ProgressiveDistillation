# Argument definitions (please set them accordingly)
# PROJ_DIR:  absolute path to directory on probing
# DATA_DIR:  absolute path to directory on data_generation
# MODEL_DIR: absolute path to location where you save the checkpoints
# TRAIN_FILE: path to the training data file
# EVAL_FILE: path to the evaluation data file


# save_freq: frequency to save checkpoints (200 for our paper)
# nlayers: number of layers 
# subsample: number of training data points (e.g. 1_024_000, 2_048_000, 4_096_000, 8_192_000)
# n_heads: number of heads in the model (e.g. 4, 8, 16, 32)
# head_dim: dimension per head (e.g. 8, 32)
# lr:  learning rate
# seed: randomness seed

# TODO: please change these accordingly
PROJ_DIR='/scratch/gpfs/ap34/ProgressiveDistillation/PCFG_autoregressive'

DATA_DIR_BASE=$PROJ_DIR'/data_generation/data/'
MODEL_DIR=$PROJ_DIR
RUN_DIR=$PROJ_DIR
MODEL='openai-community/gpt2'
data_seed=0
pcfg_type='yuanzhi_cfg3f'
TRAIN_FILE=$DATA_DIR_BASE'/yuanzhi_cfg3f.pkl/train_seed5_boundarylabels_3.pt';
EVAL_FILE=$DATA_DIR_BASE'/yuanzhi_cfg3f.pkl/eval_seed5_boundarylabels_3.pt';

BATCH_SIZE=8     #<-- Instantaneous batch size  
ACCU_STEPS=64    #<-- Number of accumulate steps per update
CACHE_DIR='/scratch/gpfs/ap34/hf_models'    #<-- cache directory, to save downloaded bert

save_freq=4   #<-- frequency of saving checkpoint

vocab_size=8     #<-- vocab size, =8 for our synthetic experiments. 


# sleep 7200

for seed in 0
do
for lr in 5e-3 
do 

for nlayers in 4 
do
for subsample in 1_024_000
do
for n_heads in  32
do
for head_dim in 8
do


hid_size=$(( head_dim * n_heads ))


subdir_log_name=$pcfg_type"_ce_lr"$lr"_hid"$hid_size"_nL"$nlayers"_samples"$subsample"_nH"$n_heads"_seed"$seed"_bt"${BATCH_SIZE}"_accsteps"${ACCU_STEPS}'_'${pcfg_type}
OUTPUT_DIR=$MODEL_DIR"/output/test/"$subdir_log_name
LOGGING_DIR=$MODEL_DIR"/output/logs/"$subdir_log_name

header="python $RUN_DIR/main_pcfg_ce.py"

${header}  \
    --do_train True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.06 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_sequence_length 1024 \
    --logging_steps 100 \
    --save_strategy "steps" \
    --save_steps $save_freq \
    --num_train_epochs 1 \
    --vocab_size ${vocab_size} \
    --overwrite_output_dir True \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --seed $seed \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    --subsample $subsample \
    --model_name_or_path $MODEL \
    --parent_path $MODEL\
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --learning_rate $lr \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCU_STEPS \
    --cache_dir $CACHE_DIR \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --per_device_eval_batch_size 32 \
    --preprocessing_num_workers 20 \
    --hidden_size $hid_size \
    --n_layers $nlayers \
    --n_heads $n_heads \
    --run_name $subdir_log_name \
    --project_name 'Autoregressive_gpt_'${pcfg_type}'_subsample'${total_samples} \
    --pad_token_id 5\
    --mask_token_id 6\
    --cls_token_id 5\

done
# wait

done 
done 
done
done
done
