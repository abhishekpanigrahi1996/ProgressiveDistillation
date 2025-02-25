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
PROJ_DIR=PCFG_mlm/
DATA_DIR=$PROJ_DIR'data_generation/data/'
MODEL_DIR=$PROJ_DIR


RUN_DIR=$PROJ_DIR
MODEL=bert-base-uncased
PARENT_MODEL=bert-base-uncased
data_seed=0
pcfg_type='yuanzhi_cfg3b'

TRAIN_FILE=$DATA_DIR'yuanzhi_cfg3b.pkl/train_seed5_mlm_maskrate0.3_80_10_10.pt'
EVAL_FILE=$DATA_DIR'yuanzhi_cfg3b.pkl/eval_seed5_mlm_maskrate0.3_80_10_10.pt';


BATCH_SIZE=8     #<-- Instantaneous batch size  
ACCU_STEPS=64    #<-- Number of accumulate steps per update
CACHE_DIR=$RUN_DIR'hf_models'  #<-- cache directory, to save downloaded bert

save_freq=32_000_000   #<-- frequency of saving checkpoint

speedfactor=1        #<-- Speedfactor (set as 1 in main paper for progressive distillation, 2 for appendix); For a teacher checkpoint at T steps, we supervise the student with T/speedfactor steps; 
                     #set as 1000 for one-shot distillation
refresh_frequency=5  # < -- the refresh frequency for changing the teacher, set as 1000 (checkpoints selected will be 1k,2k,3k,4k,5k,6k,7k,8k)

teacher_hidsize=256    #<-- hidden size of teacher (=num heads x dimension per head)
teacher_nlayers=4      #<-- hidden size of teacher (=num layers in teacher)
teacher_n_heads=32     #<-- hidden size of teacher (=num heads in teacher)

vocab_size=8           #<-- vocab size, =8 for our synthetic experiments.
 
ckpt_idx='1000'        #<-- index of first checkpoint
REF_CKPT_DIR=$RUN_DIR'output/test/yuanzhi_cfg3b.pkl_mlm_lr5e-3_hid256_nL4_samples1_024_000_nH32_seed0_bt8_accsteps64_yuanzhi_cfg3b' #<-- teacher checkpoint 
MODEL=${REF_CKPT_DIR}"/checkpoint-"$ckpt_idx    
cnt=1


for seed in 42
do

for lr in 5e-3
do 

for head_dim in 8
do 

for nlayers in 4
do

for subsample in 1_024_000
do

for n_heads in 8
do

temp=1e-20        #<-- temperature for distillation
hid_size=$(( head_dim * n_heads ))

OUTPUT_DIR=$RUN_DIR"/output/test_"${subsample}"/"$pcfg_type"_pcfg_mlm_distil_1epoch_lr"${lr}"_hid"${hid_size}"_nlayers"${nlayers}"_temp"${temp}"_speedfactor"${speedfactor}"_refreshfreq"${refresh_frequency}"_teacher"$teacher_hidsize"hidnlayers"$teacher_nlayers'_pcfgtype'$pcfg_type'_seed'${seed}'_'${pcfg_type}
LOGGING_DIR=$RUN_DIR"/output/logs_"${subsample}"/"$pcfg_type"_pcfg_mlm_distil_1epoch_lr"${lr}"_hid"${hid_size}"_nlayers"${nlayers}"_temp"${temp}"_speedfactor"${speedfactor}"_refreshfreq"${refresh_frequency}"_teacher"$teacher_hidsize"hidnlayers"$teacher_nlayers'_pcfgtype'$pcfg_type'_seed'${seed}'_'${pcfg_type}

python $RUN_DIR/distil_pcfg_mlm.py \
    --do_train True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.06\
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_sequence_length 1024 \
    --logging_steps 100 \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps ${save_freq}\
    --num_train_epochs 1\
    --overwrite_output_dir True \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --seed $seed \
    --model_name_or_path $MODEL \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --learning_rate $lr \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCU_STEPS \
    --cache_dir $CACHE_DIR \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 32\
    --parent_path $PARENT_MODEL\
    --temperature ${temp} \
    --hidden_size $hid_size \
    --n_layers $nlayers\
    --n_heads $n_heads \
    --ref_model_ckpt_dir $REF_CKPT_DIR\
    --ref_hidden_size ${teacher_hidsize}\
    --ref_n_layers ${teacher_nlayers}\
    --speed_factor ${speedfactor}\
    --refresh_frequency ${refresh_frequency}\
    --subsample $subsample \
    --n_heads $n_heads \
    --run_name $pcfg_type\
    --pcfg_name $pcfg_type \
    --vocab_size ${vocab_size} \
    --pad_token_id 5\
    --mask_token_id 6\
    --cls_token_id 5\
    --project_name 'distil_PCFG_mlm_repeat_'${pcfg_type}'_'${subsample}'_temp'${temp};

    #if [ $DEV = 1 ]; then
    #    exit
    #fi

done
# wait

done 
done 
done
done
done

