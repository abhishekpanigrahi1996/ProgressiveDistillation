export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib

DEV=0
n_gpus=1
device_shift=0

data_dimension=100
randomize_features=0
feature_coordinates='0_1_2_3_4_5'
feature_complexity=6
num_labels=2


# TODO: change this to your own
local_dir=''
save_intvl=100000
log_intvl=100000
logging_path='logs_parity/log_save100k'
output_path='ckpts_parity/ckpts_save100k'


n_steps=8000000
n_examples=$n_steps


hidden_size=50000

for seed in 0
do
for num_hidden_layers in 1
do 
for batch_size in 1
do
for learning_rate in 0.005
do
for subsample in -1
do 
for weight_decay in 0
do
warmup_ratio=0

eval_batch_size=128

num_workers=16


WANDB_MODE='disabled' \
CUDA_VISIBLE_DEVICES=0 \
python boolean_expts.py \
    --model_type='mlp' \
    --anneal_type='constant' \
    --hidden_size=$hidden_size \
    --num_layers=$num_hidden_layers \
    --learning_rate=$learning_rate \
    --weight_decay=$weight_decay \
    --logging_path=$logging_path \
    --output_path=$output_path \
    --seed=$seed \
    --num_labels=$num_labels \
    --feature_complexity=$feature_complexity \
    --randomize_features=$randomize_features \
    --feature_coordinates=$feature_coordinates \
    --data_dimension=$data_dimension \
    --n_steps=$n_steps \
    --n_examples=$n_examples \
    --num_workers=$num_workers \
    --batch_size=$batch_size \
    --eval_batch_size=$eval_batch_size \
    --subsample=$subsample \
    --log_intvl=$log_intvl \
    --save_intvl=$save_intvl \
    --warmup_ratio=$warmup_ratio

done 
done
done
done 
done
done