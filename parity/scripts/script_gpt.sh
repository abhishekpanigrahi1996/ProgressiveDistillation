export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib



randomize_features=0
feature_coordinates='0_1_2_3_4_5'
feature_complexity=6
num_labels=2


# TODO: change this to your own
local_dir=''
logging_path='logs_parity/log_save100k'
output_path='ckpts_parity/ckpts_save100k'


data_dimension=100
vocab_size=2
use_cls_head=0
tie_word_embeddings=0
add_cls_token=0


n_steps=8000000
n_examples=$n_steps


linear_mlp=0
skip_mlp=0

eval_batch_size=128
num_workers=16


for seed in 0
do
for head_dim in 8
do
for num_layers in 2
do 
for batch_size in 32
do
for learning_rate in 3e-4
do
for n_heads in 16
do 
for subsample in -1
do 
for weight_decay in 0
do
hidden_size=$((n_heads * head_dim))
warmup_ratio=0.06


save_intvl=$((100000 / $batch_size))
log_intvl=$((100000 / $batch_size))



WANDB_MODE='disabled' \
CUDA_VISIBLE_DEVICES=0 \
python boolean_expts.py \
    --model_type='gpt2' \
    --hidden_size=$hidden_size \
    --head_dim=$head_dim \
    --learning_rate=$learning_rate \
    --weight_decay=$weight_decay \
    --logging_path=$logging_path \
    --output_path=$output_path \
    --seed=$seed \
    --n_heads=$n_heads \
    --use_cls_head=$use_cls_head \
    --add_cls_token=$add_cls_token \
    --tie_word_embeddings=$tie_word_embeddings \
    --linear_mlp=$linear_mlp \
    --skip_mlp=$skip_mlp \
    --vocab_size=$vocab_size \
    --num_labels=$num_labels \
    --feature_complexity=$feature_complexity \
    --randomize_features=$randomize_features \
    --feature_coordinates=$feature_coordinates \
    --data_dimension=$data_dimension \
    --n_steps=$n_steps \
    --n_examples=$n_examples \
    --num_layers=$num_layers \
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
done 
done