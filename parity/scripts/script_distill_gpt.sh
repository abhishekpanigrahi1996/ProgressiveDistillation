export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib

# TODO: change this to your own
logging_path="logs_parity_distill_supp/log"
output_path=''

### Data-related

randomize_features=0
feature_complexity=6
num_labels=2

n_steps=8000000
n_examples=$n_steps
n_epochs=1
subsample=-1


batch_size=32
num_workers=16


### Model-related

vocab_size=2
tie_word_embeddings=0


# size of the teacher
teacher_hidden_size=256
teacher_n_heads=32
teacher_num_layers=2
# TODO: update your local dir here
if [ $teacher_hidden_size = 256 ] && [ $teacher_n_heads = 32 ]; then
  if [ $feature_complexity = 6 ]; then 
    teacher_ckpt_dir='your_dir'
    intvl='2-shot' # this is only a token & doesn't mean much.
    teacher_ckpt_step='"your_ckpts"'
  fi
fi
teacher_ckpt_dir='your_base_dir'$teacher_ckpt_dir


kl_type='forward'
kl_alpha_incre=0
if [ $kl_alpha_incre = 0 ];
then 
  kl_alpha_incre_from=0
  kl_alpha_incre_intvl=0
else 
  kl_alpha_incre_from=3200000
  kl_alpha_incre_intvl=100000
fi

cnt=0

for seed in 0
do
for num_layers in 2
do
for head_dim in 8
do
for ckpt_multiplier in 1
do
for learning_rate in 1e-4
do
for n_heads in 4
do 
hidden_size=$((n_heads * head_dim))
for kl_alpha in 0
do
for temperature in 1
do

    WANDB_MODE='disabled' \
    CUDA_VISIBLE_DEVICES=0 \
    python boolean_distill_expts.py \
        training.n_examples=$n_examples \
        training.seed=$seed \
        training.n_epochs=$n_epochs \
        training.learning_rate=$learning_rate \
        training.batch_size=$batch_size \
        training.temperature=$temperature \
        data.num_labels=$num_labels \
        data.randomize_features=$randomize_features \
        data.feature_complexity=$feature_complexity \
        data.subsample=$subsample \
        data.num_workers=$num_workers \
        model.type='gpt2' \
        model.num_layers=$num_layers \
        model.hidden_size=$hidden_size \
        model.head_dim=$head_dim \
        model.teacher_hidden_size=$teacher_hidden_size \
        model.n_heads=$n_heads \
        model.teacher_n_heads=$teacher_n_heads \
        model.teacher_num_layers=$teacher_num_layers \
        model.vocab_size=$vocab_size \
        model.tie_word_embeddings=$tie_word_embeddings \
        task.ckpt_multiplier=$ckpt_multiplier \
        task.teacher_ckpt_step=$teacher_ckpt_step \
        task.teacher_ckpt_dir=$teacher_ckpt_dir \
        task.teacher_ckpt_intvl=$intvl \
        task.kl_alpha=$kl_alpha \
        task.kl_alpha_incre=$kl_alpha_incre \
        task.kl_alpha_incre_from=$kl_alpha_incre_from \
        task.kl_alpha_incre_intvl=$kl_alpha_incre_intvl \
        logging.freq=2500 \
        logging.ckpt_freq=5000 \
        logging.logging_path=$logging_path \
        logging.output_path=$output_path
done 
done
done
done
done
done 
done
done
