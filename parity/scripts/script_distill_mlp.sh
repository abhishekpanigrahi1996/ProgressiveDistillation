export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib

DEV=0
n_gpus=1
gpu_offset=0

# TODO: set your own wandb project and entity here
wandb_project=''
wandb_entity=''
logging_path="logs_parity_distill_supp/log"
output_path="ckpts_parity_distill_supp/ckpts"

### Data-related

randomize_features=0
feature_complexity=6
num_labels=2

n_steps=8000000
if [ $DEV = 1 ]; then
    n_steps=10000
fi
n_examples=$n_steps
n_epochs=1
subsample=-1


batch_size=1
num_workers=16


### Model-related


# size of the teacher
teacher_hidden_size=50000
teacher_num_layers=1 # 1 hidden layer
# TODO: update your local dir here
if [ $teacher_hidden_size = 50000 ] && [ $teacher_num_layers = 1 ]; then
  if [ $feature_complexity = 6 ]; then 
    teacher_ckpt_dir='your_dir'
    intvl='2-shot' # this is only a token & doesn't mean much.
    teacher_ckpt_step='"your_ckpts"'
  fi
fi
teacher_ckpt_dir='your_base_dir'$teacher_ckpt_dir

wandb_mode='online'
if [ $DEV = 1 ]; then
    n_examples=4096
    n_steps=2000
    wandb_mode='disabled'
fi


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

hidden_size=1000
for num_layers in 1
do
for ckpt_multiplier in 1
do
for learning_rate in 0.005 0.01
do
for seed in 0 1
do
for kl_alpha in 0
do
for temperature in 1
do
    device_id=$((cnt % n_gpus))
    device_id=$((device_id + gpu_offset))

    WANDB_MODE=$wandb_mode \
    CUDA_VISIBLE_DEVICES=$device_id \
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
        model.type='mlp' \
        model.num_layers=$num_layers \
        model.hidden_size=$hidden_size \
        model.teacher_hidden_size=$teacher_hidden_size \
        model.teacher_num_layers=$teacher_num_layers \
        task.ckpt_multiplier=$ckpt_multiplier \
        task.teacher_ckpt_step=$teacher_ckpt_step \
        task.teacher_ckpt_dir=$teacher_ckpt_dir \
        task.teacher_ckpt_intvl=$intvl \
        task.kl_alpha=$kl_alpha \
        task.kl_alpha_incre=$kl_alpha_incre \
        task.kl_alpha_incre_from=$kl_alpha_incre_from \
        task.kl_alpha_incre_intvl=$kl_alpha_incre_intvl \
        logging.freq=50000 \
        logging.ckpt_freq=100000 \
        logging.logging_path=$logging_path \
        logging.output_path=$output_path \
        logging.wandb_project=$wandb_project \
        logging.wandb_entity=$wandb_entity &
    if [ $DEV = 1 ]; then
        exit
    fi

    cnt=$((cnt+1))
done 
done
done
done
done
done 
