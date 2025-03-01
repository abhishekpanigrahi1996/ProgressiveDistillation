
PROJ_DIR='parity_mlp'  #<-- absolute path the project directory

n_examples=20_000_000    #<-- number of examples to generate (e.g. 10M, 20M; 125*num_labels will be used as evaluation data and rest for training)
hidden_size=50_000       #<-- hidden size of the MLP to train
data_dimension=100       #<-- dimension of training data (100 in our paper)
num_labels=2             #<-- number of labels (set as 2 for 0/1 classification, set as 2^k for a k-depth hierarchical tree classification)
learning_rate=1e-3       #<-- learning rate to train (e.g. 1e-2, 5e-3, 1e-3)
seed=42                  #<-- randomness seed 
num_layers=1             #<-- modify for multilayer training
feature_complexity=6     #<-- modify if you want to change the support size of sparse parity
randomize_features=0     #<-- if set as 1, parity coordinates are selected at random; if not, the first 'feature_complexity' coordinates form sparse parity support
logging_path=$PROJ_DIR'/logs/logs_sparsehierarchy_100seeds_unsharedvars/log'         #<-- logging directory
output_path=$PROJ_DIR'/output/ckpts_sparsehierarchy_100seeds_unsharedvars/ckpts'     #<-- checkpoint save directory
save_freq=100_000        #<-- frequency of saving the model checkpoints and wandb loss

temp=1e-20                 #<-- temperature for distillation
teacher_ckpt_step=100_000  #<-- first teacher checkpoint to use, if method is progressive distillation, checkpoints are changed in intervals of  teacher_ckpt_step steps.

teacher_hidden_size=50_000 #<-- hidden size of teacher model 
teacher_ckpt_dir=$PROJ_DIR'/output/ckpts_sparsehierarchy_100seeds_unsharedvars/ckpts_numexp1000_hid50000_dim100_num_labels2_lr0.01_seed42_num_layers1_feature_complexity6_randomizefeatures_False_reg_0.0/'
method='progressive_distillation' #<-- set as distillation or progressive_distillation 

python $PROJ_DIR/main_distil.py ${n_examples} ${hidden_size} ${data_dimension} ${num_labels} ${learning_rate} ${logging_path} ${output_path} ${seed} ${num_layers} ${teacher_hidden_size} ${teacher_ckpt_dir} ${method}  ${teacher_ckpt_step} ${feature_complexity} ${randomize_features} ${temp} ${save_freq}
