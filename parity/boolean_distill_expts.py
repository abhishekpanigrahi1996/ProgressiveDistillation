import torch
import os
import numpy as np
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from utils import prepare_model, get_data_loaders, get_logits_gpt, accuracy, loss_fn, set_seed

import pdb

try:
    import wandb
    USE_WANDB = True
except:
    USE_WANDB = False
    os.environ["WANDB_DISABLED"] = "true"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




@hydra.main(config_path="conf", config_name="distill_boolean_data.yml")
def main(cfg: DictConfig):
    cwd = os.getcwd()

    cfg = OmegaConf.to_container(cfg)
    data_dimension = cfg['data']['data_dimension']
    boolean_sparsity = cfg['data']['boolean_sparsity']
    num_labels = cfg['data']['num_labels']
    feature_complexity = cfg['data']['feature_complexity']
    subsample = cfg['data']['subsample']

    learning_rate = cfg['training']['learning_rate']
    seed = cfg['training']['seed']
    n_steps = cfg['training']['n_steps']
    n_examples = cfg['training']['n_examples']
    temperature = cfg['training']['temperature']
    teacher_temperature_anneal = cfg['training']['teacher_temperature_anneal']
    teacher_temperature_anneal_type = cfg['training']['teacher_temperature_anneal_type']
    warmup_ratio = cfg['training']['warmup_ratio']

    logging_path = cfg['logging']['logging_path']
    logging_freq = cfg['logging']['freq']
    ckpt_freq = cfg['logging']['ckpt_freq']
    save_ckpt_freq = cfg['logging']['save_ckpt_freq']
    
    model_type = cfg['model']['type']
    num_layers = cfg['model']['num_layers']
    teacher_num_layers = cfg['model']['teacher_num_layers']
    hidden_size = cfg['model']['hidden_size']
    teacher_hidden_size = cfg['model']['teacher_hidden_size']
    n_heads = cfg['model']['n_heads']
    head_dim = cfg['model']['head_dim']
    if model_type == 'gpt2':
       if head_dim != hidden_size // n_heads:
          print("Warning: head_dim is not equal to hidden_size // n_heads")
          print(f"head_dim: {head_dim}, hidden_size: {hidden_size}, n_heads: {n_heads}")
          print(f"Setting head_dim to be hidden_size // n_heads = {hidden_size // n_heads}")
          print("Note -- head_dim is for logging purpose only")
          cfg['model']['head_dim'] = hidden_size // n_heads
    teacher_n_heads = cfg['model']['teacher_n_heads']
    vocab_size = cfg['model']['vocab_size']
    tie_word_embeddings = cfg['model']['tie_word_embeddings']
    
    teacher_type = cfg['task']['teacher_type']
    if teacher_type == 'ckpt':
      teacher_ckpt_dir = cfg['task']['teacher_ckpt_dir']
      teacher_ckpt_step = cfg['task']['teacher_ckpt_step']
      teacher_ckpt_step = [int(step) for step in teacher_ckpt_step.split('_')]
      ckpt_multiplier = cfg['task']['ckpt_multiplier']
    elif teacher_type in ['linear_anneal', 'fixed_temp']:
      max_anneal_temperature = cfg['task']['max_anneal_temperature']
    else:
        raise ValueError(f"Invalid teacher_type: {teacher_type}")
    method = cfg['task']['method']
    kl_type = cfg['task']['kl_type']
    kl_alpha = cfg['task']['kl_alpha']

        
    set_seed(seed)
    dtype = torch.float32 if model_type == 'mlp' else torch.long

    """
    Get data
    """
    train_loader, eval_loader = get_data_loaders(cfg, seed)


    """
    Get model
    """
    all_layers, all_parameters = prepare_model(data_dimension, num_labels,
                                               hidden_size, num_layers,
                                               model_type=model_type, n_heads=n_heads,
                                               tie_word_embeddings=tie_word_embeddings,
                                               vocab_size=vocab_size)
    if model_type == 'mlp':
        teacher_layers, _ = prepare_model(data_dimension, num_labels,
                                              teacher_hidden_size, teacher_num_layers,
                                              model_type=model_type)
    if 'gpt2' in model_type:
      optimizer = torch.optim.Adam(all_parameters, lr=learning_rate,
                                   weight_decay=cfg['training']['weight_decay'])
    else:
      optimizer = torch.optim.SGD(all_parameters, lr=learning_rate,
                                  weight_decay=cfg['training']['weight_decay'])


    n_epochs = cfg['training']['n_epochs']
    print("# Epochs:", n_epochs)
    if n_steps > n_epochs * len(train_loader):
        n_steps = n_epochs * len(train_loader)


    """
    Set logging & saving before training starts
    """

    pid = os.getpid()
    cfg['training']['pid'] = pid

    # Set output_path for saved ckpts
    output_path = cfg['logging']['output_path']
    save_suffix = f'_hid{hidden_size}' + f'_n{data_dimension}_k{feature_complexity}' + f'_num_labels{num_labels}' + f'_lr{learning_rate}' + f'_seed{seed}' + f'_num_layers{num_layers}' + f'_e{n_epochs}'
    if model_type == 'gpt2':
        save_suffix += '_nheads' + str(n_heads)
        if tie_word_embeddings:
            save_suffix += '_tied'
    if subsample != -1:
        save_suffix += '_subsample' + str(subsample)
    save_suffix += '_kl' + str(kl_alpha)
    logging_dir = logging_path + save_suffix
    cwd = os.getcwd()
    output_path = os.path.join(cwd, output_path+save_suffix)
    os.makedirs(output_path, exist_ok=True)
    cfg['logging']['output_path'] = output_path

    if USE_WANDB:
        wandb.init(project=cfg['logging']['wandb_project'], name=logging_dir, entity=cfg['logging']['wandb_entity'])
        wandb.config.update(cfg)



    global_step_cnt = -1
    prev_kl_alpha = kl_alpha
    eval_accs, curr_eval_acc, last_eval_acc = [], 0, 0
    curr_ckpt_idx, curr_ckpt_step = -1, ''
    last_update_step = 0
    for epoch in tqdm(range(n_epochs)):
        for bt, batch in tqdm(enumerate(train_loader)):
            
            """
            Check for ckpt
            """
            if teacher_type == 'ckpt' and (global_step_cnt==-1 or global_step_cnt % ckpt_freq == 0):
                if method == 'progressive_distillation':
                    frst_ind = 0
                    while max(global_step_cnt, 1)*ckpt_multiplier > teacher_ckpt_step[frst_ind] and frst_ind < len(teacher_ckpt_step) - 1:
                        frst_ind += 1

                    ckpt_step = teacher_ckpt_step[frst_ind]
                    ckpt_dir = os.path.join(teacher_ckpt_dir, f'checkpoint-{ckpt_step}')
                    if os.path.exists(ckpt_dir):
                      valid_ckpt_dir = ckpt_dir
                    else:
                        # Find the last valid ckpt
                        curr_ind = frst_ind
                        ckpt_found = 0
                        while curr_ind > 0:
                            curr_ind -= 1
                            ckpt_step = teacher_ckpt_step[curr_ind]
                            ckpt_dir = os.path.join(teacher_ckpt_dir, f'checkpoint-{ckpt_step}')
                            if os.path.exists(ckpt_dir):
                                ckpt_found = 1
                                valid_ckpt_dir = ckpt_dir
                                break
                        if not ckpt_found:
                            raise ValueError(f"No checkpoint found: {ckpt_dir}") 
                    
                    curr_ckpt_step = ckpt_step
                    
                elif method == 'progressive_till_saturate':
                    smooth_window_size = 5
                    if len(eval_accs) < smooth_window_size+1:
                        continue
                    avg_prev_acc = sum(eval_accs[-smooth_window_size-1:-1])/smooth_window_size
                    if last_eval_acc == 0:
                        last_eval_acc = avg_prev_acc
                    if (curr_eval_acc - avg_prev_acc < 0.01 and curr_eval_acc - last_eval_acc > 0.01) or (global_step_cnt - last_update_step) > 2e6:
                        # Update teacher ckpt when:
                        # - there's no accuracy gain for 5 steps, and there's progress since the last ckpt update;
                        # - or there's no update for 2e6 steps.
                        last_eval_acc = curr_eval_acc
                        last_update_step = global_step_cnt
                        curr_ckpt_idx += 1
                        curr_ckpt_idx = min(curr_ckpt_idx, len(teacher_ckpt_step)-1)
                        ckpt_step = teacher_ckpt_step[curr_ckpt_idx]
                        curr_ckpt_step = ckpt_step
                        valid_ckpt_dir = teacher_ckpt_dir + '/checkpoint-' + str(ckpt_step)

                print("Loading ckpt from", valid_ckpt_dir)
                if 'mlp' in model_type:
                   state_dict = torch.load(valid_ckpt_dir)
                   teacher_layers.load_state_dict(state_dict['model_state_dict'])
                elif 'gpt' in model_type:
                  teacher_layers, _ = prepare_model(data_dimension, num_labels,
                                                    teacher_hidden_size, teacher_num_layers,
                                                    model_type=model_type, n_heads=teacher_n_heads,
                                                    vocab_size=vocab_size,
                                                    tie_word_embeddings=tie_word_embeddings,
                                                    ckpt_path=valid_ckpt_dir, # pass in the ckpt_dir
                                                    )
            """
            Training
            """
            batch_data, batch_y = batch

            cuda_batch_data, cuda_batch_y = torch.tensor(batch_data, device=DEVICE, dtype=dtype), torch.tensor(batch_y, device=DEVICE, dtype=int)

            if 'gpt' in model_type:
              predicted_y = get_logits_gpt(cuda_batch_data, all_layers)
            elif model_type == 'mlp':
              predicted_y = all_layers(cuda_batch_data)
            auto_loss = loss_fn(predicted_y, cuda_batch_y)

            with torch.no_grad():
              if teacher_type == 'ckpt':
                if 'gpt' in model_type:
                    teacher_pred = get_logits_gpt(cuda_batch_data, teacher_layers)
                else:
                    teacher_pred = teacher_layers(cuda_batch_data)
                if teacher_temperature_anneal > 0:
                    curr_temp = max(temperature * (1 - global_step_cnt / n_steps), 0.05)
                    if teacher_temperature_anneal_type == 'linear':
                        curr_temp = teacher_temperature_anneal * curr_temp 
                    elif teacher_temperature_anneal_type == 'log':
                      curr_temp = teacher_temperature_anneal * np.log(curr_temp)
                else:
                    curr_temp = temperature 
                if curr_temp == 0:
                  teacher_pred_argmax = torch.argmax(teacher_pred, axis=-1)
                  teacher_prob = torch.nn.functional.one_hot(teacher_pred_argmax, num_classes=vocab_size)
                else:
                  teacher_prob = torch.nn.functional.softmax(teacher_pred * 1/curr_temp, dim=-1)
              elif teacher_type == 'linear_anneal' or teacher_type == 'fixed_temp':
                  if teacher_type == 'linear_anneal':
                    curr_temp = max(max_anneal_temperature * (1 - global_step_cnt / n_steps), 0.1)
                  else:
                    curr_temp = max_anneal_temperature
                  teacher_pred = torch.nn.functional.one_hot(cuda_batch_y, num_classes=vocab_size)
                  teacher_prob = torch.nn.functional.softmax(teacher_pred * 1/curr_temp, dim=-1) 
              elif teacher_type == 'none':
                teacher_prob = None
                teacher_pred = None

            if teacher_prob is None:
                loss = auto_loss
            else:
              predicted_log_y = -torch.nn.functional.log_softmax(predicted_y, dim=-1)
              if kl_type == 'forward':
                  kl_loss = torch.mean(torch.sum(teacher_prob * predicted_log_y, axis=-1))
              elif kl_type == 'reverse':
                  # NOTE: currently poor performance.
                  teacher_pred_log_y = -torch.nn.functional.log_softmax(teacher_pred * 1/temperature, dim=-1)
                  kl_loss = torch.mean(torch.sum(predicted_y * (teacher_pred_log_y - predicted_log_y), axis=-1))

              # update kl_alpha if necessary
              if cfg['task']['kl_alpha_incre'] != 0 and global_step_cnt > cfg['task']['kl_alpha_incre_from']:
                  curr_incre = cfg['task']['kl_alpha_incre'] * ((global_step_cnt - cfg['task']['kl_alpha_incre_from']) // cfg['task']['kl_alpha_incre_intvl'])
                  curr_kl_alpha = min(cfg['task']['kl_alpha_max'], kl_alpha + curr_incre)
                  curr_kl_alpha = max(cfg['task']['kl_alpha_min'], curr_kl_alpha)
                  if curr_kl_alpha != prev_kl_alpha:
                      print(f"KL alpha updated from {prev_kl_alpha} to {curr_kl_alpha}")
                      prev_kl_alpha = curr_kl_alpha
              else:
                  curr_kl_alpha = kl_alpha
              
              loss = curr_kl_alpha * auto_loss + (1.-curr_kl_alpha) * kl_loss

            loss.backward()

            # set learning rate
            curr_learning_rate = 0
            if global_step_cnt < n_steps * warmup_ratio:
                # linear warmup
                curr_learning_rate = learning_rate * (global_step_cnt+1) / (n_steps * warmup_ratio)
            else:
                # cosine annealing
                curr_learning_rate = learning_rate * 0.5 * (1 + np.cos(np.pi * (global_step_cnt - n_steps * warmup_ratio) / (n_steps * (1-warmup_ratio))))
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_learning_rate 


            if global_step_cnt % logging_freq == 0: 
                if USE_WANDB:
                    wandb_results = {
                        'Train Loss': loss.item(),
                        'KL Loss': kl_loss.item() if teacher_prob is not None else 0.0,
                        'auto_loss': auto_loss.item(),
                        'KL alpha': curr_kl_alpha,
                       }
                    wandb.log(wandb_results, step=global_step_cnt)

            optimizer.step()
            optimizer.zero_grad()

            if global_step_cnt % min(logging_freq, len(train_loader)) == 0:
                eval_loss = 0.
                eval_acc = 0.
                for bt, batch in tqdm(enumerate(eval_loader)):
                    batch_data, batch_y = batch

                    cuda_batch_data, cuda_batch_y = torch.tensor(batch_data, device=DEVICE, dtype=dtype), torch.tensor(batch_y, device=DEVICE, dtype=int)
                    with torch.no_grad():
                        if model_type == 'mlp':
                          eval_logits = all_layers(cuda_batch_data)
                        elif 'gpt' in model_type:
                          eval_logits= get_logits_gpt(cuda_batch_data, all_layers)
                        predicted_y = torch.nn.functional.softmax(eval_logits, dim=-1)

                        acc = accuracy(predicted_y, cuda_batch_y)
                        eval_acc += acc.item()
                        auto_loss = loss_fn(predicted_y, cuda_batch_y)
                        eval_loss += auto_loss.item()
                    
                eval_acc = eval_acc / len(eval_loader)
                eval_loss = eval_loss / len(eval_loader)

                curr_eval_acc = eval_acc
                eval_accs += curr_eval_acc,

                # NOTE: logging to wandb at a lower frequency to reduce training time.
                if USE_WANDB:
                    wandb.log({
                              'Eval Acc': eval_acc,
                              'Eval loss': eval_loss,
                              'curr_ckpt_step': curr_ckpt_step,
                              },
                          step=global_step_cnt)

            if global_step_cnt % save_ckpt_freq == 0:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                print("Save at:", global_step_cnt)
                fout = os.path.join(output_path, 'checkpoint-'+str(global_step_cnt))
                if model_type == 'mlp':
                  if type(all_layers) == list:
                    model_state_dict = [layer.state_dict() for layer in all_layers]
                  else:
                    model_state_dict = all_layers.state_dict()
                  torch.save({
                        'epoch': epoch, 
                        'step': global_step_cnt,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict()
                        }, fout)
                elif 'gpt' in model_type:
                  all_layers.save_pretrained(fout)
            global_step_cnt += 1
            if global_step_cnt >= n_steps:
                break

if __name__ == "__main__":
    main()





