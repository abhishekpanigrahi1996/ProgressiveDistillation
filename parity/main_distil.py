import torch
from typing import Callable, Dict, Optional, Union, List

import sys
import os
import pickle
import pandas as pd
import numpy
import random
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    USE_WANDB = True
    os.environ["WANDB_MODE"]="offline"
except:
    USE_WANDB = False
    os.environ["WANDB_DISABLED"] = "true"



def set_seed(seed: int = 42) -> None:
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    

def score(train_data, all_features, label):
    all_scores = numpy.zeros((len(train_data),))
    
    while(label > 1):
        prod_features = all_features [label//2-1]
        score_ = (1 - 2*(label%2)) * numpy.prod(train_data[:, prod_features], axis=-1)
        all_scores += score_
        label = label // 2
    return all_scores


def get_features(num_labels, d, feature_complexity, random=False):
    if random:
        all_features = numpy.random.choice(d, size=(num_labels-1, feature_complexity))
    else:
        assert (num_labels-1) * feature_complexity <= d, "Number of available components should be more"
        all_features = [range(i, i+feature_complexity) for i in range(0, (num_labels-1)*feature_complexity, feature_complexity)]
    return all_features


def boolean_data(n, d, num_labels, all_features):
    train_x = 2*numpy.random.choice(2, size=(n, d))-1
    
    train_y = numpy.zeros((n, num_labels))
    for i in range(num_labels):
        train_y[:, i] = score(train_x, all_features, i+num_labels)
    
    return train_x, numpy.argmax(train_y, axis=-1).astype(numpy.int32)
    

def loss_fn(pred, target):  
    loss_ = torch.nn.CrossEntropyLoss()
    return loss_ (pred, target)

def accuracy(pred, target):
    return (torch.argmax(pred, axis=-1) == target).type(torch.float32).mean()
    
def main():
    
    n_examples = int(sys.argv[1])
    hidden_size = int(sys.argv[2])
    data_dimension = int(sys.argv[3])
    num_labels = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    logging_path = sys.argv[6]
    output_path = sys.argv[7]
    seed = int(sys.argv[8])
    num_layers = int(sys.argv[9])
    teacher_hidden_size = int(sys.argv[10])
    
    teacher_ckpt_dir = sys.argv[11]
    method = sys.argv[12]
    teacher_ckpt_step = sys.argv[13]
    feature_complexity = int(sys.argv[14])
    randomize_features = bool(int(sys.argv[15]))
    temp = float(sys.argv[16])
    save_freq = int(sys.argv[17])


    kl_alpha = 0.0

    logging_dir = logging_path + '_numexp' + str(n_examples) + '_hid' + str(hidden_size) + '_dim' + str(data_dimension) + '_num_labels' + str(num_labels) + '_lr' + str(learning_rate) + '_seed' + str(seed) + '_num_layers' + str(num_layers) + '_distil' + method + '_teachersize' + str(teacher_hidden_size) + '_ckptstep' + str(teacher_ckpt_step) + '_kl' + str(kl_alpha)  + '_feature_complexity' + str(feature_complexity) + '_randomizefeatures_' + str(randomize_features) + '_temp' + str(temp)    
    output_dir = output_path + '_numexp' + str(n_examples) + '_hid' + str(hidden_size) + '_dim' + str(data_dimension) + '_num_labels' + str(num_labels) + '_lr' + str(learning_rate) + '_seed' + str(seed) + '_num_layers' + str(num_layers) + '_distil' + method + '_teachersize' + str(teacher_hidden_size) + '_kl' + str(kl_alpha)  + '_feature_complexity' + str(feature_complexity) + '_randomizefeatures_' + str(randomize_features) + '_temp' + str(temp)
    
    #get the features first
    set_seed(42)
    all_features = get_features(num_labels, data_dimension, feature_complexity, random=randomize_features)

    
    set_seed(seed)
    batch_size = 1
    eval_batch_size = 1

    first_layer = torch.nn.Linear(in_features=data_dimension, out_features=hidden_size).to('cuda')
    dtype = first_layer.weight.dtype
    second_layer = torch.nn.Linear(in_features=hidden_size, out_features=num_labels).to('cuda')
    relu = torch.nn.ReLU()
    

    
    all_data, all_y = boolean_data(n_examples, data_dimension, num_labels, all_features) 
    eval_split = 125 * num_labels
    train_split = len(all_data) - eval_split
    
    eval_data, eval_y = all_data[train_split:], all_y[train_split:]
    train_data, train_y = all_data[:train_split], all_y[:train_split]
    
    all_parameters = []
    for n, p in first_layer.named_parameters():
        all_parameters += [p]
    
    for n, p in second_layer.named_parameters():
        all_parameters += [p]
    
    optimizer = torch.optim.SGD(all_parameters, lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_split, eta_min=0.1*learning_rate)


    n_batches = train_split // batch_size
    print (n_batches)
    
    if USE_WANDB:
        # get pid
        pid = os.getpid()

        wandb.init(project="Progressive_distillation", name=logging_dir)
        wandb.config.update({
            "n_examples": n_examples,
            "n_steps": n_examples,
            "data_dimension": data_dimension,
            "num_labels": num_labels,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "feature_complexity": feature_complexity, 
            "randomize_features": randomize_features,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "logging_path": logging_path,
            "output_path": output_path,
            "seed": seed,
            "pid": pid,
            "temp": temp,
            "teacher_ckpt_step": teacher_ckpt_step,
        })
    else:
        writer = SummaryWriter(logging_dir)
    
    
    ## Teacher's parameters ##
    teacher_first_layer = torch.nn.Linear(in_features=data_dimension, out_features=teacher_hidden_size).to('cuda')
    teacher_second_layer = torch.nn.Linear(in_features=teacher_hidden_size, out_features=num_labels).to('cuda')
    relu = torch.nn.ReLU()
    
    def retrieve_ckpt(ckpt_dir):
        ckpt = torch.load(ckpt_dir)
        state_dicts = ckpt['model_state_dict']
        teacher_first_layer.load_state_dict(state_dicts[0])
        teacher_second_layer.load_state_dict(state_dicts[1])
    
    def teacher_predictions(cuda_batch_data, normalize=True):
        with torch.no_grad():
            logit = teacher_second_layer(relu(teacher_first_layer(cuda_batch_data))) / temp
            if normalize: 
                predicted_y = torch.nn.functional.softmax(logit, dim=-1)
            else:
                predicted_y = logit
        return predicted_y

    
    ckpt_dir = teacher_ckpt_dir + '/checkpoint-' + str(teacher_ckpt_step)
    retrieve_ckpt(ckpt_dir)
    
    if method == 'progressive_distillation': 
        teacher_ckpt_step = [int(teacher_ckpt_step)]
        while True:
            a = teacher_ckpt_step[-1]
            a += teacher_ckpt_step[0]
            if a >= n_examples:
                break
            teacher_ckpt_step += [a]    
        
          
    for train_step in tqdm(range(n_batches)):
        batch_data, batch_y = train_data[train_step*batch_size: (train_step+1)*batch_size], train_y[train_step*batch_size: (train_step+1)*batch_size]

        cuda_batch_data, cuda_batch_y = torch.tensor(batch_data, device='cuda', dtype=dtype), torch.tensor(batch_y, device='cuda', dtype=int)
        predicted_y = second_layer(relu(first_layer(cuda_batch_data)))
        auto_loss = loss_fn(predicted_y, cuda_batch_y)
        
    
        teacher_pred = teacher_predictions(cuda_batch_data)
        predicted_log_y = -torch.nn.functional.log_softmax(predicted_y, dim=-1)
        kl_loss = torch.mean(torch.sum(teacher_pred * predicted_log_y, axis=-1))

        loss = kl_alpha * auto_loss + (1.-kl_alpha) * kl_loss


        if train_step % save_freq == 0: 
            if USE_WANDB:
                wandb.log({'Train Loss': loss.item()}, step=train_step)
                lr = lr_schedule.get_last_lr()[0]
                wandb.log({'LR': lr}, step=train_step)
            else:
                writer.add_scalar('Train Loss', loss.item(), train_step)
            

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_schedule.step()

        if train_step % save_freq == 0:
            n_eval_batches = eval_split // eval_batch_size
            eval_loss = 0.
            for ebt in range(n_eval_batches):
                batch_data, batch_y = eval_data[ebt*eval_batch_size: (ebt+1)*eval_batch_size], eval_y[ebt*eval_batch_size: (ebt+1)*eval_batch_size]

                cuda_batch_data, cuda_batch_y = torch.tensor(batch_data, device='cuda', dtype=dtype), torch.tensor(batch_y, device='cuda', dtype=dtype)
                with torch.no_grad():
                    predicted_y = torch.nn.functional.softmax(second_layer(relu(first_layer(cuda_batch_data))), dim=-1)
                    loss = accuracy(predicted_y, cuda_batch_y)
                    eval_loss += loss.item()
                
                if USE_WANDB:
                    wandb.log({'Eval Acc': eval_loss / n_eval_batches}, step=train_step)
                else:
                    writer.add_scalar('Eval Acc', eval_loss / n_eval_batches, train_step)
                    
                mean_eval_loss = eval_loss / n_eval_batches

            os.makedirs(output_dir, exist_ok=True)
                
            if train_step % save_freq == 0:
                torch.save({
                    'epoch': train_step,
                    'model_state_dict': [first_layer.state_dict(), second_layer.state_dict()],
                    'optimizer_state_dict': optimizer.state_dict()
                    }, output_dir + '/checkpoint-'+str(train_step))        
                
            if method == 'progressive_distillation':
                if train_step % teacher_ckpt_step[0] == 0:
                    frst_ind = 0
                    while (train_step > teacher_ckpt_step[frst_ind]):
                        frst_ind += 1
                    ckpt_step = teacher_ckpt_step[frst_ind]

                    ckpt_dir = teacher_ckpt_dir + '/checkpoint-' + str(ckpt_step)
                    retrieve_ckpt(ckpt_dir)
                                
if __name__ == "__main__":
    main()






