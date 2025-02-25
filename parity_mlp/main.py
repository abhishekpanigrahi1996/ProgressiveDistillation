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
        if (num_labels-1) * feature_complexity <= d:
            all_features = [range(i, i+feature_complexity) for i in range(0, (num_labels-1)*feature_complexity, feature_complexity)]
        else:
            tot = 1 + 1 * (((num_labels-1) * feature_complexity) // d)
            all_ = []
            for k in range(tot):
                all_ += list(numpy.random.permutation(d))
            all_features = [all_[i: i+feature_complexity] for i in range(0, (num_labels-1)*feature_complexity, feature_complexity)]    
    
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
    feature_complexity = int(sys.argv[10])
    randomize_features = bool(int(sys.argv[11]))
    save_freq = int(sys.argv[12])
    regularization = 0.0
    batch_size = 1

    logging_dir = logging_path + '_numexp' + str(n_examples) + '_hid' + str(hidden_size) + '_dim' + str(data_dimension) + '_num_labels' + str(num_labels) + '_lr' + str(learning_rate) + '_seed' + str(seed) + '_num_layers' + str(num_layers) + '_feature_complexity' + str(feature_complexity) + '_randomizefeatures_' + str(randomize_features) + '_reg_' + str(regularization)
    output_dir = output_path + '_numexp' + str(n_examples) + '_hid' + str(hidden_size) + '_dim' + str(data_dimension) + '_num_labels' + str(num_labels) + '_lr' + str(learning_rate) + '_seed' + str(seed) + '_num_layers' + str(num_layers) + '_feature_complexity' + str(feature_complexity) + '_randomizefeatures_' + str(randomize_features) + '_reg_' + str(regularization)
    
    #get the features first
    set_seed(42)
    all_features = get_features(num_labels, data_dimension, feature_complexity, random=randomize_features)
    
    # here, we set the seed to make deterministic runs
    set_seed(seed)
    
    
    
    
    # first get the data
    all_data, all_y = boolean_data(n_examples, data_dimension, num_labels, all_features)
    
    all_layers = []
    all_parameters = []
    relu = torch.nn.ReLU()
    linear = torch.nn.Linear(in_features=data_dimension, out_features=hidden_size).to('cuda')
    dtype = linear.weight.dtype
    all_layers += [linear]
    all_parameters += [linear.weight, linear.bias]
        
    for _ in range(num_layers-1):
        # initialize the model
        linear = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size).to('cuda')
        all_layers += [linear]
        all_parameters += [linear.weight, linear.bias]
    
    linear = torch.nn.Linear(in_features=hidden_size, out_features=num_labels).to('cuda')
    
    all_layers += [linear]
    all_parameters += [linear.weight, linear.bias]

    eval_split = num_labels * 125
    eval_batch_size = min(1000, eval_split)
    train_split = len(all_data) - eval_split
    
    eval_data, eval_y = all_data[train_split:], all_y[train_split:]
    train_data, train_y = all_data[:train_split], all_y[:train_split]

    optimizer = torch.optim.SGD(all_parameters, lr=learning_rate, weight_decay=regularization)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_split, eta_min=0.1*learning_rate)

    n_batches = train_split // batch_size
    
    os.makedirs(logging_dir, exist_ok=True)

    if USE_WANDB:
        # get pid
        pid = os.getpid()

        wandb.init(project="Distillation", name=logging_dir)
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
            "regularization": regularization,
            "residual": True
        })
    else:
        writer = SummaryWriter(logging_dir)
    
    def predictions(input_):
        out = input_
        out = relu(all_layers[0](out))
        for layer in range(1, num_layers):
            out = relu(all_layers[layer-1](out)) 
        out = all_layers[num_layers](out)
        return out
    
    for bt in tqdm(range(n_batches)):
        batch_data, batch_y = train_data[bt*batch_size: (bt+1)*batch_size], train_y[bt*batch_size: (bt+1)*batch_size]

        cuda_batch_data, cuda_batch_y = torch.tensor(batch_data, device='cuda', dtype=dtype), torch.tensor(batch_y, device='cuda', dtype=int)
        predicted_y = predictions(cuda_batch_data)

        loss = loss_fn(predicted_y, cuda_batch_y)
        if bt % (save_freq // batch_size) == 0: 
            if USE_WANDB:
                wandb.log({'Train Loss': loss.item()}, step=bt)
                lr = lr_schedule.get_last_lr()[0]
                wandb.log({'LR': lr}, step=bt)
            else:
                writer.add_scalar('Train Loss', loss.item(), bt)
            

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_schedule.step()

        if bt % (save_freq // batch_size) == 0:
            n_eval_batches = eval_split // eval_batch_size
            eval_loss = 0.
            for ebt in range(n_eval_batches):
                batch_data, batch_y = eval_data[ebt*eval_batch_size: (ebt+1)*eval_batch_size], eval_y[ebt*eval_batch_size: (ebt+1)*eval_batch_size]

                cuda_batch_data, cuda_batch_y = torch.tensor(batch_data, device='cuda', dtype=dtype), torch.tensor(batch_y, device='cuda', dtype=dtype)
                with torch.no_grad():
                    predicted_y = torch.nn.functional.softmax(predictions(cuda_batch_data), dim=-1)
                    loss = accuracy(predicted_y, cuda_batch_y)
                    eval_loss += loss.item()
            if USE_WANDB:
                wandb.log({'Eval Acc': eval_loss / n_eval_batches}, step=bt)
            else:
                writer.add_scalar('Eval Acc', eval_loss / n_eval_batches, bt)
            # save checkpoint at every 1e5 steps
            os.makedirs(output_dir, exist_ok=True)

                
            if bt % (save_freq // batch_size) == 0:
                torch.save({
                    'epoch': bt,
                    'model_state_dict': [layer.state_dict() for layer in all_layers],
                    'optimizer_state_dict': optimizer.state_dict()
                    }, output_dir + '/checkpoint-'+str(bt))
            
if __name__ == "__main__":
    main()






