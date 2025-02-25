import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, BertForMaskedLM
import sys 
import numpy
import pandas as pd
from torch.utils.data import DataLoader
import transformers
from transformers import DefaultDataCollator
import os 
import wandb
from transformers import set_seed
from tqdm import tqdm

train_file = sys.argv[1]
eval_file = sys.argv[2]
model_name_or_path = sys.argv[3]
num_predictions = int(sys.argv[4])
num_positions = int(sys.argv[5])
position_embd = int(sys.argv[6])
learning_rate = float(sys.argv[7])
num_NTs = int(sys.argv[9])
logging_dir = sys.argv[8]
output_pth = sys.argv[10]

num_heads=16
n_epochs = 4
model = BertForMaskedLM.from_pretrained(model_name_or_path).cuda()

set_seed (42)
pid = os.getpid()

os.environ["WANDB_MODE"]="offline"
wandb.init(project="NT prediction Bert 4L32H rerun", name=logging_dir)



all_heads = []
all_parameters = []

for i in range(num_heads):
    linear_layer = torch.nn.Linear(model.config.hidden_size, num_predictions * num_NTs).cuda()
    position_arch = torch.nn.Linear(num_positions, position_embd, bias=False).cuda()
    all_heads += [(linear_layer, position_arch)]
    all_parameters += [p for p in linear_layer.parameters()] + [p for p in position_arch.parameters()]

all_data, all_end, all_nt = torch.load(train_file)
train_data = []
for arr_data, arr_boundary in zip(all_data, all_nt):
    for bt, label in zip(arr_data, arr_boundary):
        train_data += [{'input_ids': numpy.asarray(bt), 'labels': numpy.reshape(numpy.asarray(label), newshape=(-1,)) }]        
train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_data))  


all_data, all_end, all_nt = torch.load(eval_file)
eval_data = []
for arr_data, arr_boundary in zip(all_data, all_nt):
    for bt, label in zip(arr_data, arr_boundary):
        eval_data += [{'input_ids': numpy.asarray(bt), 'labels': numpy.reshape(numpy.asarray(label), newshape=(-1,)) }]
eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_data))  


train_dataloader = DataLoader(train_dataset, batch_size=64,
                        shuffle=False, num_workers=8,
                        collate_fn=DefaultDataCollator())

eval_dataloader = DataLoader(eval_dataset, batch_size=64,
                        shuffle=False, num_workers=8,
                        collate_fn=DefaultDataCollator())

optimizer = torch.optim.Adam(all_parameters, lr=learning_rate, weight_decay=0.0)
T_max=len(train_dataloader)*n_epochs
num_warmup_steps=(T_max*6) // 100
lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=T_max)


def model_out(batch):
    input_ids = batch['input_ids'].to(model.device)

    with torch.no_grad():
        PAD_TOKEN = 5
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == PAD_TOKEN] = 0
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    
    hidden_states = output['hidden_states'][-1]
    linear_layer, position_arch = all_heads[0]
    linear_transform = linear_layer(hidden_states)
    seq_length = hidden_states.shape[1]
    position = torch.nn.Softmax(dim=-1)(position_arch.weight[..., :seq_length].T @  position_arch.weight[..., :seq_length])
    output = torch.einsum('ijd,jl->ild', linear_transform, position)

    for i in range(1, len(all_heads)):
        linear_layer, position_arch = all_heads[i]
        linear_transform = linear_layer(hidden_states)
        position = torch.nn.Softmax(dim=-1)(position_arch.weight[..., :seq_length].T @  position_arch.weight[..., :seq_length])
        output += torch.einsum('ijd,jl->ild', linear_transform, position)
    return output

def accuracy_eval(output, target):
    target =  target[..., :-1, :]                              # <-- needs to be left shifted because BOS's labels are non-existent
    padding_mask = target.eq(-100)
    target.clamp_min_(0)

    # first reshape output
    batch_size = output.shape[0]
    seq_length = output.shape[1]

    output = torch.reshape(output, shape=(batch_size, seq_length, num_predictions, num_NTs))
    output = output[..., 1:, :, :]                                # <-- ignore BOS

    accs = (output.argmax(axis=-1) == target).type(torch.float) * (1. - (padding_mask).type(torch.float))

    total_acc = torch.sum(torch.sum(accs, axis=0), axis=0)
    num_active_elements = torch.sum(torch.sum(1. - padding_mask.long(), axis=0), axis=0)

    avg_acc = total_acc / (1. * num_active_elements)
    return avg_acc

def loss(output, target, log_wandb=False, global_steps=0):
    target =  target[..., :-1, :]                              # <-- needs to be left shifted because BOS's labels are non-existent
    target = target.unsqueeze(-1)                              # <-- label function for softmax
    padding_mask = target.eq(-100)
    target.clamp_min_(0)

    # first reshape output
    batch_size = output.shape[0]
    seq_length = output.shape[1]
    
    output = torch.reshape(output, shape=(batch_size, seq_length, num_predictions, num_NTs))
    output = output[..., 1:, :, :]                               # <-- ignore BOS

    log_output = -torch.nn.functional.log_softmax(output, dim=-1)

    auto_loss = log_output.gather(dim=-1, index=target)
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()

    total_loss = torch.sum(auto_loss)

    return total_loss/num_active_elements


def evaluate(eval_dataloader, global_steps):
    
    total_acc = torch.zeros((num_predictions,)).to(model.device)
    tot_bt  = 0.
    for batch in eval_dataloader:
        with torch.no_grad():
            batch_size = batch['labels'].shape[0]
            target = torch.reshape(batch['labels'], (batch_size, -1, num_predictions)).to(model.device)
            output = model_out(batch)
            acc_ = accuracy_eval(output, target)
            total_acc += acc_
            tot_bt += 1
    
    for i in range(num_predictions):
        wandb.log({'Eval Accuracy Tree Level:' + str(i): 100. * total_acc[i].item()/tot_bt, 'Global steps': global_steps})
    
    all_results = {}
    for i in range(num_predictions):
        all_results['Eval Accuracy Tree Level:' + str(i)] = 100. * total_acc[i].item()/tot_bt
    return all_results


eval_freq = 100
global_steps = 0
for epoch in tqdm(range(n_epochs)):
    for batch in tqdm(train_dataloader):
        global_steps += 1
        output = model_out(batch)
        batch_size = batch['labels'].shape[0]
        target = torch.reshape(batch['labels'], (batch_size, -1, num_predictions)).to(model.device)


        model_loss = loss(output, target, log_wandb=False, global_steps=global_steps)
        model_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
       
all_results = evaluate(eval_dataloader, global_steps=global_steps)
out_file = output_pth
import pickle
pickle.dump(all_results, open(out_file, 'w'), indent=4)







