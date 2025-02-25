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


eval_file = sys.argv[1]
model_name_or_path = sys.argv[2]
logging_dir = sys.argv[3]
checkpoint_num = int(sys.argv[4])
seed = int(sys.argv[5])
output_path = sys.argv[6]

model = BertForMaskedLM.from_pretrained(model_name_or_path).cuda()

set_seed (42)
pid = os.getpid()


all_data = torch.load(eval_file)

def model_out(batch):
    TV_out = {}
    for position in range(30, 450, 30):
        TV_out[position] = {}
        for level in range(1, 5):
            TV_out[position][level] = []
        TV_out[position][-1] = []
    
    id_, _, mask_, level_, pos_  = batch

    
    input_ids = id_.to(model.device)
    mask = mask_.to(model.device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=False, return_dict=True)
    
    logits = output['logits']
   
    for position, level, logit in zip(pos_, level_, logits):
        if position.item() < logit.shape[0]:
            prob = torch.nn.functional.softmax(logit, dim=-1)
            TV_out[position.item()][level.item()] += [prob.detach().cpu().numpy()[position.item()]]
    
    TV_arr = {}
    for position in range(30, 450, 30):
        TV_arr[position] = {}
        for level in range(1, 5):
            TV_arr[position][level] = []

    
    for position in range(30, 450, 30):
        for level in range(1, 5):
            seq_level = []
            for logit in TV_out[position][level]:
                if len(TV_out[position][level]) != 0:
                    seq_level += [numpy.linalg.norm(logit - TV_out[position][-1][0], ord=1)]
            if len(seq_level) > 0:
                TV_arr[position][level] += [numpy.mean(seq_level)]
    return TV_arr


all_TV_out = {}
for level in range(1, 5):
    all_TV_out[level] = []

for batch in tqdm(all_data):
    output = model_out(batch)
    for position in range(30, 450, 30):
        for level in range(1, 5):
            all_TV_out[level] += output[position][level]
    

results = {}
# here, level indicates the n-gram level
for level in range(1, 5):
    results['Mean Level ' + str(level)] = numpy.mean(all_TV_out[level])
    results['Std Level ' + str(level)]  = numpy.std(all_TV_out[level])
    results['25 Percentile n-gram' + str(1+2*level)]  = numpy.percentile(all_TV_out[level], q=25)
    results['50 Percentile n-gram' + str(1+2*level)]  = numpy.percentile(all_TV_out[level], q=50)
    results['75 Percentile n-gram' + str(1+2*level)]  = numpy.percentile(all_TV_out[level], q=75)
    results['90 Percentile n-gram' + str(1+2*level)]  = numpy.percentile(all_TV_out[level], q=90)


out_file = output_path
json.dump(results, open(out_file, 'w'), indent=4)