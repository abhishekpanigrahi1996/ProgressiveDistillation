import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json


# add a system path
import sys

# from utils import prepare_model, get_logits, get_logits_gpt, accuracy, loss_fn, collate_fn_float, collate_fn_int, get_features, boolean_data

CKPT_BASE_DIR = '../output/ckpts_sparsehierarchy_100seeds_unsharedvars/ckpts_numexp1000_hid50000_dim100_num_labels2_lr0.01_seed42_num_layers1_feature_complexity6_randomizefeatures_False_reg_0.0/'  #<-- please modify as necessary
JSON_DIR = '..'   #<-- please modify as necessary
teacher_ckpt_dir = CKPT_BASE_DIR
output_pth = JSON_DIR + "/correlations.json"

def get_features(num_labels, d, feature_complexity, random=False):
    if random:
        all_features = np.random.choice(d, size=(num_labels-1, feature_complexity))
    else:
        assert (num_labels-1) * feature_complexity <= d, "Number of available components should be more"
        all_features = [range(i, i+feature_complexity) for i in range(0, (num_labels-1)*feature_complexity, feature_complexity)]
    return all_features


def score(train_data, all_features, label):
    all_scores = np.zeros((len(train_data),))
    
    while(label > 1):
        prod_features = all_features [label//2-1]
        score_ = (1 - 2*(label%2)) * np.prod(train_data[:, prod_features], axis=-1)
        all_scores += score_
        label = label // 2
    return all_scores



def boolean_data(n, d, num_labels, all_features):
    train_x = 2*np.random.choice(2, size=(n, d))-1
    
    train_y = np.zeros((n, num_labels))
    for i in range(num_labels):
        train_y[:, i] = score(train_x, all_features, i+num_labels)
    
    return train_x, np.argmax(train_y, axis=-1).astype(np.int32)
    

data_dimension = 100
feature_complexity = 6

randomize_features = 0
feature_coordinates = [0, 1, 2, 3, 4, 5]
non_feature_coordinates = [i for i in range(data_dimension) if i not in feature_coordinates]

num_labels = 2
n_examples = 100000


all_features = get_features(num_labels, data_dimension, feature_complexity,
                            random=False)
# here, we set the seed to make deterministic runs
all_data, all_y = boolean_data(n_examples, data_dimension, num_labels, all_features)

data_cuda, y_cuda = torch.tensor(all_data).cuda(), torch.tensor(all_y).cuda()
data_cuda_mlp = data_cuda.float()
data_cuda = (1-data_cuda)/2
data_cuda = data_cuda.long()





def prepare_model(data_dimension, num_labels, hidden_size, num_layers):   
    all_layers, all_parameters = [], []
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

    return all_layers, all_parameters


model_type = 'mlp' 

teacher_hidden_size = 50000
teacher_num_layers = 1
teacher_layers, _ = prepare_model(data_dimension, num_labels, teacher_hidden_size, teacher_num_layers)

def retrieve_ckpt(teacher_layers, ckpt_dir, verbose=0):
    if verbose: print("Retrieving checkpoint from", os.path.basename(ckpt_dir))
    ckpt = torch.load(ckpt_dir)
    state_dicts = ckpt['model_state_dict']
    for i, layer in enumerate(teacher_layers):
        layer.load_state_dict(state_dicts[i])
    return teacher_layers
      
def get_outputs(x, model):
    model.eval()
    with torch.no_grad():
        output = model(x)
    return output.logits[:, -1, :]

relu = torch.nn.ReLU()

def get_outputs_mlp(x, layers):
    with torch.no_grad():
        num_layers = len(layers) - 1
        out = x
        hiddens = {}
        for li in range(num_layers):
            out = relu(layers[li](out))
        out = layers[num_layers](out)
        return out


def get_correlation(data, model,
                    coords_in_support, coords_rand,
                    data_cuda=None, y=None,
                    batch_size=10000):
    if data_cuda is None:
        data_cuda = torch.tensor(data).cuda()
    outputs = []
    n_batches = len(data) // batch_size
    for i in range(n_batches):
        batch = data_cuda[i*batch_size:(i+1)*batch_size]
        if model_type == 'gpt2':
            outputs.append(get_outputs(batch, model))
        elif model_type == 'mlp':
            outputs.append(get_outputs_mlp(batch.float(), model))
    # if model_type == 'gpt2':
    #     outputs = get_outputs(data_cuda, model)
    #     probs = torch.softmax(outputs, -1)[:, 1]
    # elif model_type == 'mlp':
    #     outputs = get_outputs_mlp(data_cuda, model)
    #     probs = torch.softmax(outputs, -1)[:, 0]
    outputs = torch.cat(outputs, 0)
    probs = torch.softmax(outputs, -1)[:, 1]
    probs = probs.cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = (preds == all_y).mean()


    # in-support features
    corrs_in_support = {}
    for poly_features in coords_in_support:
        labels = np.prod(data[:, poly_features], axis=1)
        corr = (labels * probs).mean().__abs__()
        key = '_'.join([str(i) for i in sorted(poly_features)])
        corrs_in_support[key] = corr
    
    # other features
    cnt = 0
    corrs_off_support = {}
    for poly_features in coords_rand:
        labels = np.prod(data[:, poly_features], axis=1)
        corr = (labels * probs).mean().__abs__()
        key = '_'.join([str(i) for i in sorted(poly_features)])
        corrs_off_support[key] = corr
        cnt += 1
    return corrs_in_support, corrs_off_support, acc, probs
        


verbose = 0
ckpt_steps = range(0, 7900000, 100000)

ckpt_steps = sorted(ckpt_steps)
# ckpt_steps = ckpt_steps[::2]

# Get feature coords
from itertools import combinations
corr_in_support_across_k, corr_off_support_across_k = {}, {}
ks = range(1,6)
# ks = range(1,2)

accs_across_steps = {}
probs_across_steps = {}
for k in ks:
    coords_in_support = list(combinations(feature_coordinates, k))
    coords_in_support_set = [set(each) for each in coords_in_support]

    n_rand = 100
    cnt = 0
    coords_rand = []
    while cnt < n_rand:
        rand_coords = np.random.choice(range(data_dimension), k, replace=False)
        if set(rand_coords) in coords_in_support_set:
            continue
        coords_rand.append(rand_coords)
        cnt += 1

    corrs_in_steps, corrs_off_steps = {}, {}
    for ckpt_step in tqdm(ckpt_steps):
        ckpt_path = os.path.join(teacher_ckpt_dir, f'checkpoint-{ckpt_step}')
        if not os.path.exists(ckpt_path):
            if verbose: print("Skipping", ckpt_step)
            continue
        if verbose: print(ckpt_step)
        model, _ = prepare_model(data_dimension, num_labels, teacher_hidden_size, teacher_num_layers)
        model = retrieve_ckpt(model, ckpt_path)
        corrs_in_support, corrs_off_support, acc, probs = get_correlation(all_data, model,
                                                            coords_in_support, coords_rand,
                                                            data_cuda=data_cuda if model_type == 'gpt2' else data_cuda_mlp)
        accs_across_steps[ckpt_step] = acc
        probs_across_steps[ckpt_step] = probs
        corrs_in_steps[ckpt_step] = corrs_in_support
        corrs_off_steps[ckpt_step] = corrs_off_support
    corr_in_support_across_k[k] = corrs_in_steps
    corr_off_support_across_k[k] = corrs_off_steps

correlations = {}
correlations['corr_in_support_across_k'] = corr_in_support_across_k
correlations['corr_off_support_across_k'] = corr_off_support_across_k

json.dump(correlations, open(output_pth, 'w'), indent=4)
