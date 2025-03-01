import os
import numpy as np
import torch 
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

import random
from transformers import AutoModelForCausalLM, AutoConfig, BertLMHeadModel

from transformers.pytorch_utils import Conv1D
from transformers.modeling_utils import load_state_dict
from parity.utils_modeling_gpt2 import GPT2LMHeadModelAvgHead

import pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Model related
"""

def load_torch_ckpt(model, fckpt):
  ckpt = torch.load(fckpt, map_location=DEVICE)
  model.load_state_dict(ckpt['model_state_dict'])
  print("Loaded ckpt", fckpt)
  return model


def prepare_model(dim_in, dim_out, hidden_size, num_layers,
                  model_type='mlp',
                  n_heads=4, vocab_size=50257,
                  ckpt_path='',
                  tie_word_embeddings=1,
                  attention_log_scale=0, attention_log_scale_degree=1,
                  skip_causal_mask=0,
                  linear_mlp=0, skip_mlp=0,):
  if model_type == 'mlp':
      layers = []
      params = []
      relu = torch.nn.ReLU()
      # input layer 
      linear = torch.nn.Linear(in_features=dim_in, out_features=hidden_size)
      layers += [linear, relu]
      params += [linear.weight, linear.bias]
      # hidden layers
      for _ in range(num_layers-1):
          # initialize the model
          linear = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
          layers += [linear, relu]
          params += [linear.weight, linear.bias]
      # output layer
      linear = torch.nn.Linear(in_features=hidden_size, out_features=dim_out)
      layers += [linear]
      params += [linear.weight, linear.bias]
      # return layers, params
      model = torch.nn.Sequential(*layers).to(DEVICE)
      params = model.parameters()
      return model, params
  
  elif 'gpt2' in model_type:
      config = AutoConfig.from_pretrained('gpt2')
      config.n_embd = hidden_size
      config.n_head = n_heads
      config.n_layer = num_layers
      config.vocab_size = vocab_size
      config.tie_word_embeddings = tie_word_embeddings
    
      if ckpt_path != '' and os.path.exists(ckpt_path):
          print("Loading pretrained model from", ckpt_path, '\n')
          
          if model_type == 'gpt2':
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, config=config).to(DEVICE)
          elif model_type == 'gpt2_avg_head':
            model = GPT2LMHeadModelAvgHead(config).to(DEVICE)
            fname_pt = os.path.join(ckpt_path, 'model.safetensors')
            state_dict = load_state_dict(fname_pt)
            model.load_state_dict(state_dict)
      else:
          print("Initializing model from scratch\n")
          if model_type == 'gpt2':
            model = AutoModelForCausalLM.from_config(config=config).to(DEVICE)
          elif model_type == 'gpt2_avg_head':
            model = GPT2LMHeadModelAvgHead(config).to(DEVICE)
      
      if linear_mlp:
          for layer in model.transformer.h:
              dim_in, din_out = layer.mlp.c_fc.weight.shape
              layer.mlp.c_fc = Conv1D(dim_in, dim_in).to(DEVICE)
              layer.mlp.c_proj = torch.nn.Identity()
      if skip_mlp:
          for layer in model.transformer.h:
              layer.mlp = torch.nn.Identity()
      return model, model.parameters()



def get_mlp(n_input, n_hidden, n_output, n_layers):
  layers = []
  for i in range(n_layers):
      if i == 0:
          layers.append(torch.nn.Linear(n_input, n_hidden))
      else:
          layers.append(torch.nn.Linear(n_hidden, n_hidden))
      layers.append(torch.nn.ReLU())
  layers.append(torch.nn.Linear(n_hidden, n_output))
  return torch.nn.Sequential(*layers)


relu = torch.nn.ReLU()

def get_logits(input_, layers, return_hidden=0):
    num_layers = len(layers) - 1
    out = input_
    hiddens = {}
    for li in range(num_layers):
        out = relu(layers[li](out))
        if return_hidden:
            hiddens[li] = out.detach().cpu().numpy()
    out = layers[num_layers](out)
    if return_hidden:
        return out, hiddens
    return out

def get_logits_gpt(input_, model):
    output = model(input_ids=input_)
    logits = output.logits
    return logits[:, -1, :]



"""
Data related
"""

def get_data_loaders(cfg, seed):
    set_seed(seed)

    data_type = cfg['data']['data_type']
    num_labels = cfg['data']['num_labels']
    num_workers = cfg['data']['num_workers']
    model_type = cfg['model']['type']

    if data_type == 'hierarchical':
        data_dimension = cfg['data']['data_dimension']
        feature_complexity = cfg['data']['feature_complexity']
        randomize_features = cfg['data']['randomize_features']
        n_examples = cfg['training']['n_examples']
        batch_size = cfg['training']['batch_size']

        all_features = get_features(num_labels, data_dimension, feature_complexity, random=randomize_features)
        # here, we set the seed to make deterministic runs
        all_data, all_y = boolean_data(n_examples, data_dimension, num_labels, all_features)
        if model_type == 'gpt2':
          # convert {-1, 1} to {1, 0}
          all_data = (1 - all_data) // 2
          all_data = all_data.astype(np.int64)


        eval_split = num_labels * min(2048, len(all_data)//4)
        train_split = len(all_data) - eval_split
        eval_batch_size = min(1000, eval_split)
        
        train_data, train_y = all_data[:train_split], all_y[:train_split]
        train_dataset = HierarchicalData(train_data, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
        eval_data, eval_y = all_data[train_split:], all_y[train_split:]
        eval_dataset = HierarchicalData(eval_data, eval_y)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size,
                                                  shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader



"""
Hierarchical tree data
"""

def get_features(num_labels, d, feature_complexity, random=False, feature_coordinates=None):
    if random:
        all_features = np.random.choice(d, size=(num_labels-1, feature_complexity))
    else:
        assert (num_labels-1) * feature_complexity <= d, "Number of available components should be more"
        if feature_coordinates is None:
            all_features = [range(i, i+feature_complexity) for i in range(0, (num_labels-1)*feature_complexity, feature_complexity)]
        else:
            all_features = [feature_coordinates]
    return all_features


def boolean_data(n, d, num_labels, all_features):
    def score(train_data, all_features, label):
        all_scores = np.zeros((len(train_data),))
        
        while(label > 1):
            prod_features = all_features [label//2-1]
            score_ = (1 - 2*(label%2)) * np.prod(train_data[:, prod_features], axis=-1)
            all_scores += score_
            label = label // 2
            
        return all_scores
    
    train_x = 2*np.random.choice(2, size=(n, d))-1
    
    train_y = np.zeros((n, num_labels))
    for i in range(num_labels):
        train_y[:, i] = score(train_x, all_features, i+num_labels)
    
    return train_x, np.argmax(train_y, axis=-1).astype(np.int32)

 
class HierarchicalData(Dataset):
  def __init__(self, data, labels):
      self.inputs = data
      self.labels = labels
      self.n_examples = len(data)
  
  def __getitem__(self, idx):
      return self.inputs[idx], self.labels[idx]

  def __len__(self):
      return self.n_examples




"""
Training related
"""
def loss_fn(pred, target): 
    loss_ = torch.nn.CrossEntropyLoss()
    return loss_ (pred, target)

def accuracy(pred, target):
    return (torch.argmax(pred, axis=-1) == target).type(torch.float32).mean()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

