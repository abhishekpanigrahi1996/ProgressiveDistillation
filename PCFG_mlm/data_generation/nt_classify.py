# a DataLoader for PCFG
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import nltk
import pickle
import pdb

from nltk.parse import EarleyChartParser

# from pcfg_defs import *

# TODO: please chance this to yours
LOCAL_DATA_DIR = '.'
LOCAL_DATA_DIR = 'data'

PAD_TOKEN = 0
MASK_TOKEN = 7


def generate_random_sentence(grammar, start_symbol, end_list=[], parent_terminal=[]):
    production = np.random.choice([p for p in grammar.productions(lhs=nltk.Nonterminal(start_symbol))],
                                  p=[p.prob() for p in grammar.productions(lhs=nltk.Nonterminal(start_symbol))])
    sentence = []
    end_symbol_list = []
    nt_symbol_list  = []

    total_nts = len(production.rhs())
    counter = 0
    for sym in production.rhs():
        counter += 1
        if counter == total_nts: sym_end_list = end_list + [1]    
        else: sym_end_list = end_list + [0]    
        sym_parent_terminal = parent_terminal + [start_symbol]

        if isinstance(sym, nltk.Nonterminal):
            gen_sent, gen_list, gen_nt = generate_random_sentence(grammar, \
                                            start_symbol=str(sym), \
                                            end_list=sym_end_list, parent_terminal=sym_parent_terminal)
            sentence.extend(gen_sent)
            end_symbol_list += [k for k in gen_list]
            nt_symbol_list  += [k for k in gen_nt]
        else:
            end_symbol_list += [sym_end_list]
            nt_symbol_list  += [sym_parent_terminal]
            sentence.append(sym)
    return ''.join(sentence), end_symbol_list, nt_symbol_list

ENFORCED_MAX_LEN = 768
ACTUAL_MAX_LEN = -1
MAX_DEPTH = 7

def collate_pad_same_len(batch):
  # pad to enforce a common max length
  init_seqs = [b[0] for b in batch]
  init_end_list = [b[2] for b in batch]
  init_nt_list = [b[3] for b in batch]

  # find max length among all seqs
  max_len = max([len(b) for b in init_seqs])
  seqs = [ seq + [PAD_TOKEN] * (max_len - len(seq)) for seq in init_seqs ]
  init_end_list = [ seq + [[-100] * MAX_DEPTH] * (max_len - len(seq)) for seq in init_end_list ]
  init_nt_list  = [ seq + [[-100] * MAX_DEPTH] * (max_len - len(seq)) for seq in init_nt_list ]


  return torch.tensor(seqs), torch.tensor(init_end_list), torch.tensor(init_nt_list), -1 #labels, [b[1] for b in batch], max_len


class PCFGDataset(Dataset):
  def __init__(self, pcfg=None, num_samples=1e4, max_length=512, max_depth=-1):
    self.grammar = pcfg['grammar']
    self.start_symbol = pcfg['start_symbol']
    self.map2tokens = pcfg['map']
    self.num_samples = num_samples
    self.max_length = max_length
    self.max_depth = max_depth
    self.replace_dict = {1:'a', 2:'b', 3:'c', 7:'d', 8:'e', 9:'f', 10:'g', 11:'h', 12:'i', 13:'j', 14:'k', 15:'l', 16:'m', 17:'n', 18:'o', 19:'p', 20:'q', 21:'r', 22:'s', 23:'t', 24:'u', 25:'v', 26:'w', 27:'x', 28:'y', 29:'z'}
    self.reverse_map = {}
    for key in self.replace_dict:
      self.reverse_map [self.replace_dict[key]] = key

  def __len__(self):
    return self.num_samples
  
  def __getitem__(self, idx):
    sentence, sentence_end_list, sentence_nts = generate_random_sentence(self.grammar, self.start_symbol)
    sentence_encoded = [self.map2tokens['BOS']] + [self.map2tokens[c] for c in sentence]
    return sentence_encoded, sentence, sentence_end_list, [[self.reverse_map[c] for c in lst] for lst in sentence_nts]
  

def tensor2string(tensor, map2char):
  return ''.join([map2char[i] for i in tensor])

def string2tensor(string, map2int):
  return torch.tensor([map2int[c] for c in string])


if __name__ == '__main__':
  from tqdm import tqdm
  import pdb

  # add an argument parser here
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--pcfg_def', type=str, default="PCFG_def/yuanzhi_cfg3b.pkl")
  parser.add_argument('--num_train_examples', type=int, default=10240)
  parser.add_argument('--num_test_examples', type=int, default=4096)


  args = parser.parse_args()

  if args.pcfg_def and os.path.exists(args.pcfg_def):
      """
      Load the PCFG definition from the given path
      """
      fpcfg_def = args.pcfg_def
      print("Loading from", fpcfg_def)
      with open(fpcfg_def, 'rb') as f:
         pcfg = pickle.load(f)
      fdir_out = os.path.join(LOCAL_DATA_DIR, os.path.basename(fpcfg_def))
      os.makedirs(fdir_out, exist_ok=True)
      seed = 0
      PAD_TOKEN = pcfg['map']['PAD']
      collate_fn = collate_pad_same_len
  else:
      raise NotImplementedError


  for split, size in [('train', args.num_train_examples), ('eval', args.num_test_examples)]:
    n_samples = int(size)
    dataset = PCFGDataset(pcfg=pcfg, num_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, num_workers=8,
                            collate_fn=collate_fn)
    
    fname_out = os.path.join(fdir_out, f'{split}_seed{seed}_boundary.pt')
    all_data = []
    all_end = []
    all_nt = []
    for bi,batch in tqdm(enumerate(dataloader)):
      sentence_encoded = batch[0]
      sentence_end_list = batch[1]
      sentence_nt_list = batch[2]

      all_data.append(sentence_encoded)
      all_end.append(sentence_end_list)
      all_nt.append(sentence_nt_list)

    torch.save([all_data, all_end, all_nt], fname_out)
    if os.path.exists(fname_out+'_tmp'):
      # remove the tmp file
      os.remove(fname_out+'_tmp')


