# a DataLoader for PCFG
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import nltk
import pickle
import pdb
import numpy



# TODO: please chance this to yours
LOCAL_DATA_DIR = '.'
LOCAL_DATA_DIR = 'data'

PAD_TOKEN = 0
MASK_TOKEN = 7


def generate_random_sentence(grammar, start_symbol):
    production = np.random.choice([p for p in grammar.productions(lhs=nltk.Nonterminal(start_symbol))],
                                  p=[p.prob() for p in grammar.productions(lhs=nltk.Nonterminal(start_symbol))])
    sentence = []
    for sym in production.rhs():
        if isinstance(sym, nltk.Nonterminal):
            sentence.extend(generate_random_sentence(grammar, start_symbol=str(sym)))
        else:
            sentence.append(sym)
    return ''.join(sentence)

ENFORCED_MAX_LEN = 768
ACTUAL_MAX_LEN = -1
POSITIONS_TO_MASK = range(30, 450, 30)
#[50, 100, 150, 200, 250, 300, 350, 400]
#MASK_RATE = 0.15

#TOKEN_MASK_RATE = MASK_RATE * 0.8
#UNMODIFIED_MASK_RATE = MASK_RATE * 0.1
#WRONG_MASK_RATE = MASK_RATE * 0.1
ALL_CHARS = numpy.arange(3)
ROBUST_OR_CLOSE = True 

def collate_pad_same_len(batch):
  max_len = max([len(b[0]) for b in batch])
  max_len = -1
  init_seqs = [b[0] for b in batch]

  attention_mask = []
  input_ids = []
  levels = []
  positions = []
  labels = []
  for counter in range(len(init_seqs)):
    id_ = init_seqs[counter]

    for p in POSITIONS_TO_MASK:
        if p < len(id_): 
            og_id_ = numpy.copy(init_seqs[counter])

            id_ = numpy.copy(init_seqs[counter])
            mask_ = numpy.ones_like(id_)
            mask_[p] = 0.
            id_[p] = MASK_TOKEN

            input_ids += [id_]
            attention_mask += [mask_]
            labels += [-100 * mask_ + (1-mask_) * numpy.asarray(og_id_)]
            levels += [-1]
            positions += [p]

            for level in range(1, 5):

                if level == 3:
                  left_start = p - 2
                  right_end  = min(p + 2, len(id_))
                elif level == 4:
                  left_start = p - 1
                  right_end  = p + 1
                else:    
                  left_start = p - level
                  right_end = min(p + level + 1, len(id_))

                perturbed_id_ = numpy.copy(id_)
                perturbed_mask_ = numpy.copy(mask_)
                
                if ROBUST_OR_CLOSE == True:
                    for p1 in range(left_start, right_end):
                        perturbed_id_[p1] = MASK_TOKEN
                        perturbed_mask_[p1] = 0.    
                else:
                    for p1 in range(len(perturbed_id_)):
                        if p1 < left_start or p1 >= right_end: 
                            perturbed_id_[p1] = MASK_TOKEN
                            perturbed_mask_[p1] = 0.    

                input_ids += [perturbed_id_]
                attention_mask += [perturbed_mask_]
                labels += [-100 * perturbed_mask_ + (1-perturbed_mask_) * numpy.asarray(og_id_)]
                levels += [level]
                positions += [p]

  input_ids = [torch.tensor(seq) for seq in input_ids]
  labels = [torch.tensor(seq) for seq in labels]
  attention_mask = [torch.tensor(seq) for seq in attention_mask]
  
  levels = [torch.tensor(level) for level in levels]
  positions = [torch.tensor(position) for position in positions]

  return input_ids, labels, attention_mask, levels, positions, [b[1] for b in batch], max_len


class PCFGDataset(Dataset):
  def __init__(self, pcfg=None, num_samples=1e4, max_length=512, max_depth=-1):
    self.grammar = pcfg['grammar']
    self.start_symbol = pcfg['start_symbol']
    self.map2tokens = pcfg['map']
    self.num_samples = num_samples
    self.max_length = max_length
    self.max_depth = max_depth

  def __len__(self):
    return self.num_samples
  
  def __getitem__(self, idx):
    sentence = generate_random_sentence(self.grammar, self.start_symbol)
    sentence_encoded = [self.map2tokens['BOS']] + [self.map2tokens[c] for c in sentence]
    return sentence_encoded, sentence
  

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
  parser.add_argument('--mask_rate', type=float, default=0.3)
  parser.add_argument('--robust_or_close', type=bool, default=True)
  parser.add_argument('--num_train_examples', type=int, default=1024)
  parser.add_argument('--num_test_examples', type=int, default=1024)


  args = parser.parse_args()
  ROBUST_OR_CLOSE = args.robust_or_close
  #MASK_RATE = args.mask_rate

  #TOKEN_MASK_RATE = MASK_RATE * 0.8
  #UNMODIFIED_MASK_RATE = MASK_RATE * 0.1
  #WRONG_MASK_RATE = MASK_RATE * 0.1

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
      seed = args.seed 
      PAD_TOKEN = pcfg['map']['PAD']
      MASK_TOKEN = pcfg['map']['MASK']
      collate_fn = collate_pad_same_len

      np.random.seed(seed)  
  else:
     raise NotImplementedError

  # 5120*1e3
  for split, size in [('eval', args.num_test_examples)]:
    n_samples = int(size)
    dataset = PCFGDataset(pcfg=pcfg, num_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1,
                            collate_fn=collate_fn)
    if ROBUST_OR_CLOSE == True:
        fname_out = os.path.join(fdir_out, f'{split}_seed{seed}_mlm_MI_ngramrobust.pt')
    else:
        fname_out = os.path.join(fdir_out, f'{split}_seed{seed}_mlm_MI_ngramclose.pt')
    all_data = []
    for bi,batch in tqdm(enumerate(dataloader)):
      #input_ids, labels, attention_mask
      sentence_encoded = batch[0]
      labels_encoded = batch[1]
      mask = batch[2]
      levels = batch[3]
      positions = batch[4]

      if type(sentence_encoded) == list:
        sentence_encoded = torch.stack(sentence_encoded)
        labels_encoded = torch.stack(labels_encoded)
        mask = torch.stack(mask)
        levels = torch.stack(levels)
        positions = torch.stack(positions)
       
      else:
         ACTUAL_MAX_LEN = max(ACTUAL_MAX_LEN, batch[6])
      
      all_data += [(sentence_encoded, labels_encoded, mask, levels, positions)]
      
    torch.save(all_data, fname_out)
    if os.path.exists(fname_out+'_tmp'):
      # remove the tmp file
      os.remove(fname_out+'_tmp')


