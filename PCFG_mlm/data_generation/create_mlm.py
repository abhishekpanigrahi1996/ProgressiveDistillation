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
MASK_RATE = 0.15

TOKEN_MASK_RATE = MASK_RATE * 0.8
UNMODIFIED_MASK_RATE = MASK_RATE * 0.1
WRONG_MASK_RATE = MASK_RATE * 0.1
ALL_CHARS = numpy.arange(3)

def collate_pad_same_len(batch):
  max_len = max([len(b[0]) for b in batch])
  max_len = -1
  # pad to enforce a common max length
  init_seqs = [b[0] for b in batch]
  #seqs[0] += [PAD_TOKEN] * (ENFORCED_MAX_LEN - len(seqs[0]))

  # 15% masking
  # use mask map
  # 80-10-10 principle from BERT paper
  attention_mask = [numpy.random.choice(4, size=len(seq), p=[TOKEN_MASK_RATE, UNMODIFIED_MASK_RATE, WRONG_MASK_RATE, 1.-MASK_RATE]) for seq in init_seqs]
  random_seqs = [ numpy.random.choice(ALL_CHARS, len(seq)) for seq in init_seqs  ]  

  unmasked_ = [ numpy.equal(mask, 3).astype(int) for mask in attention_mask ]
  wrong_masked_ = [ numpy.equal(mask, 2).astype(int) for mask in attention_mask ]
  unmodified_masked_ = [ numpy.equal(mask, 1).astype(int) for mask in attention_mask ]
  true_masked_ = [ numpy.equal(mask, 0).astype(int) for mask in attention_mask ]
  
  
  labels = [ -100 * mask + (1-mask) * numpy.asarray(seq) for mask, seq in zip(unmasked_, init_seqs) ]

  input_ids = []
  for counter in range(len(init_seqs)):
    umask = unmasked_[counter]
    mask_t = true_masked_[counter]
    mask_u = unmodified_masked_[counter]
    mask_w = wrong_masked_[counter]

    seq = init_seqs[counter]
    ran_seq = random_seqs[counter]

    id_ = umask * numpy.asarray(seq) + MASK_TOKEN * mask_t + mask_u * numpy.asarray(seq) + mask_w * ran_seq
    input_ids += [id_]

  attention_mask = [ mask for mask in unmasked_ ]

  input_ids = [torch.tensor(seq) for seq in input_ids]
  labels = [torch.tensor(seq) for seq in labels]
  attention_mask = [torch.tensor(seq) for seq in attention_mask]


  input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=False, padding_value=PAD_TOKEN)
  labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=False, padding_value=-100)
  attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=False, padding_value=0)

  return input_ids, labels, attention_mask, [b[1] for b in batch], max_len


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
  parser.add_argument('--num_train_examples', type=int, default=10240)
  parser.add_argument('--num_test_examples', type=int, default=4096)

  args = parser.parse_args()

  MASK_RATE = args.mask_rate

  TOKEN_MASK_RATE = MASK_RATE * 0.8
  UNMODIFIED_MASK_RATE = MASK_RATE * 0.1
  WRONG_MASK_RATE = MASK_RATE * 0.1

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
  for split, size in [('eval', args.num_test_examples), ('train', args.num_train_examples)]:

    n_samples = int(size)
    dataset = PCFGDataset(pcfg=pcfg, num_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, num_workers=8,
                            collate_fn=collate_fn)
    
    fname_out = os.path.join(fdir_out, f'{split}_seed{seed}_mlm_maskrate{MASK_RATE}_80_10_10.pt')
    all_data = []
    all_labels = []
    all_masks = []
    for bi,batch in tqdm(enumerate(dataloader)):
      sentence_encoded = batch[0]
      labels_encoded = batch[1]
      mask = batch[2]

      if type(sentence_encoded) == list:
        sentence_encoded = torch.stack(sentence_encoded)
        labels_encoded = torch.stack(labels_encoded)
      else:
         ACTUAL_MAX_LEN = max(ACTUAL_MAX_LEN, batch[4])
      sentence_encoded = sentence_encoded.T
      labels_encoded = labels_encoded.T
      mask = mask.T 
      

      all_data.append(sentence_encoded)
      all_labels.append(labels_encoded)
      all_masks.append(mask)
      
      if bi % 1000 == 0 and 0: # NOTE: set to 1 to save intermediate files
        all_data_tensor = torch.cat(all_data, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        torch.save([all_data_tensor, all_labels_tensor], fname_out+'_tmp')
    
    torch.save([all_data, all_labels, all_masks], fname_out)
    if os.path.exists(fname_out+'_tmp'):
      # remove the tmp file
      os.remove(fname_out+'_tmp')


