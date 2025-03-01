# a DataLoader for PCFG
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
import pickle


# TODO: please chance this to yours
LOCAL_DATA_DIR = '.'
#'/usr1/bingbin/datasets'
LOCAL_DATA_DIR = 'data'

PAD_TOKEN = 0

def generate_random_sentence(grammar, start_symbol, desired_level, level=0, end_label=0):
    production = np.random.choice([p for p in grammar.productions(lhs=nltk.Nonterminal(start_symbol))],
                                  p=[p.prob() for p in grammar.productions(lhs=nltk.Nonterminal(start_symbol))])
    sentence = []
    end_label_list = []

    total_nts = len(production.rhs())
    counter = 0
    for sym in production.rhs():
        counter += 1
        if level == desired_level:
            if counter == total_nts: forwarded_end_label = 1
            else: forwarded_end_label = 0
        else:
            if counter == total_nts: forwarded_end_label = end_label
            else: forwarded_end_label = 0

        if isinstance(sym, nltk.Nonterminal):
            gen_sent, gen_list = generate_random_sentence(grammar, \
                                            start_symbol=str(sym), \
                                            desired_level=desired_level, \
                                            level=level+1, \
                                            end_label=forwarded_end_label)
            sentence.extend(gen_sent)
            end_label_list += [k for k in gen_list]
        else:
            end_label_list  += [forwarded_end_label]
            sentence.append(sym)
    return ''.join(sentence), end_label_list


ENFORCED_MAX_LEN = 768
ACTUAL_MAX_LEN = -1

def collate_pad_same_len(batch):
  max_len = max([len(b[0]) for b in batch])
  max_len = -1
  # pad to enforce a common max length
  #init_seqs = [b[0] for b in batch]
  init_init_seqs = [b[0] for b in batch]
  init_end_label_list = [b[2] for b in batch]
  
  init_seqs = [torch.tensor(seq) for seq in init_init_seqs]
  labels = [torch.tensor(seq) for seq in init_init_seqs]
  init_end_label_list = [torch.tensor(seq) for seq in init_end_label_list]
  for counter in range(len(labels)):
    indices =  (init_end_label_list[counter] == 0)
    labels [counter] [ indices ] = -100

  seqs = torch.nn.utils.rnn.pad_sequence(init_seqs, batch_first=False, padding_value=PAD_TOKEN)
  labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=False, padding_value=-100)
  
  return seqs, labels, max_len


class PCFGDataset(Dataset):
  def __init__(self, pcfg=None, num_samples=1e4, max_length=512, max_depth=-1, desired_level=3):
    self.grammar = pcfg['grammar']
    self.start_symbol = pcfg['start_symbol']
    self.map2tokens = pcfg['map']
    self.num_samples = num_samples
    self.max_length = max_length
    self.max_depth = max_depth
    self.desired_level = desired_level

  def __len__(self):
    return self.num_samples
  
  def __getitem__(self, idx):
    sentence, end_label_list = generate_random_sentence(self.grammar, self.start_symbol, self.desired_level)
    sentence_encoded = [self.map2tokens['BOS']] + [self.map2tokens[c] for c in sentence]
    end_label_list = [0] + [l for l in end_label_list]
    return sentence_encoded, sentence, end_label_list
  

def tensor2string(tensor, map2char):
  return ''.join([map2char[i] for i in tensor])

def string2tensor(string, map2int):
  return torch.tensor([map2int[c] for c in string])


if __name__ == '__main__':
  from tqdm import tqdm

  # add an argument parser here
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--pcfg_def', type=str, default="")
  parser.add_argument('--desired_level', type=float, default=3)
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
      print (fdir_out)
      seed = args.seed 
      desired_level = args.desired_level
      PAD_TOKEN = pcfg['map']['PAD']
      collate_fn = collate_pad_same_len

      np.random.seed(seed)  
  else:
      raise NotImplementedError


  # 5120*1e3
  for split, size in [('eval', args.num_test_examples), ('train', args.num_train_examples)]: 
    n_samples = int(size)
    dataset = PCFGDataset(pcfg=pcfg, num_samples=n_samples, desired_level=desired_level)
    dataloader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, num_workers=8,
                            collate_fn=collate_fn)
    
    fname_out = os.path.join(fdir_out, f'{split}_seed{seed}_boundarylabels_' + str(desired_level) + '.pt')
    all_data = []
    all_labels = []
    #print(len(dataloader))
    for bi,batch in tqdm(enumerate(dataloader)):
      sentence_encoded = batch[0]
      labels_encoded = batch[1]
      # sentence_str = batch[1]
      if type(sentence_encoded) == list:
        sentence_encoded = torch.stack(sentence_encoded)
        labels_encoded = torch.stack(labels_encoded)
      else:
         ACTUAL_MAX_LEN = max(ACTUAL_MAX_LEN, batch[2])
      sentence_encoded = sentence_encoded.T
      labels_encoded = labels_encoded.T
      all_data.append(sentence_encoded)
      all_labels.append(labels_encoded)
      if bi % 1000 == 0 and 0: # NOTE: set to 1 to save intermediate files
        all_data_tensor = torch.cat(all_data, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        torch.save([all_data_tensor, all_labels_tensor], fname_out+'_tmp')
    
    torch.save([all_data, all_labels], fname_out)
    if os.path.exists(fname_out+'_tmp'):
      os.remove(fname_out+'_tmp')


