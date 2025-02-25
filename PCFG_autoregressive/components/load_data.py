from typing import Optional, List, Union
from dataclasses import dataclass, field
import torch
from functools import partial
from datasets import load_dataset, load_from_disk, DatasetDict
import logging
import numpy as np


def encode_format(example, tokenizer, max_seq_length):
    
    
    if 'input_ids' not in example:
        if 'text' in example:
            example_text = example['text']
            example_text = example_text + tokenizer.eos_token
            if max_seq_length != -1:
                tokenized_example = tokenizer(example_text, return_tensors='pt', truncation=True, max_length=max_seq_length)
            else:
                tokenized_example = tokenizer(example_text, return_tensors='pt', truncation=False)
            input_ids = tokenized_example.input_ids
        elif 'question' in example:
            example_text = example['code'] # Simple code training
            example_text = example_text + tokenizer.eos_token
            if max_seq_length != -1:
                tokenized_example = tokenizer(example_text, return_tensors='pt', truncation=True, max_length=max_seq_length)
            else:
                tokenized_example = tokenizer(example_text, return_tensors='pt', truncation=False)
            input_ids = tokenized_example.input_ids
    else:
        input_ids = torch.tensor(example['input_ids'])
        
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_format_wotokenizer(example):
    
    #example_text = example['text']
    #example_text = example_text + tokenizer.eos_token
    tokenized_example = torch.tensor(example['input_ids'], dtype=torch.int32)
    #tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example #.input_ids
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def get_encode_function(raw_datasets, tokenizer, max_seq_length):
    # Preprocessing the datasets.
    encode_function = partial(
        encode_format,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    return encode_function



def get_encode_function_wotokenizer(raw_datasets):
    # Preprocessing the datasets.
    encode_function = partial(
        encode_format_wotokenizer,
    )
    return encode_function


def chunk_examples(examples, max_seq_length):
    chunk_ids = []
    chunk_mask = []
    chunk_labels = []
    for id_, mask, label_ in zip(examples["input_ids"], examples["attention_mask"], examples["labels"]):
        chunk_ids += [id_[i:i + max_seq_length] for i in range(0, len(id_), max_seq_length)]
        chunk_mask += [mask[i:i + max_seq_length] for i in range(0, len(id_), max_seq_length)]
        chunk_labels += [label_[i:i + max_seq_length] for i in range(0, len(id_), max_seq_length)]
    return {"input_ids": chunk_ids, "attention_mask": chunk_mask, "labels": chunk_labels}


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers, cache_dir, overwrite_cache=False, no_tokenizer=False, chunk_tokenizer=False):
    #if "train" in raw_datasets and "input_ids" in raw_datasets["train"].features: return raw_datasets
    #if "input_ids" in raw_datasets.features: return raw_datasets
    
    print ("Here")
    if no_tokenizer:
        encode_function = get_encode_function_wotokenizer(raw_datasets)
    else:
        if not chunk_tokenizer:
            encode_function = get_encode_function(raw_datasets, tokenizer, max_seq_length)
        else:
            encode_function = get_encode_function(raw_datasets, tokenizer, -1)
            
    # To speed up this part, we use multiprocessing.   
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        cache_file_name=cache_dir,
        desc="Tokenizing and reformatting instruction data",
    )
    
    if chunk_tokenizer:
        lm_datasets = lm_datasets.map(partial(chunk_examples, max_seq_length=max_seq_length), 
                                      batched=True, 
                                      num_proc=processing_num_workers, 
                                      remove_columns=lm_datasets.column_names)
    lm_datasets.set_format(type="pt")
    return lm_datasets