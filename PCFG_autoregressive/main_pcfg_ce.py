import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from components.all_arguments import ModelArguments, TrainingArguments, DataArguments
from components.load_data import encode_data
from datasets import Dataset, load_from_disk
from transformers import Trainer

import os
import pandas as pd
import pickle
import numpy
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import glob
from transformers import set_seed


def main():
    parser = HfArgumentParser((ModelArguments,TrainingArguments,DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses() 

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir,)
    tokenizer.pad_token = tokenizer.eos_token

    ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            cache_dir=model_args.cache_dir,)
    ref_config.hidden_size=model_args.hidden_size
    ref_config.num_layers=model_args.n_layers
    ref_config.intermediate_size=4*ref_config.hidden_size
    ref_config.num_heads=model_args.n_heads
    small_model_config = ref_config
    
    seed = training_args.seed
    set_seed(seed)
    
    model = AutoModelForCausalLM.from_config(small_model_config)

    # Train
    loaded_dataset = torch.load(data_args.train_file)
    train_dataset = []
    for bt in tqdm(loaded_dataset):
        new_dic = {'input_ids': numpy.asarray(bt), 'labels': numpy.asarray(bt)}
        train_dataset += [new_dic] 
    train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_dataset))  

    if data_args.subsample != -1:
        # shuffle then select
        train_dataset = train_dataset.shuffle(seed=seed).select(range(data_args.subsample))

    # Eval
    loaded_dataset = torch.load(data_args.eval_file)
    eval_dataset = []
    for bt in tqdm(loaded_dataset):
        new_dic = {'input_ids': numpy.asarray(bt)}
        eval_dataset += [new_dic] 
    eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset))  
    
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = encode_data(train_dataset,
                                tokenizer,
                                max_seq_length=data_args.max_sequence_length,
                                processing_num_workers=data_args.preprocessing_num_workers,
                                overwrite_cache=data_args.overwrite_cache,
                                cache_dir=model_args.cache_dir + '/' + data_args.train_split + '_cache.arrow',
                                no_tokenizer=False,
                                chunk_tokenizer=False
                                )

    eval_dataset = encode_data(eval_dataset,
                            tokenizer,
                            max_seq_length=data_args.max_sequence_length,
                            processing_num_workers=data_args.preprocessing_num_workers,
                            overwrite_cache=data_args.overwrite_cache,
                            cache_dir=model_args.cache_dir + '/' + data_args.train_split + '_eval_cache.arrow',
                            no_tokenizer=False,
                            chunk_tokenizer=False
                            )
      

    ### --- This is a weird way --- ###
    try:
        train_dataset = train_dataset['train']
    except:
        train_dataset = train_dataset

    try:
        eval_dataset = eval_dataset['train']
    except:
        eval_dataset = eval_dataset
    ### --- This is a weird way --- ###

    
    if data_args.subsample != -1:
        train_dataset = train_dataset.select(range(data_args.subsample))

    ### Define Trainer ###    
    trainer_class = Trainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),)
   
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    
if __name__ == "__main__":
    main()






