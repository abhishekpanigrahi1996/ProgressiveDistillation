import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from components.all_arguments import ModelArguments, TrainingArguments, DataArguments
from components.load_data import encode_data
from components.KL_trainer_progressive import KLTrainer_progressive

from datasets import Dataset, concatenate_datasets, load_from_disk

import os
import pandas as pd
import numpy
from tqdm import tqdm
from transformers import set_seed
import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy



import pdb

def main():
    parser = HfArgumentParser((ModelArguments,TrainingArguments,DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()    
    training_args.tn_heads = model_args.n_heads
    training_args.thidden_size = model_args.hidden_size
    training_args.tn_layers = model_args.n_layers
    training_args.t_temperature = model_args.temperature
    # get pid
    pid = os.getpid()
    training_args.pid = pid

    os.environ["WANDB_PROJECT"]=training_args.project_name
    os.environ["WANDB_MODE"]="offline"

    seed = training_args.seed
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.parent_path,
                                            cache_dir=model_args.cache_dir,)
    tokenizer.pad_token = tokenizer.eos_token
    ref_config = AutoConfig.from_pretrained(model_args.parent_path,
                                        cache_dir=model_args.cache_dir,)
    
    ref_config.n_embd = model_args.ref_hidden_size
    ref_config.n_head = model_args.ref_n_heads
    ref_config.n_layer = model_args.ref_n_layers
    if model_args.vocab_size != -1:
        ref_config.vocab_size = model_args.vocab_size
        tokenizer.pad_token_id = tokenizer.eos_token_id = model_args.vocab_size - 1
    
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)#, config=ref_config) #<-- removed this

    small_model_config = AutoConfig.from_pretrained(model_args.parent_path,
                                            cache_dir=model_args.cache_dir,)

    small_model_config.n_embd = model_args.hidden_size
    small_model_config.n_head = model_args.n_heads
    small_model_config.n_layer = model_args.n_layers
    if model_args.vocab_size != -1:
        small_model_config.vocab_size = model_args.vocab_size
        tokenizer.pad_token_id = tokenizer.eos_token_id = model_args.vocab_size - 1

    model = AutoModelForCausalLM.from_config(small_model_config)

    
    # Train
    if ':' in data_args.train_file:
        all_fs = data_args.train_file.split(':')
        train_dataset = []            
        for f in tqdm(all_fs):
            try:
                d = load_from_disk(f)    
            except:
                continue
            train_dataset += [d]                
        train_dataset = concatenate_datasets(train_dataset)
    else:   
        try:
            loaded_dataset = torch.load(data_args.train_file)
            train_dataset = []
            for bt in tqdm(loaded_dataset):
                new_dic = {'input_ids': numpy.asarray(bt), 'labels': numpy.asarray(bt)}
                train_dataset += [new_dic] 
            train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_dataset))  
        except:
            train_dataset = load_from_disk(data_args.train_file)
        
    if data_args.subsample != -1:
        # shuffle then select
        train_dataset = train_dataset.shuffle(seed=seed).select(range(min(data_args.subsample, len(train_dataset))))

    # Eval
    loaded_dataset = torch.load(data_args.eval_file)
    eval_dataset = []
    if len(loaded_dataset) > 1:
        data, labels = loaded_dataset
        for arr_bt, arr_label in tqdm(zip(data, labels)):
            for bt, label in tqdm(zip(arr_bt, arr_label)):
                new_dic = {'input_ids': numpy.asarray(bt), 'labels': numpy.asarray(label)}
                eval_dataset += [new_dic] 
        eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset)) 
    else: 
        for bt in tqdm(loaded_dataset):
            new_dic = {'input_ids': numpy.asarray(bt), 'labels': numpy.asarray(bt)}
            eval_dataset += [new_dic] 
        eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset)) 

     

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    

    # define trainer       
    trainer_class = KLTrainer_progressive
        
    trainer = trainer_class(
        ref_model=ref_model,
        temperature=model_args.temperature,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),)


    if trainer.is_fsdp_enabled:
        # Override accelerate defaults
        trainer.accelerator.state.fsdp_plugin.limit_all_gathers = True
        trainer.accelerator.state.fsdp_plugin.sync_module_states = False

        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
        trainer.accelerator.state.fsdp_plugin.backward_prefetch = BackwardPrefetch.BACKWARD_PRE

        # Identify which modules have "_fsdp_wrap" attribute set to True and wrap these
        def fsdp_policy_fn(module):
            return getattr(module, "_fsdp_wrap", False)

        # Identify which modules have "layer" in their class name and use these
        # as the basic FSDP blocks that are sharded and exchanged between GPUs
        # def layer_policy_fn(module):
            # return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=fsdp_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
    
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  
        
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
if __name__ == "__main__":
    main()






