import torch
from transformers import AutoTokenizer, AutoConfig, BertForMaskedLM
from transformers import HfArgumentParser, DataCollatorForSeq2Seq
from components.all_arguments import ModelArguments, TrainingArguments, DataArguments
from datasets import Dataset, load_from_disk, concatenate_datasets

from components.MLM_trainer import MLMtrainer
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import functools

import os
import pandas as pd
import numpy
from tqdm import tqdm
from transformers import set_seed


def main():
    parser = HfArgumentParser((ModelArguments,TrainingArguments,DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()    
    training_args.tn_heads = model_args.n_heads
    training_args.thidden_size = model_args.hidden_size
    training_args.tn_layers = model_args.n_layers
    
    os.environ["WANDB_PROJECT"]=training_args.project_name
    os.environ["WANDB_MODE"]="offline"
    
    # get pid
    pid = os.getpid()
    training_args.pid = pid
    # set seed
    seed = training_args.seed
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir,)
    tokenizer.pad_token = tokenizer.eos_token

    ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            cache_dir=model_args.cache_dir,)

    ref_config.hidden_size = model_args.hidden_size
    ref_config.intermediate_size = model_args.hidden_size * 4
    ref_config.num_hidden_layers = model_args.n_layers
    ref_config.num_attention_heads = model_args.n_heads
    if model_args.vocab_size != -1:
        ref_config.vocab_size = model_args.vocab_size
        tokenizer.pad_token_id = tokenizer.sep_token_id = tokenizer.unk_token_id = model_args.pad_token_id
        tokenizer.mask_token_id = model_args.mask_token_id
        tokenizer.cls_token_id = model_args.cls_token_id
    ref_config.max_position_embeddings = 1024
    model = BertForMaskedLM(ref_config)
    trainer = MLMtrainer

    
    # Train
    loaded_dataset = torch.load(data_args.train_file)
    train_dataset = []
    if len(loaded_dataset) > 1:
        data, labels, masks = loaded_dataset
        for arr_bt, arr_label, arr_mask in tqdm(zip(data, labels, masks)):
            for bt, label, mask in tqdm(zip(arr_bt, arr_label, arr_mask)):
                new_dic = {'input_ids': numpy.asarray(bt), 'labels': numpy.asarray(label), 'attention_mask': numpy.asarray(mask)}
                train_dataset += [new_dic] 
        train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_dataset)) 

    # Eval
    loaded_dataset = torch.load(data_args.eval_file)
    eval_dataset = []
    if len(loaded_dataset) > 1:
        data, labels, masks = loaded_dataset
        for arr_bt, arr_label, arr_mask in tqdm(zip(data, labels, masks)):
            for bt, label, mask in tqdm(zip(arr_bt, arr_label, arr_mask)):
                new_dic = {'input_ids': numpy.asarray(bt), 'labels': numpy.asarray(label), 'attention_mask': numpy.asarray(mask)}
                eval_dataset += [new_dic] 
        eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset)) 
    

    if data_args.subsample != -1:
        # shuffle then select
        train_dataset = train_dataset.shuffle(seed=seed).select(range(min(data_args.subsample, len(train_dataset))))


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

    ### Define Trainer ###    
    trainer_class = trainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=data_args.max_sequence_length),
        )
    
    if trainer.is_fsdp_enabled:
        # Override accelerate defaults
        trainer.accelerator.state.fsdp_plugin.limit_all_gathers = True
        trainer.accelerator.state.fsdp_plugin.sync_module_states = False

        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
        trainer.accelerator.state.fsdp_plugin.backward_prefetch = BackwardPrefetch.BACKWARD_PRE

        def fsdp_policy_fn(module):
            return getattr(module, "_fsdp_wrap", False)

        
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=fsdp_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy


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






