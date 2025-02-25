from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Union, List
from transformers import TrainingArguments as TA

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to the teacher model"}
    )
    
    parent_path: str = field(
        metadata={"help": "Path to the parent model"}
    )
    
    cache_dir: Optional[str] = field(
        default='/scratch/gpfs/ap34/hf_models',
        metadata={"help": "cache dir"}
    )
    
    #arguments for the
    n_layers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of layers in the student model"}
    )
    
    n_heads: Optional[int] = field(
        default=16,
        metadata={"help": "Number of attention heads in the student model"}
    )
        
    hidden_size: Optional[int] = field(
        default=64,
        metadata={"help": "Hidden size of the student model"}
    )
    
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "Temperature to apply to teacher's logits"}
    )
   
    
    ref_hidden_size: Optional[int] = field(
        default=64,
        metadata={"help": "Hidden size of the teacher model"}
    )
        
    ref_n_layers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of layers in the teacher model"}
    )

    ref_n_heads: Optional[int] = field(
        default=16,
        metadata={"help": "Number of attention heads in the teacher model"}
    )

    vocab_size:  Optional[int] = field(
        default=-1,
        metadata={"help": "Size of vocabulary"}
    )

    pad_token_id: Optional[int] = field(
        default=-1,
        metadata={"help": "pad token id"}
    )
        
    mask_token_id: Optional[int] = field(
        default=-1,
        metadata={"help": "mask token id"}
    )    

    cls_token_id: Optional[int] = field(
        default=-1,
        metadata={"help": "cls token id"}
    )    
   
@dataclass
class TrainingArguments(TA):
    """
    Arguments pertaining to training arguments for distillation
    """
    #loss function
    loss: Optional[str] = field(
        default='teacher_KL',
        metadata={"help": "teacher_KL/next_word, Loss function to use to train the student"}
    )

    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed to use for training (including data shuffling)"}
    )
    

    distill_version: Optional[str] = field(
        default='gt_label',
        metadata={"help": "the distillation version to use: 'gt_label', 'topk"}
    )

    distill_topk: Optional[int] = field(
        default=5,
        metadata={"help": "topk to use for distillation"}
    )
    
    boundaries: Optional[str] = field(
        default="0_20",
        metadata={"help": "Weight of loss for two-step KL loss (or boundary distillation), list represented as str separated by '_' "}
    )
        
    teacher_ckpts: Optional[str] = field(
        default="0_20",
        metadata={"help": "If boundary distillation, list of teacher checkpoints represented as str separated by '_' "}
    )    
    
    num_heads: Optional[int] = field(
        default=2,
        metadata={"help": "Path to surrogate of the teacher model"}
    )
        
    ref_model_ckpt_dir: Optional[str] = field(
        default="",
        metadata={"help": "Path to the progressive checkpoints of the teacher"}
    ) 
    
    refresh_frequency: Optional[int] = field(
        default=5,
        metadata={"help": "Number of samples to compute the calibrated sampling over"}
    )
    
    speed_factor: Optional[float] = field(
        default=5,
        metadata={"help": "Number of samples to compute the calibrated sampling over"}
    )

    pcfg_name: Optional[str] = field(
        default='',
        metadata={"help": "Name of the PCFG"}
    )

    
    train_mode: Optional[str] = field(
        default='CE',
        metadata={"help": "Training mode: 'CE' (cross entropy) or 'distill'."}
    )

    pid: Optional[int] = field(
        default=-1,
        metadata={"help": "Process ID"}
    )

    project_name: Optional[str] = field(
        default='PCFG_classification_repeat',
        metadata={"help": "Name of the PCFG"}
    )

    shift_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to shift input and labels 2 to the left"}
    ) 

    
    
@dataclass
class DataArguments:
    #dataset to train on
    train_file: Optional[str] = field(
        default='',
        metadata={"help": " Path to the dataset"}
    )

    eval_file: Optional[str] = field(
        default='',
        metadata={"help": " Path to the dataset"}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
        
         
    max_sequence_length: Optional[int] = field(
        default=512,
        metadata={"help": "Context length"}
    )
        
    subsample:  Optional[int] = field(
        default=-1,
        metadata={"help": "If we want to train on a subsample of data"}
    )
    
    
@dataclass
class Unionable:
    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))