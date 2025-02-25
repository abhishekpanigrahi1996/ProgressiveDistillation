from transformers import Trainer
from typing import Dict
import torch 
import pdb 
import torch
from torch.utils.data import DataLoader


class MLMtrainer(Trainer):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_version = 'CE'

        self.step_cnt = 0
        self.logging_steps = self.args.logging_steps
        self.eval_acc = 0.
        self.eval_num = 0.
        self.eval_start = False

    def accuracy_eval(self, model, inputs):
        labels = inputs.pop('labels')
        padding_mask = labels.eq(-100)   

        outputs = model(**inputs)
        #apply_softmax
        preds = outputs.logits.detach()
        accs = (preds.argmax(axis=-1) == labels).type(torch.float) * (labels != -100).type(torch.float)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        acc = accs.sum() / (1e-10 + num_active_elements)
        
        self.eval_acc += acc.item() 
        self.eval_num += 1

    def compute_loss(self, model, inputs, return_outputs=False):

 
        ret = super().compute_loss(model, inputs, return_outputs=return_outputs)
        self.step_cnt += 1
        if not model.training:
            if not self.eval_start:
                self.eval_start = True
            self.accuracy_eval(model, inputs)
        else:
            if self.eval_start:
                self.eval_start = False
                self.log({"accuracy": self.eval_acc/self.eval_num})
                self.eval_acc = 0.
                self.eval_num = 0.


        return ret 

    # custom log to include the step number
    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        output = {**logs, **{"step": self.step_cnt}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,   # <--  Change
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))