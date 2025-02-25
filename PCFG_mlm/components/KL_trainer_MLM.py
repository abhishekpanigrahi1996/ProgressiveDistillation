import torch
from transformers import Trainer
from transformers import AutoModelForCausalLM, BertForMaskedLM
from typing import Dict
import os 
from glob import glob 
from torch.utils.data import DataLoader
import pdb 

class KLTrainer_MLM(Trainer):
    
    def __init__(self, ref_model, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model.cuda()  #here we say that we have another model to refer to
        self.ref_config = self.ref_model.config
        self.ignore_index = -100
        self.temperature = temperature
        self.speed_factor = self.args.speed_factor
        self.refresh_frequency = self.args.refresh_frequency
        self.ref_created_ckpts = []
      
        self.ref_model.requires_grad = False
        self.ref_model_ckpt_dir = self.args.ref_model_ckpt_dir

        self.distill_version = self.args.distill_version
        self.distill_topk = self.args.distill_topk

        self.step_cnt = 0
        self.logging_steps = self.args.logging_steps
        
        all_ckpts = glob(os.path.join(self.ref_model_ckpt_dir, 'checkpoint-*'))
        self.all_ckpts = {int(x.split('-')[-1]):x for x in all_ckpts}

        self.eval_acc = 0.
        self.eval_num = 0.
        self.eval_start = False
        
    def accuracy_eval(self, model, inputs, labels, padding_mask):
        if len(labels.shape) == 3:
            labels = labels.squeeze()
            padding_mask = padding_mask.squeeze()

        outputs = model(**inputs)
        preds = outputs.logits.detach()
        accs = (preds.argmax(axis=-1) == labels).type(torch.float) * (1. - (padding_mask).type(torch.float))
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        acc = accs.sum() / (1e-10 + num_active_elements)
        
        

        self.eval_acc += acc.item() 
        self.eval_num += 1

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """


        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        logits = logits.contiguous()
        labels = labels.contiguous()
        
        if labels.dim() == logits.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        labels.clamp_min_(0)
        
        if model.training:
            if self.eval_start:
                self.eval_start = False
                self.log({"accuracy": self.eval_acc/self.eval_num})
                self.eval_acc = 0.
                self.eval_num = 0.

            ### Train loss
            with torch.inference_mode():
                ref_outputs = self.ref_model(**inputs) 
                ref_logits = ref_outputs["logits"]
                ref_logits =  ref_logits.contiguous()

            log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
            ref_probs = torch.nn.functional.softmax(ref_logits * 1./self.temperature, dim=-1)

            if self.distill_version == 'gt_label':
              kl_loss = torch.sum(ref_probs * log_probs, axis=-1, keepdim=True)
              
            elif self.distill_version == 'topk':
              # learn from the top k choices in the entire vocab
              topk_ref_probs, topk_indices = torch.topk(ref_probs, self.distill_topk, dim=-1)
              topk_log_probs = torch.gather(log_probs, dim=-1, index=topk_indices)
              kl_loss = torch.sum(topk_ref_probs * topk_log_probs, axis=-1, keepdim=True)

            auto_loss = log_probs.gather(dim=-1, index=labels)
            loss = kl_loss
            
            loss.masked_fill_(padding_mask, 0.0)
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            loss = loss.sum() / num_active_elements

            if self.step_cnt % self.logging_steps == 0:
              losses_no_grad = {'auto_loss': auto_loss.mean().item(),
                                'kl_loss': kl_loss.mean().item()}
              self.log(losses_no_grad)
        else:
            if not self.eval_start:
                self.eval_start = True
            self.accuracy_eval(model, inputs, labels, padding_mask)
               
            ### Eval loss
            log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
            loss = log_probs.gather(dim=-1, index=labels)
        
            loss.masked_fill_(padding_mask, 0.0)
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            loss = loss.sum() / num_active_elements

            if self.step_cnt % self.logging_steps == 0:
              losses_no_grad = {'eval_loss': loss.item()}
              self.log(losses_no_grad)

        self.step_cnt += 1

        # return (losses, outputs) if return_outputs else losses 
        return (loss, outputs) if return_outputs else loss
    

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        
        ### Load a new checkpoint to the ref_model in steps of refresh frequency
        if self.speed_factor != -1:
            
            if self.state.global_step % self.refresh_frequency == 0 :
                
                if self.speed_factor < 1:
                    checkpoint_step = (1 + int(self.speed_factor * (self.state.global_step // self.refresh_frequency))) * self.refresh_frequency
                else:
                    checkpoint_step = (self.speed_factor * int(1 + self.state.global_step // self.refresh_frequency)) * self.refresh_frequency
                checkpoint_step = int(checkpoint_step)    
                          
                if checkpoint_step in self.ref_created_ckpts:
                    pass
                else:
                    if checkpoint_step in self.all_ckpts:
                      valid_checkpoint_step = checkpoint_step
                    else:
                      valid_checkpoint_step = max([k for k in self.all_ckpts.keys() if k-1 < checkpoint_step])
                    try:
                        ckpt_path = os.path.join(self.ref_model_ckpt_dir, 'checkpoint-'+str(valid_checkpoint_step))
                        self.ref_model = BertForMaskedLM.from_pretrained(ckpt_path, config=self.ref_config)
                        print ('Loaded Teacher ckpt', valid_checkpoint_step)
                        self.ref_created_ckpts += [valid_checkpoint_step]
                        self.ref_model.cuda()
                    except:
                        self.ref_created_ckpts += [valid_checkpoint_step]
                        print(f"\n\nCheckpoint step = {valid_checkpoint_step}: not found & skipped.\n\n")
                        pass
        else:
            raise NotImplementedError  

        self.ref_model.eval()
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            if type(loss) == tuple:
                loss = loss[0]

        if self.args.n_gpu > 1:
            loss = loss.mean() 
            
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
      

    # custom log to include the step number
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        #if self.args.include_num_input_tokens_seen:
        #    logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen


        # track the step as the number of batches seen
        # (originally step = self.state.global_step)
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