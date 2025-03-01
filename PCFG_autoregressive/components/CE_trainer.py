from transformers import Trainer
from typing import Dict, Optional
import torch 
import torch
from torch.utils.data import SequentialSampler

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import nested_detach


class CETrainer(Trainer):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_version = 'CE'

        self.step_cnt = 0
        self.logging_steps = self.args.logging_steps
        self.eval_acc = 0.
        self.eval_num = 0.
        self.eval_start = False
        self.label_names = ['labels']

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        if self.args.ordered:
            return SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()


    def accuracy_eval(self, model, inputs):
        labels = inputs.pop('labels')[..., 1:]
        padding_mask = labels.eq(-100)   

        outputs = model(**inputs)
        #apply_softmax
        preds = outputs.logits.detach()
        accs = (preds[..., :-1, :].argmax(axis=-1) == labels).type(torch.float) * (labels != -100).type(torch.float)
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
        

        # track the step as the number of batches seen
        # originally step = self.state.global_step
        output = {**logs, **{"step": self.step_cnt}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else any(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():    
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)