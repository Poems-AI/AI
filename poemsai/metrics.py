from datasets import load_dataset, load_metric
import gc
from happytransformer import fine_tuning_util
import numpy as np
from poemsai.torch_utils import get_positions_between
import tempfile
from transformers import default_data_collator, Trainer, TrainingArguments
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_truncate#
import torch
from typing import Callable


__all__ = ['compute_lm_metrics', 'eval_model_with_metrics', 'MetadataLessLoss']


def compute_lm_metrics(eval_preds, expect_preds=False):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    # TODO: ignore padding
    labels = labels[:, 1:].reshape(-1)
    logits = logits[:, :-1]
    predictions = (logits if expect_preds else np.argmax(logits, axis=-1)).reshape(-1)
    return metric.compute(predictions=predictions, references=labels)


def eval_model(model, input_filepath, tokenizer, bs=8):
    "Not useful for GPT2, runs OOM always. Use `eval_model_with_metrics` instead."
    datasets = load_dataset("text", data_files={"eval": input_filepath})
    n_procs = 1
    mlm = False
    tokenized_dataset = fine_tuning_util.preprocess_concatenate(tokenizer, datasets, n_procs, mlm)
    dataset = tokenized_dataset['eval']
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        eval_args = TrainingArguments(tmp_dir_name, 
                                      per_device_eval_batch_size=bs, 
                                      report_to=None,
                                      eval_accumulation_steps=1)
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_lm_metrics
    )
    return trainer.evaluate()


def build_trainer_for_eval(model, input_filepath, tokenizer, bs):
    datasets = load_dataset("text", data_files={"eval": input_filepath})
    n_procs = 1
    mlm = False
    tokenized_dataset = fine_tuning_util.preprocess_concatenate(tokenizer, datasets, n_procs, mlm)
    dataset = tokenized_dataset['eval']
    
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        eval_args = TrainingArguments(tmp_dir_name, 
                                      per_device_eval_batch_size=bs, 
                                      report_to=None)
                                      #eval_accumulation_steps=1)
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        data_collator=default_data_collator
    )
    return trainer


def eval_model_with_metrics(model, input_filepath, tokenizer, compute_metrics=compute_lm_metrics,
                            bs=16, on_step=None):
    """Rewrite of happytransformer+HuggingFace eval method that doesn't produce OOM errors related to metrics.
    
    For models whose output logits are big and when `compute_metrics` is not None, HuggingFace 
    `Trainer.evaluate` method accumulates the logits of all data in device memory. That causes
    out of memory errors. This method caches predictions instead of logits to prevent the issue.
    
    Warning: unlike Trainer.evaluate, it's not able to handle distributed training."""
    trainer = build_trainer_for_eval(model, input_filepath, tokenizer, bs)
    dataloader = trainer.get_eval_dataloader()
    
    was_training = model.training
    model.eval()
    
    # Initialize containers
    # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    losses_host = None
    preds_host = None
    labels_host = None
    # losses/preds/labels on CPU (final containers)
    all_losses = None
    all_preds = None
    all_labels = None
    
    num_samples = 0
    
    # Main evaluation loop
    for step, inputs in enumerate(dataloader):
        num_samples += inputs['input_ids'].shape[0]
        
        # Prediction step
        loss, logits, labels = trainer.prediction_step(model, inputs, False)#, ignore_keys=ignore_keys)
        
        # Update containers on host
        if loss is not None:
            #losses = self._nested_gather(loss.repeat(bs))
            losses = loss.repeat(bs)
            losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
        if logits is not None:
            preds = logits.argmax(dim=-1)
            preds_host = preds if preds_host is None else nested_concat(preds_host, preds, padding_index=-100)
        if labels is not None:
            labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            
        if trainer.args.eval_accumulation_steps is not None and (step + 1) % trainer.args.eval_accumulation_steps == 0:
            if losses_host is not None:
                losses = nested_numpify(losses_host)
                all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            if preds_host is not None:
                logits = nested_numpify(preds_host)
                all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
            if labels_host is not None:
                labels = nested_numpify(labels_host)
                all_labels = (
                    labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                )            
            
            # Set back to None to begin a new accumulation
            losses_host, preds_host, labels_host = None, None, None
            torch.cuda.empty_cache()
            gc.collect()
           
        if on_step is not None:
            on_step(step, trainer, loss, logits, labels)
        

    # Gather all remaining tensors and put them back on the CPU
    if losses_host is not None:
        losses = nested_numpify(losses_host)
        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    if preds_host is not None:
        logits = nested_numpify(preds_host)
        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    if labels_host is not None:
        labels = nested_numpify(labels_host)
        all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        
    # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    # samplers has been rounded to a multiple of batch_size, so we truncate.
    if all_losses is not None:
        all_losses = all_losses[:num_samples]
    if all_preds is not None:
        all_preds = nested_truncate(all_preds, num_samples)
    if all_labels is not None:
        all_labels = nested_truncate(all_labels, num_samples)

    metrics = compute_metrics((all_preds, all_labels), expect_preds=True) 
    if all_losses is not None: metrics['eval_loss'] = all_losses.mean().item()
        
    if was_training: model.train()
        
    return metrics


class MetadataLessLoss:
    "Mask the tokens of the target that are considered metadata before passing them to `inner_loss`."
    def __init__(self, inner_loss:Callable, begin_verse_id=None, end_verse_id=None, 
                 end_poem_id=None, ignore_index=-100):
        self.inner_loss = inner_loss
        self.begin_verse_id = begin_verse_id
        self.end_verse_id = end_verse_id
        self.end_poem_id = end_poem_id
        self.ignore_index = ignore_index

    def __call__(self, preds, target):
        target = target.clone()
        bs = target.shape[0]
        seq_len = target.shape[1]
        ignored_positions_by_batch = []

        if (self.begin_verse_id is not None) and (self.end_verse_id is not None):
            # Ignore tokens since end_verse_id until next begin_verse_id, 
            # end_verse_id not included
            for i in range(bs):                            
                ignored_positions_by_batch.append([])
                ini_pos = 0
                for j in range(seq_len):           
                    if target[i][j] == self.end_verse_id:
                        ini_pos = j + 1
                    elif target[i][j] == self.begin_verse_id:
                        ignored_positions_by_batch[i].append((ini_pos, j))
                        ini_pos = -1
                # Add the last slice in case there's no begin_verse_id after
                # the last end_verse_id
                if ini_pos > 0:
                    ignored_positions_by_batch[i].append((ini_pos, j))
                        
        elif (self.end_poem_id is not None) and (self.begin_verse_id is not None):
            # Ignore tokens since end_poem_id until next begin_verse_id, 
            # begin_verse_id not included
            for i in range(bs):
                ignored_positions_by_batch.append([])
                INSIDE_VERSE = -1
                ini_pos = INSIDE_VERSE
                for j in range(seq_len):
                    if target[i][j] == self.end_poem_id:
                        ini_pos = j
                    elif target[i][j] == self.begin_verse_id:
                        if ini_pos != INSIDE_VERSE:
                            ignored_positions_by_batch[i].append((ini_pos, j-1))
                            ini_pos = INSIDE_VERSE

        elif (self.end_poem_id is not None) and (self.end_verse_id is not None):
            # Ignore tokens since last end_verse_id of a poem until end_poem_id, 
            # end_verse_id not included
            for i in range(bs):
                ignored_positions_by_batch.append([])
                ini_pos = 0
                for j in range(seq_len):
                    if target[i][j] == self.end_poem_id:
                        ignored_positions_by_batch[i].append((ini_pos, j))
                        ini_pos = -1
                    if target[i][j] == self.end_verse_id:
                        ini_pos = j + 1
        
        for i, ignored_positions_batch in enumerate(ignored_positions_by_batch):
            for j,k in ignored_positions_batch:
                target[i,j:k+1] = self.ignore_index

        return self.inner_loss(preds, target)


class MetadataLessLossFast:
    """Implementation of MetadataLessLoss a bit more vectorized.
    
    Still incomplete. The case where there are only end-verse and
    end-poem tags is not covered.
    """
    def __init__(self, inner_loss:Callable, begin_verse_id=None, end_verse_id=None, 
                 end_poem_id=None, ignore_index=-100):
        self.inner_loss = inner_loss
        self.begin_verse_id = begin_verse_id
        self.end_verse_id = end_verse_id
        self.end_poem_id = end_poem_id
        self.ignore_index = ignore_index

    def __call__(self, preds, target):
        target = target.clone()
        bs = target.shape[0]
        ignored_positions_by_batch = []

        if (self.begin_verse_id is not None) and (self.end_verse_id is not None):
            # Ignore tokens since end_verse_id until next begin_verse_id, 
            # end_verse_id not included
            for i in range(bs):
                slice_begin_positions, slice_end_positions = get_positions_between(target[i],
                                                                                   self.end_verse_id, 
                                                                                   self.begin_verse_id)
                # If a begin_verse_id appears before the first end_verse_id, we need to add a slice from
                # 0 to its position
                bov_idxs = (target[i] == self.begin_verse_id).nonzero()
                eov_idxs = (target[i] == self.end_verse_id).nonzero()
                if (len(bov_idxs) > 0) and ((len(eov_idxs) == 0) or (bov_idxs[0] < eov_idxs[0])):
                    # Subtract 1 to workaround the general + 1 that avoids including end_verse_id
                    slice_begin_positions = torch.cat([slice_begin_positions, slice_begin_positions.new_zeros(1) - 1])
                    slice_end_positions = torch.cat([slice_end_positions, bov_idxs[0]])
                ignored_positions_by_batch.append((slice_begin_positions+1, slice_end_positions))
        elif (self.end_poem_id is not None) and (self.begin_verse_id is not None):
            # Ignore tokens since end_poem_id until next begin_verse_id, 
            # begin_verse_id not included
            for i in range(bs):
                slice_begin_positions, slice_end_positions = get_positions_between(target[i], 
                                                                                   self.end_poem_id, 
                                                                                   self.begin_verse_id)
                ignored_positions_by_batch.append((slice_begin_positions, slice_end_positions-1))
        elif (self.end_poem_id is not None) and (self.end_verse_id is not None):
            # TODO: Should ignore tokens since last end_verse_id of a poem until end_poem_id, 
            # end_verse_id not included
            pass
                         
        for i, (slice_begin_positions, slice_end_positions) in enumerate(ignored_positions_by_batch):
            for j,k in zip (slice_begin_positions, slice_end_positions):
                target[i,j:k+1] = self.ignore_index
        
        return self.inner_loss(preds, target)
