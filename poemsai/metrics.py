from datasets import load_dataset, load_metric
import gc
from happytransformer import fine_tuning_util
import numpy as np
import tempfile
from transformers import default_data_collator, Trainer, TrainingArguments
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_truncate#
import torch


__all__ = ['compute_lm_metrics', 'eval_model_with_metrics']


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
    print(dataset)
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
