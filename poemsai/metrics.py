from datasets import load_dataset, load_metric
import gc
from happytransformer import fine_tuning_util
import numpy as np
from poemsai.trainer import PoemsTrainer
from poemsai.torch_utils import get_positions_between
import tempfile
from transformers import default_data_collator, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from typing import Callable


__all__ = ['compute_lm_metrics', 'compute_clf_accuracy', 'eval_model_with_metrics', 
           'MetadataLessLoss', 'preprocess_logits_for_metadataless_loss', 
           'preprocess_logits_for_accuracy', 'compute_lm_accuracy', 
           'get_compute_metrics_metadataless']


accuracy_metric = load_metric("accuracy")


def compute_lm_metrics(eval_preds, expect_preds=False):
    logits, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    logits = logits[:, :-1]
    predictions = (logits if expect_preds else np.argmax(logits, axis=-1)).reshape(-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_clf_accuracy(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1).reshape(-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


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


def preprocess_logits_for_accuracy(logits, labels):
    return logits.argmax(dim=-1)


def compute_lm_accuracy(eval_preds):
    """Computes the accuracy given the predictions and labels, contained in `eval_preds`.
    
    It assumes the logits have been reduced with argmax(-1) by `preprocess_logits_for_accuracy`.
    """
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by `preprocess_logits_for_metrics` but we need to shift the labels.
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


def eval_model_with_metrics(model, input_filepath, tokenizer, compute_metrics=compute_lm_accuracy,
                            preprocess_logits_for_metrics=preprocess_logits_for_accuracy,
                            bs=16):
    "Evaluates `model` straightaway with `compute_metrics` using the data contained in `input_filepath`."
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
    trainer = PoemsTrainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    return trainer.evaluate()


class MetadataLessLoss:
    "Masks the tokens of the target that are considered metadata before passing them to `inner_loss`."
    def __init__(self, inner_loss:Callable, begin_verse_id=None, end_verse_id=None, 
                 end_poem_id=None, ignore_index=-100, flatten_inner_loss_args=False):
        self.inner_loss = inner_loss
        self.begin_verse_id = begin_verse_id
        self.end_verse_id = end_verse_id
        self.end_poem_id = end_poem_id
        self.ignore_index = ignore_index
        self.flatten_inner_loss_args = flatten_inner_loss_args

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

        if self.flatten_inner_loss_args:
            preds = preds.view(-1, preds.shape[-1])
            target = target.view(-1)

        return self.inner_loss(preds, target)


class MetadataLessLossFast:
    """Implementation of MetadataLessLoss a bit more vectorized.
    
    Still incomplete. The case where there are only end-verse and
    end-poem tags is not covered.
    """
    def __init__(self, inner_loss:Callable, begin_verse_id=None, end_verse_id=None, 
                 end_poem_id=None, ignore_index=-100, flatten_inner_loss_args=False):
        self.inner_loss = inner_loss
        self.begin_verse_id = begin_verse_id
        self.end_verse_id = end_verse_id
        self.end_poem_id = end_poem_id
        self.ignore_index = ignore_index
        self.flatten_inner_loss_args = flatten_inner_loss_args

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
        
        if self.flatten_inner_loss_args:
            preds = preds.view(-1, preds.shape[-1])
            target = target.view(-1)

        return self.inner_loss(preds, target)


def preprocess_logits_for_metadataless_loss(logits, labels):
    labels = labels[:, 1:]
    logits = logits[:, :-1]
    return -torch.gather(F.log_softmax(logits, dim=-1), -1, labels[..., None]).squeeze(-1)


def get_compute_metrics_metadataless(**loss_init_kargs):
    "It returns a functon that computes the `MetadataLessLoss` like a HuggingFace metric."
    ignore_index = -100
    def _inner_loss(preds, target): 
        return preds[target != ignore_index].mean()
    loss_fn = MetadataLessLoss(_inner_loss, ignore_index=ignore_index, **loss_init_kargs)
    
    def compute_metadataless_loss(eval_preds):
        # preds just contains the value of the log softmax of the logits for the entries given
        # by the labels, so it has the same rank as labels.
        preds, labels = eval_preds
        # preds have been already truncated (last item of the sequence), by 
        # preprocess_logits_for_metadataless_loss so we only need to shift the labels
        loss = loss_fn(torch.Tensor(preds), torch.Tensor(labels[:, 1:]))
        return {'Metadata-less val. loss': loss}
    
    return compute_metadataless_loss
