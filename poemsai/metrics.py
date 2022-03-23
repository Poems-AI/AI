import copy
from datasets import Dataset, load_dataset, load_metric
import gc
from happytransformer import fine_tuning_util
import io
import numpy as np
import pandas as pd
from poemsai.data import (
    BaseLabelsDecoder, BaseLabelsWriter, label_type_to_str, LabelsType, LabelsWriterStd, 
    PoemsFileConfig, PoemsIOWriter, VerseGrouping
)
from poemsai.trainer import PoemsTrainer
from poemsai.torch_utils import get_positions_between
import tempfile
from transformers import default_data_collator, TextGenerationPipeline, Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List


__all__ = [
    'compute_lm_metrics', 'compute_clf_accuracy', 'eval_model_with_metrics', 'MetadataLessLoss', 
    'preprocess_logits_for_metadataless_loss', 'preprocess_logits_for_accuracy', 
    'compute_lm_accuracy', 'get_compute_metrics_metadataless', 'ConditionalGenEvaluator', 
    'ConditionalGenLoss', 
]


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
                                      report_to=["none"],
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
                                      report_to=["none"],
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
                 end_poem_id=None, ignore_index=-100, flatten_inner_loss_args=False,
                 n_initial_verses_to_ignore=0):
        self.inner_loss = inner_loss
        self.begin_verse_id = begin_verse_id
        self.end_verse_id = end_verse_id
        self.end_poem_id = end_poem_id
        self.ignore_index = ignore_index
        self.flatten_inner_loss_args = flatten_inner_loss_args
        self.n_initial_verses_to_ignore = n_initial_verses_to_ignore

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
                DONT_KNOW = -1
                # At the beginning we don't know if we start at the beginning of a poem
                # because a poem may have been split into two sequences
                n_verses_completed = DONT_KNOW
                for j in range(seq_len):
                    if target[i][j] == self.end_poem_id:
                        if self.n_initial_verses_to_ignore > 0:
                            #ini_pos = j
                            n_verses_completed = 0
                        else:
                            ignored_positions_by_batch[i].append((ini_pos, j))                            
                            ini_pos = -1

                    if target[i][j] == self.end_verse_id:
                        if n_verses_completed != DONT_KNOW:
                            n_verses_completed += 1
                            if n_verses_completed == self.n_initial_verses_to_ignore:
                                ignored_positions_by_batch[i].append((ini_pos, j))
                                ini_pos = j + 1
                            elif n_verses_completed > self.n_initial_verses_to_ignore:
                                ini_pos = j + 1
                        else:
                            ini_pos = j + 1

                # Ignore the last segment if we ended the loop while being at the beginning of a poem
                # but before the (self.n_initial_verses_to_ignore)th end of verse
                if 0 <= n_verses_completed < self.n_initial_verses_to_ignore:
                    ignored_positions_by_batch[i].append((ini_pos, j))
        
        for i, ignored_positions_batch in enumerate(ignored_positions_by_batch):
            for j,k in ignored_positions_batch:
                target[i,j:k+1] = self.ignore_index

        if self.flatten_inner_loss_args:
            preds = preds.view(-1, preds.shape[-1])
            target = target.view(-1)

        return self.inner_loss(preds, target)


class MetadataLessLossFast:
    """Implementation of `MetadataLessLoss` a bit more vectorized.
    
    Still incomplete. The case where there are only end-verse and end-poem tags is not covered. 
    It's only worth going on if a specific need of speed arises.
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


class ConditionalGenEvaluator:
    def __init__(self, gen_model, gen_tokenizer, clf_model, clf_tokenizer, 
                 file_conf:PoemsFileConfig, cat_evaluated:str, all_cats_ordered:List[str]=None, 
                 labels_writer:BaseLabelsWriter=None, device=-1):
        self.clf_model = clf_model
        self.clf_tokenizer = clf_tokenizer
        self.file_conf = copy.deepcopy(file_conf)
        # The verse_grouping setting is irrelevant for training because the line breaks are
        # removed during preprocessing but here we need to ensure that no extra line breaks 
        # are added (there's still file_conf.end_of_verse_token), with OnePoemBySequence
        self.file_conf.verse_grouping = VerseGrouping.OnePoemBySequence
        self.cat_evaluated = cat_evaluated
        self.all_cats_ordered = all_cats_ordered
        assert (all_cats_ordered is None) or (len(all_cats_ordered) == 0) or (self.cat_evaluated in all_cats_ordered), (
            '`all_cats_ordered` must include the category evaluated by this object'
        )
        self.labels_writer = labels_writer if labels_writer is not None else LabelsWriterStd()
        self.gen_pipeline = TextGenerationPipeline(model=gen_model, tokenizer=gen_tokenizer, device=device)
    
    def _get_labels(self) -> List[str]:
        return list(self.clf_model.config.label2id.keys())
    
    def _label_to_dict(self, label:str) -> Dict[str, str]:
        labels_dict = dict()
        if (self.all_cats_ordered is not None) and (len(self.all_cats_ordered) > 0):
            for cat in self.all_cats_ordered:
                labels_dict[cat] = label if cat == self.cat_evaluated else ''
        else:
            labels_dict[self.cat_evaluated] = label
        return labels_dict

    def _labels_dict_to_str(self, labels_dict:dict) -> str:
        with io.StringIO() as stream:
            poems_writer = PoemsIOWriter(stream, self.file_conf)
            self.labels_writer.write_labels(labels_dict, poems_writer)
            return stream.getvalue()
    
    def _label_to_formatted_str(self, label:str) -> str:
        return self._labels_dict_to_str(self._label_to_dict(label))

    def _preprocess_clf_ds(self, ds):
        tokenized_ds = self.clf_tokenizer(ds["text"], truncation=True)
        tokenized_ds["labels"] = [self.clf_model.config.label2id[l] for l in ds["labels"]]
        return tokenized_ds    
    
    def eval_with_labels_as_prompt(self, min_samples=1000):
        labels = self._get_labels()
        text = [self._label_to_formatted_str(l) for l in labels]
        if len(labels) < min_samples:
            int_ratio = round(min_samples / len(labels))
            labels = labels * int_ratio
            text = text * int_ratio
        return self._evaluate(text, labels)
    
    def eval_with_seq_fragment_as_prompt(self, labeled_df, seq_len_pct=0.25, max_prompt_len=100):
        labels = self._get_labels()
        # Filter out sequences whose labels are not used by the classifier
        labeled_df = labeled_df[labeled_df.labels.isin(labels)]
        
        def _format_text(text):
            with io.StringIO() as stream:
                poems_writer = PoemsIOWriter(stream, self.file_conf)
                for verse in text.split('\n'):
                    poems_writer.write_verse(verse)
                formatted_text = stream.getvalue()
            # Delete the last end of verse token
            eov_token = self.file_conf.end_of_verse_token
            if (len(eov_token) > 0) and (eov_token in formatted_text):
                formatted_text = formatted_text[:formatted_text.rindex(eov_token)]

            end_idx = min(max_prompt_len, int(len(formatted_text) * seq_len_pct))
            # Expand until the end of the last word that fits, at least partially, into the
            # maximum length
            if ' ' in formatted_text:
                end_idx = formatted_text.index(' ', end_idx)
            return formatted_text[:end_idx]

        text = labeled_df.apply(lambda row: self._label_to_formatted_str(row.labels) + _format_text(row.text), axis=1)
        return self._evaluate(text.to_list(), labeled_df.labels.to_list())
    
    def _replace_special_tokens(self, text:str):
        text = text.replace(self.file_conf.beginning_of_verse_token, '')
        text = text.replace(self.file_conf.end_of_poem_token, '')
        text = text.replace(self.file_conf.end_of_verse_token, '\n')
        return text
    
    def _evaluate(self, prompts, labels):
        out_sequences = self.gen_pipeline(
            prompts, 
            min_length=10,
            #return_full_text=False,
            #return_text=False,
            max_length=100,
            do_sample=True,
            early_stopping=False,
            num_beams=1,
            temperature=1.,
            top_k=50,
            no_repeat_ngram_size=0,
            top_p=0.98,
            return_tensors=True,
        )        
        # Convert to text and delete the prompts from out_sequences
        # We tokenize the prompts first because some of their characters may disappear in the decoded out_text
        #tokenized_prompts = self.gen_pipeline.tokenizer(prompts)['input_ids']
        out_text = [self.gen_pipeline.tokenizer.decode(seq_token_ids['generated_token_ids'][0])
                    for seq_token_ids in out_sequences]
        out_text = [self._replace_special_tokens(t) for t in out_text]
        
        clf_input_df = pd.DataFrame(columns=['text', 'labels'])
        # Delete the labels from out_text before evaluation
        n_total_cats = 1 if self.all_cats_ordered is None else len(self.all_cats_ordered)
        n_verses_for_labels = self.labels_writer.num_verses_needed(n_total_cats)
        clf_input_df['text'] = ['\n'.join(seq.split('\n')[n_verses_for_labels:]) for seq in out_text]
        clf_input_df['labels'] = labels

        ds = Dataset.from_pandas(clf_input_df)
        tokenized_ds = ds.map(self._preprocess_clf_ds, batched=True)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tr_args = TrainingArguments(tmp_dir_name, report_to=["none"])
            clf_trainer = Trainer(
                model=self.clf_model, 
                args=tr_args, 
                eval_dataset=tokenized_ds, 
                tokenizer=self.clf_tokenizer,
                compute_metrics=compute_clf_accuracy,
            )
            eval_results = clf_trainer.evaluate()
        return eval_results


class ConditionalGenLoss:
    """Loss that incentivizes a text generator to produce sequences of a type contained in the target text.
    
    `clf` predicts a class for each poem generated by a text generator (`gen_output` param of `__call__`) and
    each class is compared with a label contained in `gen_output`, resulting in a lower loss when the classes
    coincide.
    
    It assumes:
    1. `clf` has the same base architecture and tokenizer as the generator.
    2. If the generator has a bigger vocab than the classifier, the additional tokens have the biggest indices.    
    
    Args
        clf: HuggingFace classifier of poems by `cat`, already trained. Its input embedding layer is modified and 
            expanded with dummy entries if it has a lower size than the output embeddings of the generator, i.e. 
            if the vocab of 'clf' is smaller than the vocab of the generator. If you are going to use this classifier 
            later, consider passing a clone.
        clf_tokenizer: tokenizer used by `clf`.
        gen_tokenizer: tokenizer of the text generator that creates the input `gen_output` received by `__call__`.
        cat: category or type of labels for which `clf` classifies sequences.
        labels_decoder: decoder able to extract the labels from the text of a poem encoded according to `file_config`
            and a `BaseLabelsWriter` child class.
            It must be of type `LabelsDecoderKeyValue` if the input labels of the generator were prepended to each
            poem using `LabelsWriterKeyValue`.
            It must be of type `LabelsDecoderExplained` if the input labels of the generator were prepended to each
            poem using `LabelsWriterExplained`.
        file_config: configuration used to generate the inputs of the text generator.
        gen_eop_token_id: id of the end of poem token in the generator vocabulary.
        gen_bov_token_id: id of the beginning of verse token in the generator vocabulary.
        gen_eov_token_id: id of the end of verse token in the generator vocabulary.
        device: device, with Pytorch format, where the input tensors are expected to be. If needed, `clf` weights are 
            moved to this device.
    """
    def __init__(
        self, clf:nn.Module, clf_tokenizer, gen_tokenizer, cat:LabelsType, labels_decoder:BaseLabelsDecoder, 
        file_config:PoemsFileConfig, gen_eop_token_id:int, gen_bov_token_id:int=None, gen_eov_token_id:int=None, 
        device=None
    ):
        self.clf = clf
        self.clf.eval()
        if device is not None:
            self.clf.to(device)
        self.clf_tokenizer = clf_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.cat = cat
        self.labels_decoder = labels_decoder
        self.file_config = file_config
        self.gen_eop_token_id = gen_eop_token_id
        self.gen_bov_token_id = gen_bov_token_id
        self.gen_eov_token_id = gen_eov_token_id
        self.clf_whitespace_token_id = clf_tokenizer.encode(' ')[0]
        self.clf_eol_token_id = clf_tokenizer.encode('\n')[0]
        self.inner_loss = nn.CrossEntropyLoss(reduction='sum')
        
        gen_vocab_sz = len(self.gen_tokenizer)
        self._add_missing_entries_to_clf_embedding(clf, gen_vocab_sz)
        
    def _add_missing_entries_to_clf_embedding(self, clf, gen_vocab_sz):
        clf_emb_w = clf.get_input_embeddings().weight
        clf_vocab_sz = clf_emb_w.shape[0]
        
        if clf_vocab_sz < gen_vocab_sz:
            # We resize but don't set the content of the new entries, it's irrelevant because
            # in `__call__` we modify the predictions that reference the extra tokens to tokens known by the
            # classifier, by calling `_replace_special_tokens_for_clf`.
            clf.resize_token_embeddings(gen_vocab_sz)
    
    def _split_by_poem(self, single_gen_output, single_gen_target):
        seq_len = single_gen_output.shape[0]
        split_idxs = (single_gen_target == self.gen_eop_token_id).nonzero().view(-1)
        if split_idxs.shape[0] == 0:
            return None, None

        split_sizes = (
            [split_idxs[0] + 1]
            + [split_idxs[j] - split_idxs[j-1] for j in range(1, split_idxs.shape[0])] 
            + [seq_len - split_idxs[-1] - 1]
        )
        # The first split is most likely the end of a poem, so it doesn't have labels and we ignore it
        poems_tensor_list = torch.split(single_gen_output, split_sizes, dim=0)[1:]
        # size: (n_poems in single_gen_target - 1) x (max_seq_len of poems in single_gen_target) x vocab_sz
        poems_tensor = torch.nn.utils.rnn.pad_sequence(poems_tensor_list, batch_first=True, padding_value=float("-inf"))
        assert poems_tensor.requires_grad

        # Set the pad_token_id as the max logit for the padded positions
        clf_pad_token_id = self.clf_tokenizer.pad_token_id
        for i, poem_len in enumerate(split_sizes[1:]):
            poems_tensor[i, poem_len:, clf_pad_token_id] = 1e6

        labels_by_poem = self.labels_decoder.decode_labels(
            self.gen_tokenizer.decode(single_gen_target[split_sizes[0]:].tolist()), self.file_config
        )

        return poems_tensor, labels_by_poem

    def _replace_special_tokens_for_clf(self, gen_output):
        gen_preds = gen_output.argmax(dim=-1)
        gen_output = gen_output.clone()
        
        if self.gen_bov_token_id is not None:
            # Change the prediction to ' ' where it's a beginning of verse
            bov_idxs = (gen_preds == self.gen_bov_token_id).nonzero().t()
            gen_output[bov_idxs[0], bov_idxs[1], self.clf_whitespace_token_id] = gen_output[bov_idxs[0], bov_idxs[1], self.gen_bov_token_id]
            gen_output[bov_idxs[0], bov_idxs[1], self.gen_bov_token_id] = float('-inf')
        if self.gen_eov_token_id is not None:
            # Change the prediction to '\n' where it's an end of verse
            eov_idxs = (gen_preds == self.gen_eov_token_id).nonzero().t()
            gen_output[eov_idxs[0], eov_idxs[1], self.clf_eol_token_id] = gen_output[eov_idxs[0], eov_idxs[1], self.gen_eov_token_id]
            gen_output[eov_idxs[0], eov_idxs[1], self.gen_eov_token_id] = float('-inf')
        if self.gen_eop_token_id is not None:
            # Change the prediction to '\n' where it's an end of poem
            eop_idxs = (gen_preds == self.gen_eop_token_id).nonzero().t()
            gen_output[eop_idxs[0], eop_idxs[1], self.clf_eol_token_id] = gen_output[eop_idxs[0], eop_idxs[1], self.gen_eop_token_id]
            gen_output[eop_idxs[0], eop_idxs[1], self.gen_eop_token_id] = float('-inf')

        return gen_output
    
    def __call__(self, gen_output, gen_target):
        """
        Calculates the cross entropy loss between the label contained in the header of each poem and the prediction of `self.clf`.
        
        Args:
            gen_output: Output sequence of text generator, of size (batch size, sequence length, vocab size)
            gen_target: Target of text generator, tensor of dtype long, of size (batch size, sequence length). 
                It should be shifted if needed. No shift is done here.   
        """
        # TODO MAYBE (if needed for encoder-decoder): add param clf_target=None: size (bs, n_poems_in_batch) for 
        # more flexibility. It would be used instead of gen_target when it's passed, so the process of gen_target
        # should be skipped in that case.
        bs = gen_output.shape[0]
        clf_emb_w = self.clf.get_input_embeddings().weight
        loss = torch.tensor(0., device=gen_output.device, requires_grad=True)
        gen_output = self._replace_special_tokens_for_clf(gen_output)
        n_poems = 0
        
        for i in range(bs): 
            poems_tensor, labels_by_poem = self._split_by_poem(gen_output[i], gen_target[i])

            if poems_tensor is not None:
                clf_emb_out = F.softmax(poems_tensor, dim=-1) @ clf_emb_w
                assert clf_emb_out.requires_grad
                # size: (n poems in batch[i] - 1) x n classes
                clf_preds = self.clf(input_embeds=clf_emb_out).logits

                non_empty_label_idxs = []

                clf_label_ids = []
                for i, labels_by_cat in enumerate(labels_by_poem):
                    label = labels_by_cat[label_type_to_str(self.cat)]
                    if label == '': continue
                    clf_label_ids.append(self.clf.config.label2id[label])
                    non_empty_label_idxs.append(i)
                
                if len(clf_label_ids) > 0:
                    clf_preds = clf_preds[non_empty_label_idxs]
                    clf_label_ids_t = torch.tensor(clf_label_ids, device=clf_preds.device)
                    loss = loss + self.inner_loss(clf_preds, clf_label_ids_t)
                    n_poems += len(clf_label_ids)
                      
            #else:
            #    There's just 1 poem in item `i` of the batch, most likely without labels, so we ignore it
            #    poems_tensor = output[i]
            
        return loss/n_poems if n_poems > 0 else loss
    # TODO: WATCHOUT, DON'T FORGET TO SHIFT TARGET IN CALLING PLACE!!!!!!!!!!!!
