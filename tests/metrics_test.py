from dataclasses import dataclass
import math
from poemsai.data import LabelsType, LabelsDecoderExplained, PoemsFileConfig
from poemsai.metrics import compute_lm_metrics, ConditionalGenLoss, MetadataLessLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def test_compute_lm_metrics():
    # (bs, seq length, vocab size) = (2, 3, 4)
    logits = torch.Tensor([
        [[0, 0.1, 0.2, 1.3], [0.1, 0, 0, 0], [-0.5, -0.2, -0.6, -0.9], [0.1, 0.2, 0.3, 0.4]],
        [[-0.1, 0.2, 0.3, 0.25], [1., 1.1, 0.7, 0.6], [0.5, 0.5, 0.5, 0.55], [0.1, 0.2, 0.3, 0.4]]
    ])
    preds = torch.Tensor([[3, 0, 1, 3], [2, 1, 3, 3]])
    labels_eq = torch.Tensor([[0, 3, 0, 1], [0, 2, 1, 3]])
    labels_dif = torch.Tensor([[0, 2, 1, 0], [0, 3, 0, 2]])
    labels_half = torch.Tensor([[0, 2, 0, 3], [0, 2, 2, 3]])
    metric = 'accuracy'
    
    assert math.isclose(compute_lm_metrics((logits, labels_eq))[metric], 1)
    assert math.isclose(compute_lm_metrics((logits, labels_dif))[metric], 0)
    assert math.isclose(compute_lm_metrics((logits, labels_half))[metric], 0.5)
    assert math.isclose(compute_lm_metrics((preds, labels_eq), expect_preds=True)[metric], 1)
    assert math.isclose(compute_lm_metrics((preds, labels_dif), expect_preds=True)[metric], 0)
    assert math.isclose(compute_lm_metrics((preds, labels_half), expect_preds=True)[metric], 0.5)


def test_metadataless_loss():
    class FakeLoss:
        def __init__(self):
            self.call_args = []
            
        def __call__(self, preds, target):
            self.call_args.append((preds, target))
            
    BOV_ID = 97
    EOV_ID = 98
    EOP_ID = 99
    IGN_IDX = -1
    loss_bov_tag = MetadataLessLoss(FakeLoss(), begin_verse_id=BOV_ID, end_verse_id=None, 
                                    end_poem_id=None, ignore_index=IGN_IDX)
    loss_end_tags = MetadataLessLoss(FakeLoss(), begin_verse_id=None, end_verse_id=EOV_ID, 
                                     end_poem_id=EOP_ID, ignore_index=IGN_IDX)
    loss_all_tags = MetadataLessLoss(FakeLoss(), begin_verse_id=BOV_ID, end_verse_id=EOV_ID, 
                                     end_poem_id=EOP_ID, ignore_index=IGN_IDX)
    loss_end_tags_ign_2_verses = MetadataLessLoss(FakeLoss(), begin_verse_id=None, end_verse_id=EOV_ID, 
                                                  end_poem_id=EOP_ID, ignore_index=IGN_IDX, 
                                                  n_initial_verses_to_ignore=2)
    labels = torch.Tensor([
        [BOV_ID, 1, 2, EOV_ID, BOV_ID, 3, EOV_ID, EOP_ID, BOV_ID, 4, 5, EOV_ID, BOV_ID, 6, EOV_ID, EOP_ID, BOV_ID, 7, 8, 9, EOV_ID],
        [BOV_ID, 1, 2, EOV_ID, 24, 35, BOV_ID, 3, EOV_ID, 28, EOP_ID, 23, 34, 56, BOV_ID, 4, EOV_ID, 77, 88, BOV_ID, 5],
        [30, 40, BOV_ID, 1, EOV_ID, 33, BOV_ID, 2, 3, EOV_ID, 44, 55, EOP_ID, 45, BOV_ID, 4, EOV_ID, EOP_ID, 88, 56, 62],
        [1, 2, EOV_ID, 3, EOP_ID, 23, 43, BOV_ID, 4, 5, 6, EOV_ID, BOV_ID, 7, 8, EOV_ID, 12, 15, 14, 18, 19],
    ])
    preds = torch.rand(*labels.shape, 5)
    loss_bov_tag(preds, labels)
    loss_end_tags(preds, labels)
    loss_all_tags(preds, labels)
    loss_end_tags_ign_2_verses(preds, labels)
    
    expected_loss_bov_target = labels
    expected_loss_end_tags_target = torch.Tensor([
        [BOV_ID, 1, 2, EOV_ID, BOV_ID, 3, EOV_ID, IGN_IDX, BOV_ID, 4, 5, EOV_ID, BOV_ID, 6, EOV_ID, IGN_IDX, BOV_ID, 7, 8, 9, EOV_ID],
        [BOV_ID, 1, 2, EOV_ID, 24, 35, BOV_ID, 3, EOV_ID, IGN_IDX, IGN_IDX, 23, 34, 56, BOV_ID, 4, EOV_ID, 77, 88, BOV_ID, 5],
        [30, 40, BOV_ID, 1, EOV_ID, 33, BOV_ID, 2, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, 45, BOV_ID, 4, EOV_ID, IGN_IDX, 88, 56, 62],
        # In this case, the end of this sequence is ambiguous, you can't know if the final tokens are between EOV and EOP 
        # or between two EOV
        [1, 2, EOV_ID, IGN_IDX, IGN_IDX, 23, 43, BOV_ID, 4, 5, 6, EOV_ID, BOV_ID, 7, 8, EOV_ID, 12, 15, 14, 18, 19]
    ])
    expected_loss_all_tags_target = torch.Tensor([
        [IGN_IDX, 1, 2, EOV_ID, IGN_IDX, 3, EOV_ID, IGN_IDX, IGN_IDX, 4, 5, EOV_ID, IGN_IDX, 6, EOV_ID, IGN_IDX, IGN_IDX, 7, 8, 9, EOV_ID],
        [IGN_IDX, 1, 2, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 4, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, 5],
        [IGN_IDX, IGN_IDX, IGN_IDX, 1, EOV_ID, IGN_IDX, IGN_IDX, 2, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 4, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX],
        [1, 2, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 4, 5, 6, EOV_ID, IGN_IDX, 7, 8, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX]
    ])
    expected_loss_end_tags_ign_2_verses_target = torch.Tensor([
        [BOV_ID, 1, 2, EOV_ID, BOV_ID, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX],
        [BOV_ID, 1, 2, EOV_ID, 24, 35, BOV_ID, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX],
        [30, 40, BOV_ID, 1, EOV_ID, 33, BOV_ID, 2, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX],
        # In this case, the end of this sequence is ambiguous, you can't know if the final tokens are between EOV and EOP 
        # or between two EOV
        [1, 2, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 12, 15, 14, 18, 19],
    ])    
    
    assert loss_bov_tag.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_bov_tag.inner_loss.call_args[0][1] == expected_loss_bov_target)
    assert loss_end_tags.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_end_tags.inner_loss.call_args[0][1] == expected_loss_end_tags_target)
    assert loss_all_tags.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_all_tags.inner_loss.call_args[0][1] == expected_loss_all_tags_target)
    assert loss_end_tags_ign_2_verses.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_end_tags_ign_2_verses.inner_loss.call_args[0][1] == expected_loss_end_tags_ign_2_verses_target)


class FakeTokenizer:
    def __init__(self, eov_token:str=None, eop_token:str=None, pad_token:str=None):
        self.eop_token = eop_token
        # It could be an array, we use a dict for readability of ids
        self.id2token = {
            0: 'sea',
            1: 'darkness',
            2: 'freedom',
            3: 'love',
            4: 'dreams',
            5: 'nightmare',
            6: 'sonnet',
            7: 'lyric',
            8: 'verse',
            9: 'anaphora',
            10: 'This',
            11: 'is',
            12: 'a',
            13: 'poem',
            14: 'about',
            15: 'with',
            16: 'form',
            17: ':',
            18: '?',
            19: 'bla',
            20: 'ble',
            21: 'bli',
            22: 'blo',
            23: 'blu',
            24: '\n',
            25: '',
        }
        if pad_token is not None:
            self.pad_token_id = len(self.id2token)
            self.id2token[self.pad_token_id] = pad_token
        else:
            self.pad_token_id = None
        if eov_token is not None:
            # 26: eov_token,
            self.id2token[len(self.id2token)] = eov_token
        if eop_token is not None:
            # 27-: eop_token
            self.id2token[len(self.id2token)] = eop_token
        self.token2id = {tok: id for id, tok in self.id2token.items()}
        
    def __len__(self): return len(self.id2token)
        
    def encode(self, text:str):
        return [self.token2id[tok] for tok in text.split(' ')]
        
    def decode(self, token_ids:List[int]):
        text = ' '.join([self.id2token[tok_id] for tok_id in token_ids])
        text = text.replace(' :', ':')
        return text
    

@dataclass
class FakeClfOut:
    logits:torch.Tensor
    

class FakeClfConfig:
    def __init__(self, label2id:dict):
        self.label2id = label2id

    
class FakeClf(nn.Module):
    "Predicts for every sequence 'seq', the token with index ((seq[0] + 1) % vocab_sz) % n_classes"
    def __init__(self, vocab_sz:int, labels:List[str]):#, hid_sz:int):
        super().__init__()
        self.n_classes = len(labels)
        label2id = dict(zip(labels, range(self.n_classes)))
        self.config = FakeClfConfig(label2id)
        hid_sz = vocab_sz
        self.embedding = nn.Embedding(vocab_sz, hid_sz)
        with torch.no_grad():
            # eye shifted one position to the right, so:
            # for idx 0, returns                0 1 0 0 ...
            # for idx 1, returns                0 0 1 0 ...
            # ...
            # for idx `vocab_sz - 1`, returns   1 0 0 0 ...
            self.embedding.weight[:] = torch.roll(torch.eye(vocab_sz), 1, 1)
        self.linear = nn.Linear(vocab_sz, self.n_classes)
        with torch.no_grad():
            self.linear.bias[:] = 0
            # This linear layer is equivalent to a mod of the argmax if the input is one-hot
            lin_w = (torch.eye(self.n_classes).repeat(math.ceil(vocab_sz / self.n_classes), 1)[:vocab_sz]).t()
            lin_w.requires_grad = True
            self.linear.weight = nn.Parameter(lin_w)
            assert self.linear.weight.requires_grad
        
    def get_input_embeddings(self):
        return self.embedding
    
    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        n = min(old_num_tokens, new_num_tokens)
        with torch.no_grad():
            new_embeddings.weight[:n, :] = old_embeddings.weight[:n, :]
        self.embedding = new_embeddings
    
    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embedding(input_ids)
        out = self.linear(inputs_embeds[:, 0])
        return FakeClfOut(logits=out)


def test_conditional_loss():    
    EOV = '<EOV>'
    EOP = '<EOP>'
    PAD = '<PAD>'
    gen_tokenizer = FakeTokenizer(eov_token=EOV, eop_token=EOP)
    
    # There are extra spaces to make encoding easier for FakeTokenizer
    target1 = (
        f"This is a poem with sonnet form about dreams : {EOV} bla about blu blu {EOV} bla {EOV} {EOP} "
        + f"This is a poem with verse form about freedom : {EOV} ble about blu blu {EOV} ble {EOV} {EOP} "
        + f"This is"
    )
    output1 = (
        f"a a a a a a a a a a a bli bli bli bli bli bli bli bli "
        + f"bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla "
        + f"ble ble"
    )
    target2 = (
        f"{EOV} {EOP} This is a poem with lyric form about sea : {EOV} blu {EOV} {EOP} " 
        + f"This is a poem with lyric form about sea : {EOV} {EOP} "
        + f"This is a poem with sonnet form about ? : {EOV} {EOP}"
    )
    output2 = (
        f": : ble ble ble ble ble ble ble ble ble ble ble ble ble ble " 
        + f"bli bli bli bli bli bli bli bli bli bli {EOV} {EOP} "
        + f"{EOV} {EOP} blo blo blo blo blo blo blo blo blo blo"
    )
    target3 = (
        f"This is a poem with anaphora form about ? : {EOV} bla {EOV} {EOP} "
        + f"This a poem with anaphora form about ? : {EOV} bla bla bla bla bla : : : : : : : : : : :"
    )
    output3 = (
        f"a a a a a a a a a a {EOV} {EOP} {EOV} a "
        + f"This a poem with anaphora form about ? : {EOV} bla bla bla bla bla : : : : : : : : : : :"
    )
    target4 = (
        f"This is a poem with anaphora form about freedom : {EOV} bla {EOV} {EOP} "
        + f"This a poem with ? form about sea : {EOV} bla bla bla bla bla : : : : : : : : : : :"
    )
    output4 = (
        f"a a a a a a a a a a {EOV} {EOP} {EOV} a "
        + f"This a poem with anaphora form about ? : {EOV} bla bla bla bla bla : : : : : : : : : : :"
    )
    gen_target = torch.tensor([
        gen_tokenizer.encode(target1),
        gen_tokenizer.encode(target2),
        gen_tokenizer.encode(target3),
        gen_tokenizer.encode(target4),
    ], dtype=torch.long)
    gen_preds = torch.tensor([
        gen_tokenizer.encode(output1),
        gen_tokenizer.encode(output2),
        gen_tokenizer.encode(output3),
        gen_tokenizer.encode(output4),
    ])
    gen_output = (F.one_hot(gen_preds) * 1e6 - 1e6/2).float()
    gen_output.requires_grad = True
    clf_tokenizer = FakeTokenizer(pad_token=PAD)
    file_config = PoemsFileConfig(
        #beginning_of_verse_token = '',
        end_of_verse_token = EOV,
        end_of_poem_token = EOP,
    )
    clf = FakeClf(len(clf_tokenizer), labels=["sonnet", "lyric", "verse", "anaphora"])
    gen_bov_token_id=None
    gen_eov_token_id = gen_tokenizer.encode(EOV)[0]
    gen_eop_token_id = gen_tokenizer.encode(EOP)[0]
    
    cond_loss = ConditionalGenLoss(
        clf, clf_tokenizer, gen_tokenizer, LabelsType.Forms, LabelsDecoderExplained(), file_config, 
        gen_eop_token_id, gen_bov_token_id, gen_eov_token_id,
    )
    
    # First words of gen_output:    bla, ble, bli, <EOV>, This
    # Idxs in vocab                  19   20   21   26     10
    # Replacements                                  24
    #  + 1 % vocabsz                  0    1    2    1      3
    expected_clf_preds = torch.tensor([0, 1, 2, 1, 3])
    expected_clf_logits = F.one_hot(expected_clf_preds).float()
    # verse, lyric, lyric, sonnet, anaphora
    cond_loss_target = torch.tensor([2, 1, 1, 0, 3], dtype=torch.long)
    expected_loss = F.cross_entropy(expected_clf_logits, cond_loss_target)
    
    actual_loss = cond_loss(gen_output, gen_target)
    
    assert torch.isclose(expected_loss, actual_loss)
