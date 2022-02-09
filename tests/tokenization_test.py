from poemsai.tokenization import add_special_token
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_add_special_token():
    token = '[<bla>]'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    copy_from = tokenizer.all_special_tokens[0]
    orig_token_id = tokenizer(copy_from)['input_ids'][0]
    emb = model.get_input_embeddings()
    orig_emb_rows = emb.weight.shape[0]
    
    add_special_token(token, tokenizer, model, copy_from=copy_from)
    new_emb = model.get_input_embeddings()
    
    assert token in tokenizer.additional_special_tokens
    assert torch.allclose(new_emb.weight[-1], new_emb.weight[orig_token_id])
    assert new_emb.weight.shape[0] == orig_emb_rows + 1, f'{new_emb.weight.shape[0]}, {orig_emb_rows + 1}'
