import torch
import warnings


__all__ = ['add_special_token']


def add_special_token(token:str, tokenizer, model, copy_from:str=''):
    if token in tokenizer.all_special_tokens:
        warn_msg = f'Token {token} is already a special token'
        if copy_from != '': warn_msg += f', copy from {copy_from} not applied'
        warnings.warn(warn_msg)
        return tokenizer.convert_tokens_to_ids(token)
    
    n_added_tokens = tokenizer.add_special_tokens({
        'additional_special_tokens': tokenizer.additional_special_tokens + [token]
    })
    assert n_added_tokens == 1, (
        f'Unable to add {token}. It already exists but not as a special token. '
        + 'You should use a complex pattern for special tokens to avoid this situation.'
    )
    model.resize_token_embeddings(len(tokenizer))
    
    if copy_from != '':
        emb = model.get_input_embeddings()
        orig_token_id = tokenizer.convert_tokens_to_ids(copy_from)
        with torch.no_grad():
            emb.weight[-1] = emb.weight[orig_token_id]
            
    return len(tokenizer) - 1
