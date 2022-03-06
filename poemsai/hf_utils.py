__all__ = ['get_model_id', 'model_to_url']


def get_model_id(checkpoint_name:str, user:str):
    return f"{user}/{checkpoint_name}"


def model_to_url(checkpoint_name:str, user:str, pwd:str=None):
    login_str = '' if pwd is None else f'{user}:{pwd}@'
    model_id = get_model_id(checkpoint_name, user)
    return f"https://{login_str}huggingface.co/{model_id}"
