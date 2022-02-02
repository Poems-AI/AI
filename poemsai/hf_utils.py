__all__ = ['model_to_url']


def model_to_url(model_name, user, pwd=None):
    login_str = '' if pwd is None else f'{user}:{pwd}@'
    return f"https://{login_str}huggingface.co/{user}/{model_name}"
