from poemsai.hf_utils import model_to_url
import os
from pathlib import Path

__all__ = ['download_checkpoint_from_hf_hub', 'commit_checkpoint_to_hf_hub']


def download_checkpoint_from_hf_hub(model_name, user, pwd):
    model_url = model_to_url(model_name, user, pwd)
    get_ipython().system("git clone {model_url}")


def commit_checkpoint_to_hf_hub(model_name, user, checkpoint_path, message='Update model',
                                base_repo_path='./', pwd=None):
    shell = get_ipython()
    cwd = os.getcwd()
    if base_repo_path[-1] != '/': base_repo_path += '/'
    repo_path = Path(base_repo_path + model_name)
    if not repo_path.exists():
        repo_path.mkdir()
        model_url = model_to_url(model_name, user, pwd=pwd)
        shell.system("!git clone {model_url} $repo_path")
    shell.system("cp {checkpoint_path}/* $repo_path")
    shell.run_line_magic('cd', str(repo_path))
    shell.system("git add .")
    shell.system("git status")
    shell.system('git commit -m "{message}"')
    shell.system("git push")
    shell.run_line_magic('cd', cwd)
