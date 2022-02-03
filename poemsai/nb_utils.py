import os
from pathlib import Path

__all__ = ['download_checkpoint_from_hf_hub', 'commit_checkpoint_to_hf_hub']


def download_checkpoint_from_hf_hub(model_name, user, pwd):
    model_url = model_to_url(custom_model_name, user, pwd)
    get_ipython().system("git clone {model_url}")


# def commit_checkpoint_to_hf_hub(model_name, user, checkpoint_path, message='Update model',
#                                 base_repo_path='./', pwd=None):
#     cwd = os.getcwd()
#     if base_repo_path[-1] != '/': base_repo_path += '/'
#     repo_path = Path(base_repo_path + model_name)
#     if not repo_path.exists():
#         repo_path.mkdir()
#         !git clone {model_to_url(model_name, user, pwd=pwd)} $repo_path
#     !cp {checkpoint_path}/* $repo_path
#     %cd $repo_path
#     !git add .
#     !git status
#     !git commit -m "{message}"
#     !git push
#     %cd $cwd
