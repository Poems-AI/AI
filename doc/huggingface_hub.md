# Communication with Hugging Face Hub

If you just want to share a model with the community, it's enough to execute:

```
happy_gen.tokenizer.push_to_hub(model_url)
happy_gen.model.push_to_hub(model_url)
```

(remember that `model_url` should have the format "https://huggingface.co/{user}/{model_name}")

However, if you want to save a model in the middle of a training run, in order to be able to continue later, during a different session, you should save the optimizer state too. The easiest way to do it is
to call our method `nb_utils.commit_checkpoint_to_hf_hub`; of course, this will also push the tokenizer and the model. For instance:

```
commit_checkpoint_to_hub('gpt2-poems.en', 'davidlf', get_last_checkpoint('./checkpoints'),
                          message='Save checkpoint at 10 epochs', pwd='[your_huggingface_password]')
```

To resume a training run from a checkpoint you stored in the Hub:

1) Clone the repo:

```
custom_model_name = 'gpt2-poems.en'
HF_USER = 'davidlf'
hf_pwd = 'your_hf_password'
!git clone {model_to_url(custom_model_name, HF_USER, hf_pwd)}
```

2) Create the Trainer as always

3) Call trainer.train() passing the path of the repo you just cloned:

```
trainer.train(resume_from_checkpoint=custom_model_name)
```
