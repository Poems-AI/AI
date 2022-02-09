# Error-prone scenarios

**WARNING**: The Hugging Face Trainer, and especially its defaults, are designed to prepare a full training run.

As a consequence, if you call `Trainer.train()` multiple times, many undesired things can happen:

**To go on with a previous optimizer state**, you need to pass `resume_from_checkpoint=True` to `Trainer.train`. 
However, if you do, it loads the checkpoint with the biggest index (last number of checkpoint folder name). 
This index is restarted every time you run `trainer.train()` if you don't set `resume_from_checkpoint`. So if you execute:
```
trainer.args.num_train_epochs = 3
trainer.train()
trainer.args.num_train_epochs = 2
trainer.train()
trainer.train(resume_from_checkpoint=True)
```

If, for example, an epoch has 1000 steps, before the third call to `trainer.train()` we'll have:
* checkpoint-1000: generated after 1st epoch of second run
* checkpoint-2000: generated after 2nd epoch of second run
* checkpoint-3000: generated after 3rd epoch of first run

The checkpoint chosen will be checkpoint-3000, despite not being the oldest.

You can force checkpoint-2000 to be used, by passing `resume_from_checkpoint = './checkpoints/checkpoint-2000'`

Moreover, **the default learning rate scheduler is a linear scheduler** that makes the learning rate decrease 
from `TrainingArguments.learning_rate` to 0 after `TrainingArguments.num_train_epochs`. This means that:
1) if you call `trainer.train()` two times using the same `Trainer` instance, the second time will be useless,\
   because the learning rate will be 0 since the beginning.
   
2) The same is going to happen if you create a new Trainer instance for the same model and resume from a previous checkpoint. For instance:

```
model = ...
tr_args = TrainingArguments(num_train_epochs=3)
trainer = Trainer(model=model, args=tr_args, ...)
trainer.train()
tr_args = TrainingArguments(num_train_epochs=10)
trainer = Trainer(model=model, args=tr_args, ...)
trainer.train(resume_from_checkpoint=True)
```

3) If you download from the Hub a model that you already trained for many epochs, to train a bit more, and create a new Trainer but don't set `resume_from_checkpoint`, then, differently from the two previous cases, the learning rate will start from the value that you give, but that 
learning rate is going to be clearly higher than the learning rate you were using in the last epochs of the previous training session; 
in most cases, this will lead to an increase of the loss, at least during some epochs until the learning rate goes down again because of the scheduler. 
For instance:

```
model = ...
tr_args = TrainingArguments(num_train_epochs=30, learning_rate=5e-5),
trainer = Trainer(model=model, args=tr_args, ...)
trainer.train()
# learning rate is close to 0, much lower than 5e-5, at the end
model.push_to_hub('https://huggingface.co/myuser/mymodel')
# End of session

# ...A day after, in a different session
model = AutoModelForCausalLM.from_pretrained('myuser/mymodel')
tr_args = TrainingArguments(num_train_epochs=10, learning_rate=5e-5),
trainer = Trainer(model=model, args=tr_args, ...)
trainer.train()
# The learning_rate is 5e-5 at the beginning, and the loss probably goes up
```


# How to proceed

In conclusion, the lesson to learn from all the above is that you should either:
- a) Since the beginning, set `TrainingArguments.num_train_epochs` to the total number of training epochs you are running.
- b) If you don't know beforehand the number of epochs you are going to train the model for, you can:
  - Use a constant learning rate scheduler. For this, set the `lr_scheduler_type` param of `TrainingArguments` to
   `SchedulerType.CONSTANT` or `SchedulerType.CONSTANT_WITH_WARMUP`. In the latter case, don't 
    forget to set also the parameter `warmup_steps`.
  - Pass your own scheduler to Trainer.\_\_init\_\_. The parameter `optimizers` is a size 2 tuple, with the optimizer being the first element and the scheduler, the second one.
