
All the training configurations have been tested with the following specifications:
- Input files have end of verse (\n) and end of poem (<|endoftext|>) tags
- Train with an initial lr of 5e-5 and a linear scheduler (HuggingFace default) that would make the lr=0 after 50 epochs 
- Perform evaluation once per epoch

The baseline is a GPT2 language model with Hugging Face defaults:
- dropout=0.1
- hidden size = 768
- cross entropy loss

# Conditional metrics explained

The metrics for conditional generation are calculated with the following procedure:
- Form/topic conditional loss/accuracy A:
  1. Generate text using just the labels as prompt.
  2. Delete the labels from the generated text.
  3. Feed the generated text to the corresponding classifier to evaluate the metrics.
- Form/topic conditional loss/accuracy B:
  1. Generate text using as prompt an initial substring of each sequence in the validation set, with its label prepended. We are limiting the prompt to the minimum between 100 characters and 1/4 of the length of the text.
  2. Delete the labels from the generated text.
  3. Feed the generated text to the corresponding classifier to evaluate the metrics.

In both cases, the labels are prepended with the same format used for the training dataset of the evaluated model.


# Results

The baseline model has been trained without any kind of conditioning, so it was to the be expected that its conditional metrics were poor.

Moreover, looking at the baseline results, it's obvious that prepending the topic of a poem to the prompt works much better as a form of "zero-shot conditioning" than prepending the form (structure) of the poem.

If a model was perfectly conditioned, we should expect it to score for the conditional metrics exactly like the corresponding classifier scores for the validation set. So these could serve as theoretical upper bounds:
- Poems classifier by topic: 2.3173 validation loss, 0.5664 validation accuracy
- Poems classifier by form: 2.6072 validation loss, 0.4435 validation accuracy


| Configuration tested             | Best metadata-less val. loss | Epoch of best | Training loss @epoch of best | Form cond. loss A | Form cond. acc. A | Topic cond. loss A | Topic cond. acc. A | Form cond. loss B | Form cond. acc. B | Topic cond. loss B | Topic cond. acc. B |
| -------------------------------- | ---------------------------- | ------------- | ---------------------------- | ------------------ | ---------------- | ------------- | ----------------- | ----------------- | ----------------- | ------------------ | ------------------ |
| Baseline                         | 3.4298                       | 4             |  3.1653                      | 6.6016             | 0.0354           | 4.9290          | 0.1458            | 4.6616            | 0.1858            | 3.3372             | 0.4096             |
| Trained with label prepended(*1) | 3.4034 | 4 | 3.1323  | 6.5839 | 0.0443 | 5.2700 | 0.0972 | 4.4535 | 0.2029 | 3.2910 | 0.4110 |
| Trained with one verse for each label, '?' if not available | 3.3488 | 3 | 3.2262 | 7.0400 | 0.0167 | 3.2677 | 0.4226 | 4.5748 | 0.1923 | 3.0476 | 0.4491 |
| Trained with labels inserted as a single additional verse at the beginning of the poem, with key: value format, '?' if not available | 3.3441 | 3 | 3.1723 | 6.6939 | 0.0324 | 2.5142 | 0.5456 | 4.5728 | 0.1858 | 2.7020 | 0.5069 |
| Trained with a verse inserted for each label at the beginning of the poem, with key: value format, '?' if not available | 3.3454 | 4 | 3.0966 | 6.6060 | 0.0324 | 3.1880 |  0.4276 | 4.5001 | 0.1989 | 2.9276 | 0.4720 |
| Trained with a description of labels verse at the beginning of the poem, '?' for labels not available | 3.3396 | 4 | 3.0975 | 6.4872 | 0.0462 | 3.0526 | 0.4554 | 4.4573 | 0.1849 | 2.8412 | 0.4794 |
| Trained with a description of labels verse at the beginning of the poem, no description for categories not available | 3.3486 | 3 | 3.1779 | 6.3701 | 0.0610 | 2.8156 | 0.4653 | 4.4168 | 0.2021 | 2.7301 | 0.5065 |
| <span style='margin-left: 1em'>+ conditional loss (w=1)</span> | 3.4068  | 5 | 3.1848(*2) | 4.7267 | 0.2153 | 0.7637 | 0.8829 | 3.7641 | 0.3011 | 2.4828 | 0.5597 |
| <span style='margin-left: 1em'>+ conditional loss (w=0.5)</span> | 3.3951 | 5 | 3.1447(*2) | 4.2007 | 0.2861 | 1.0246 | 0.8214 | 3.4086 | 0.3576 | 2.2528 | 0.5890 |
| Trained with a description of labels verse at the beginning of the poem, missing labels filled by classifiers | 3.3355 | 4 | 3.0313 | 6.3723 | 0.0325 | 2.5009 | 0.5268 | 4.3886 | 0.1980 | 2.8035 | 0.4963 |


Legend:
- *1: some poems have only topic as label and others have only form as label, so some examples have a topic in the first position and other examples have the form in the first position of the sequence.
- *2: the training loss doesn't include the conditional loss to be comparable with the rest of the configurations.

Several models score surprisingly well for the Topic conditional loss/accuracy A. After looking at the generated text in those cases, most likely it's because just repeating the topic inside the poem gives valuable information to the classifier.
