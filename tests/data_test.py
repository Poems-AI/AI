from functools import partial
import io
import os
import pandas as pd
from pathlib import Path
from poemsai.config import set_config_value
from poemsai.data import (build_labeled_dfs_from_splits, label_type_to_str, LabeledPoem, 
                          LabeledPoemsIOWriter, LabelsDecoderExplained, LabelsDecoderKeyValue,
                          LabelsWriterExplained,  LabelsWriterKeyValue, LabelsWriterKeyValueMultiverse, 
                          LabelsWriterStd, LabelsType, merge_poems, PoemsFileConfig, PoemsIOWriter, 
                          VerseGrouping)
import tempfile


def test_merge_poems():
    fake_poems_reader = [
        ["  This is a     one verse poem    "],
        ["This is a", "two verse poem   "],
        ["This is a", "five verse", "poem, yes", " believe it", " or not   "],
    ]
    simple_conf = PoemsFileConfig(remove_multispaces = True, 
                                  beginning_of_verse_token = '',
                                  end_of_verse_token = '\\n', 
                                  end_of_poem_token = '',
                                  n_prev_verses_terminations = 0, 
                                  verse_grouping = VerseGrouping.OneVerseBySequence)
    full_conf = PoemsFileConfig(remove_multispaces = True, 
                                beginning_of_verse_token = '<BOS>',
                                end_of_verse_token = '<EOS>', 
                                end_of_poem_token = '<EOP>',
                                n_prev_verses_terminations = 2,
                                verse_grouping = VerseGrouping.OneVerseBySequence)
    full_conf_poem_line = PoemsFileConfig(remove_multispaces = True, 
                                          beginning_of_verse_token = '<BOS>',
                                          end_of_verse_token = '<EOS>', 
                                          end_of_poem_token = '<EOP>',
                                          n_prev_verses_terminations = 2,
                                          verse_grouping = VerseGrouping.OnePoemBySequence)
    
    outputs = []
    for conf in (simple_conf, full_conf, full_conf_poem_line):
        with io.StringIO() as f:
            writer = PoemsIOWriter(f, conf)
            merge_poems(fake_poems_reader, writer)
            outputs.append(f.getvalue())

    simple_conf_out, full_conf_out, full_conf_poem_line_out = outputs
    lb = os.linesep
    assert simple_conf_out == (f"This is a one verse poem \\n{lb}" + 
                               f"This is a \\n{lb}two verse poem \\n{lb}" +
                               f"This is a \\n{lb}five verse \\n{lb}poem, yes \\n{lb}believe it \\n{lb}or not \\n{lb}")
    assert full_conf_out == (f"<BOS>This is a one verse poem <EOS><EOP>{lb}" + 
                             f"<BOS>This is a <EOS>{lb}a<BOS>two verse poem <EOS><EOP>{lb}" +
                             f"<BOS>This is a <EOS>{lb}a<BOS>five verse <EOS>{lb}a verse<BOS>poem, yes <EOS>{lb}verse yes<BOS>believe it <EOS>{lb}yes it<BOS>or not <EOS><EOP>{lb}")
    assert full_conf_poem_line_out == (f"<BOS>This is a one verse poem <EOS><EOP>{lb}" + 
                                       f"<BOS>This is a <EOS>a<BOS>two verse poem <EOS><EOP>{lb}" +
                                       f"<BOS>This is a <EOS>a<BOS>five verse <EOS>a verse<BOS>poem, yes <EOS>verse yes<BOS>believe it <EOS>yes it<BOS>or not <EOS><EOP>{lb}")


def _write_poems_to_str(poems, labels_writer, file_conf) -> str:
    with io.StringIO() as out_stream:
        writer = LabeledPoemsIOWriter(out_stream, file_conf, labels_writer=labels_writer)
        for poem in poems:
            writer.write_poem(poem)
        return out_stream.getvalue()


def test_labeled_poems_file_writer():
    file_conf = PoemsFileConfig(remove_multispaces = True, 
                                beginning_of_verse_token = '<BOS>',
                                end_of_verse_token = '<EOS>', 
                                end_of_poem_token = '<EOP>',
                                n_prev_verses_terminations = 0,
                                verse_grouping = VerseGrouping.OnePoemBySequence)
    forms_cat = label_type_to_str(LabelsType.Forms)
    topics_cat = label_type_to_str(LabelsType.Topics)
    poems = [
        LabeledPoem(['Firstverse', 'Second verse'], {forms_cat: 'label1'}),
        LabeledPoem(['First verse', 'Secondverse'], {topics_cat: 'label2'}),
        LabeledPoem(['1st verse', '2nd verse'], {forms_cat: 'label1', topics_cat: 'label2'}),
        LabeledPoem(['1st verse', '2nd verse'], {forms_cat: '', topics_cat: 'label2'}),
        LabeledPoem(['1st verse', '2nd verse'], {forms_cat: 'label1', topics_cat: ''}),
    ]

    result_std = _write_poems_to_str(poems, LabelsWriterStd(), file_conf)
    result_key_value = _write_poems_to_str(poems, LabelsWriterKeyValue(), file_conf)
    result_key_value_multiverse = _write_poems_to_str(poems, LabelsWriterKeyValueMultiverse(), file_conf)
    result_explained = _write_poems_to_str(poems, LabelsWriterExplained(), file_conf)
    result_explained_omit_empty = _write_poems_to_str(
        poems, 
        LabelsWriterExplained(omit_empty=True), 
        file_conf
    )

    lb = os.linesep   
    expected_result_std = (
        "<BOS>label1 <EOS><BOS>Firstverse <EOS><BOS>Second verse <EOS><EOP>" + lb
        + "<BOS>label2 <EOS><BOS>First verse <EOS><BOS>Secondverse <EOS><EOP>" + lb
        + "<BOS>label1 <EOS><BOS>label2 <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + "<BOS>? <EOS><BOS>label2 <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + "<BOS>label1 <EOS><BOS>? <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
    )
    expected_result_key_value = (
        f"<BOS>{forms_cat}: label1 <EOS><BOS>Firstverse <EOS><BOS>Second verse <EOS><EOP>" + lb
        + f"<BOS>{topics_cat}: label2 <EOS><BOS>First verse <EOS><BOS>Secondverse <EOS><EOP>" + lb
        + f"<BOS>{forms_cat}: label1, {topics_cat}: label2 <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + f"<BOS>{forms_cat}: ?, {topics_cat}: label2 <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + f"<BOS>{forms_cat}: label1, {topics_cat}: ? <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
    )
    expected_result_key_value_multiverse = (
        f"<BOS>{forms_cat}: label1 <EOS><BOS>Firstverse <EOS><BOS>Second verse <EOS><EOP>" + lb
        + f"<BOS>{topics_cat}: label2 <EOS><BOS>First verse <EOS><BOS>Secondverse <EOS><EOP>" + lb
        + f"<BOS>{forms_cat}: label1 <EOS><BOS>{topics_cat}: label2 <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + f"<BOS>{forms_cat}: ? <EOS><BOS>{topics_cat}: label2 <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + f"<BOS>{forms_cat}: label1 <EOS><BOS>{topics_cat}: ? <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
    )
    expected_result_explained = (
        "<BOS>This is a poem with label1 form: <EOS><BOS>Firstverse <EOS><BOS>Second verse <EOS><EOP>" + lb
        + "<BOS>This is a poem about label2: <EOS><BOS>First verse <EOS><BOS>Secondverse <EOS><EOP>" + lb
        + "<BOS>This is a poem with label1 form about label2: <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + "<BOS>This is a poem with ? form about label2: <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + "<BOS>This is a poem with label1 form about ?: <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
    )
    expected_result_explained_omit_empty = (
        "<BOS>This is a poem with label1 form: <EOS><BOS>Firstverse <EOS><BOS>Second verse <EOS><EOP>" + lb
        + "<BOS>This is a poem about label2: <EOS><BOS>First verse <EOS><BOS>Secondverse <EOS><EOP>" + lb
        + "<BOS>This is a poem with label1 form about label2: <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + "<BOS>This is a poem about label2: <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
        + "<BOS>This is a poem with label1 form: <EOS><BOS>1st verse <EOS><BOS>2nd verse <EOS><EOP>" + lb
    )

    assert result_std == expected_result_std
    assert result_key_value == expected_result_key_value
    assert result_key_value_multiverse == expected_result_key_value_multiverse
    assert result_explained == expected_result_explained
    assert result_explained_omit_empty == expected_result_explained_omit_empty


def test_labels_decoders():
    file_conf = PoemsFileConfig(remove_multispaces = True, 
                                beginning_of_verse_token = '<BOS>',
                                end_of_verse_token = '<EOS>', 
                                end_of_poem_token = '<EOP>',
                                n_prev_verses_terminations = 0,
                                verse_grouping = VerseGrouping.OnePoemBySequence)
    forms_cat = label_type_to_str(LabelsType.Forms)
    topics_cat = label_type_to_str(LabelsType.Topics)
    decoder_kv = LabelsDecoderKeyValue()
    decoder_exp = LabelsDecoderExplained()

    text_kv = (
        f"<BOS>{forms_cat}: sonnet, {topics_cat}: sky<EOS><EOP>"
        + f"<BOS>{forms_cat}: lyric, {topics_cat}: dreams <EOS><BOS>{forms_cat}: whatever, {topics_cat}: whatever<EOS><EOP>"
        + f"<BOS>{forms_cat}:?, {topics_cat}: happiness <EOS><BOS>{forms_cat}: whatever, {topics_cat}: whatever<EOS><EOP>"
        + f"<BOS>{forms_cat}: ?, {topics_cat}: sea<EOS><BOS>1st verse<EOS><BOS>{forms_cat}: whatever, {topics_cat}: whatever<EOS><EOP>"
        + f"<BOS>{forms_cat}: free-style, {topics_cat}: ?<EOS><BOS>{forms_cat}: whatever, {topics_cat}: whatever<EOS><EOP>"
        + f"<BOS>{forms_cat}: ?, {topics_cat}:? <EOS><BOS>{forms_cat}: whatever, {topics_cat}: whatever<EOS><EOP>"
        + f"<BOS>{forms_cat}: lyric"
    )
    text_kv2 = f"{topics_cat}: love<EOS><BOS>First verse<EOS>"
    text_exp = (
        f"<BOS>This is a poem with sonnet form about sky: <EOS><EOP>"
        + f"<BOS>This is a poem with lyric form about dreams: <EOS><BOS>This is a poem with whatever form about whatever <EOS><EOP>"
        + f"<BOS>This is a poem with ? form about happiness: <EOS><BOS>This a poem with whatever form about whatever: <EOS><EOP>"
        + f"<BOS>This is a poem with  ? form about sea:<EOS><BOS>1st verse<EOS><BOS>This is a poem with whatever form about whatever<EOS><EOP>"
        + f"<BOS>This is a poem with free-style form about ?:<EOS><BOS>This is a poem with whatever form about whatever<EOS><EOP>"
        # '?' could appear joined because the decoder of some tokenizers removes the spaces between a word and the question mark 
        + f"<BOS>This is a poem with? form about?: <EOS><BOS>First verse ...<EOS><EOP>"
        + f"<BOS>This is a poem with lyric"
    )
    text_exp2 = f"about love: <EOS><BOS>First verse<EOS>"

    actual_labels_from_kv = decoder_kv.decode_labels(text_kv, file_conf)
    actual_labels_from_kv2 = decoder_kv.decode_labels(text_kv2, file_conf)    
    actual_labels_from_exp = decoder_exp.decode_labels(text_exp, file_conf)
    actual_labels_from_exp2 = decoder_exp.decode_labels(text_exp2, file_conf)

    expected_labels = [
        {forms_cat: 'sonnet', topics_cat: 'sky'},
        {forms_cat: 'lyric', topics_cat: 'dreams'},
        {forms_cat: '', topics_cat: 'happiness'},
        {forms_cat: '', topics_cat: 'sea'},
        {forms_cat: 'free-style', topics_cat: ''},
        {forms_cat: '', topics_cat: ''},
        {forms_cat: 'lyric', topics_cat: ''},
    ]
    expected_labels2 = [{forms_cat: '', topics_cat: 'love'}]

    assert actual_labels_from_kv == expected_labels
    assert actual_labels_from_kv2 == expected_labels2
    assert actual_labels_from_exp == expected_labels
    assert actual_labels_from_exp2 == expected_labels2


def test_build_labeled_dfs_from_splits():
    poems_content = [f'content of poem {i}' for i in range(7)]

    with tempfile.TemporaryDirectory() as tmpdir:
        set_config_value('KAGGLE_DS_ROOT', tmpdir)
        tmpdir_path = Path(tmpdir)
        os.mkdir(tmpdir_path/'topics')
        os.mkdir(tmpdir_path/'forms')
        os.mkdir(tmpdir_path/'topics/a')
        os.mkdir(tmpdir_path/'topics/b')
        os.mkdir(tmpdir_path/'topics/c')
        os.mkdir(tmpdir_path/'topics/d')
        os.mkdir(tmpdir_path/'forms/a')
        poems_locations = [
            tmpdir_path/'topics/a/poem0.txt',
            tmpdir_path/'topics/a/poem1.txt',
            tmpdir_path/'topics/b/poem2.txt',
            tmpdir_path/'topics/c/poem3.txt',
            tmpdir_path/'topics/c/poem4.txt',
            tmpdir_path/'topics/d/poem5.txt',
            tmpdir_path/'forms/a/poem6.txt',
        ]
        for poem_content, poem_location in zip(poems_content, poems_locations):
            with open(poem_location, 'w') as f:
                f.write(poem_content)
        poems_locations.append(tmpdir_path/'topics/d/poem7doesntexist.txt')

        splits_df = pd.DataFrame({
            'Poem name': ['Poem 0', 'Poem 1', 'Poem 2', 'Poem 3', 'Poem 4', 'Poem 5', 'Poem 6', 'Poem 7'],
            'Location': [
                '[KAGGLE_DS_ROOT]/topics/a/poem0.txt',
                '[KAGGLE_DS_ROOT]/topics/a/poem1.txt',
                '[KAGGLE_DS_ROOT]/topics/b/poem2.txt',
                '[KAGGLE_DS_ROOT]/topics/c/poem3.txt',
                '[KAGGLE_DS_ROOT]/topics/c/poem4.txt',
                '[KAGGLE_DS_ROOT]/topics/d/poem5.txt',
                '[KAGGLE_DS_ROOT]/forms/a/poem6.txt',
                '[KAGGLE_DS_ROOT]/topics/d/poem7doesntexist.txt',
            ],
            'Split': ['Train', 'Validation', 'Validation', 'Validation', 'Train', 'Train', 'Train', 'Train']
        })
        expected_train_df = pd.DataFrame({
            'text': [
                'content of poem 0',
                'content of poem 4',
                'content of poem 5',
            ],
            'labels': ['a', 'c', 'd']
        })
        expected_valid_df = pd.DataFrame({
            'text': [
                'content of poem 1',
                'content of poem 2',
                'content of poem 3',
            ],
            'labels': ['a', 'b', 'c']
        })

        train_df, valid_df = build_labeled_dfs_from_splits(splits_df, LabelsType.Topics)
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        assert expected_train_df.equals(train_df)
        assert expected_valid_df.equals(valid_df)
