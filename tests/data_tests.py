import io
import os
import pandas as pd
from pathlib import Path
from poemsai.config import set_config_value
from poemsai.data import (build_labeled_dfs_from_splits, LabeledPoem, LabeledPoemsFileWriter, LabelsType,
                          merge_poems, PoemsFileConfig, PoemsFileWriter, VerseGrouping)
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
            writer = PoemsFileWriter(f, conf)
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


def test_labeled_poems_file_writer():
    file_conf = PoemsFileConfig(remove_multispaces = True, 
                                beginning_of_verse_token = '<BOS>',
                                end_of_verse_token = '<EOS>', 
                                end_of_poem_token = '<EOP>',
                                n_prev_verses_terminations = 0,
                                verse_grouping = VerseGrouping.OnePoemBySequence)
    poems = [
        LabeledPoem(['Firstverse', 'Second verse'], 'label1'),
        LabeledPoem(['First verse', 'Secondverse'], 'label2'),
    ]
    with io.StringIO() as out_stream:
        writer = LabeledPoemsFileWriter(out_stream, file_conf)
        for poem in poems:
            writer.write_poem(poem)
        result = out_stream.getvalue()

    lb = os.linesep   
    expected_result = ("<BOS>label1 <EOS><BOS>Firstverse <EOS><BOS>Second verse <EOS><EOP>" + lb
                       + "<BOS>label2 <EOS><BOS>First verse <EOS><BOS>Secondverse <EOS><EOP>" + lb)
    assert result == expected_result


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
