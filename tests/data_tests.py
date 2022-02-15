import io
import os
from poemsai.data import (LabeledPoem, LabeledPoemsFileWriter, merge_poems, PoemsFileConfig, PoemsFileWriter, 
                          VerseGrouping)


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
