from abc import ABC, abstractmethod
from datasets import Dataset
from dataclasses import asdict, dataclass
from enum import auto, Enum
import io
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from poemsai.config import get_config_value
from poemsai.hf_utils import get_model_id
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from typing import Callable, Dict, List, Optional, Tuple, Union


__all__ = [
    'Lang', 'lang_to_str', 'DataSource', 'get_ds_path', 'get_text_files', 'get_file_lines',
    'get_ds_root_placeholder', 'get_data_sources', 'SplitterFactory', 'data_splits_to_df', 
    'PoemsDfReader', 'PoemsFileReader', 'ComposedPoemsReader', 'ReaderFactory', 'PoemsFileConfig', 
    'PoemsIOWriter', 'merge_poems', 'DfCache', 'LabelsType', 'label_type_to_str', 'LabeledPoem', 
    'LabeledPoemsSplitsDfContentReader', 'LabeledPoemsSplitsDfReader', 'LabeledPoemsIOWriter', 
    'BaseLabelsWriter', 'LabelsWriterStd', 'LabelsWriterKeyValue', 'LabelsWriterKeyValueMultiverse', 
    'LabelsWriterExplained', 'LabelsDecoderKeyValue', 'LabelsDecoderExplained', 'build_labeled_dfs_from_splits', 
    'build_datasets_for_encoder_decoder', 'LabelsEstimator',
]


class Lang(Enum):
    English = "en"
    Spanish = "es"


def lang_to_str(lang:Lang):
    return lang.value


class DataSource(Enum):
    Marcos = auto()
    Kaggle = auto()


POEM_NAME_DF_COL = 'Poem name'
POEM_LOCATION_DF_COL = 'Location'


DS_ROOTS_CONFIG_KEYS = {
    DataSource.Marcos: 'OWN_DS_ROOT',
    DataSource.Kaggle: 'KAGGLE_DS_ROOT'
}


def get_ds_root_path(source:DataSource):
    return get_config_value(DS_ROOTS_CONFIG_KEYS[source])


def get_ds_path(lang:Lang, source:DataSource):
    assert source in (DataSource.Marcos, DataSource.Kaggle), 'Not implemented for given DataSource'
    if source == DataSource.Marcos:
        path = Path(f'dataset/marcos_de_la_fuente.txt/{lang_to_str(lang)}.txt')
    elif source == DataSource.Kaggle:
        relative_path = 'poemsdataset' if lang == Lang.English else 'spanish-poetry-dataset/poems.csv'
        path = Path(get_ds_root_path(source))/relative_path
    return path.resolve()


def get_text_files(path:Path):
    result = []
    if not path.is_dir(): return result
    
    for child in path.iterdir():
        if child.is_dir():
            result.extend(get_text_files(child))
        elif (child.suffix.lower() == '.txt'):
            result.append(child)
    return result


def get_file_lines(path:Path) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.readlines()
    return text 


def get_ds_root_placeholder(source:DataSource):
    return f'[{DS_ROOTS_CONFIG_KEYS[source]}]'


def _replace_ds_root_w_placeholder(path_str:str):
    path_str = path_str.replace(os.sep, '/')
    ds_roots = [(get_ds_root_path(source), get_ds_root_placeholder(source)) for source in DataSource]
    for ds_root, ds_root_placeholder in ds_roots:
        ds_root = str(Path(ds_root).resolve()).replace(os.sep, '/')
        try:
            ds_root_idx = path_str.index(ds_root)
        except ValueError:
            ds_root_idx = None
        if ds_root_idx == 0:
            return path_str.replace(ds_root, ds_root_placeholder)
    return path_str


class PoemsFileList:
    def __init__(self, paths:List[Path]):
        self.paths = paths
        
    def __iter__(self):
        return iter(self.paths)
    
    def __len__(self):
        return len(self.paths)
    
    def to_metadata_df(self):
        df = pd.DataFrame(columns=[POEM_NAME_DF_COL, POEM_LOCATION_DF_COL])
        _fn_to_poem_name = lambda filename: filename.split('.')[0]
        df[POEM_NAME_DF_COL] = [_fn_to_poem_name(p.name) for p in self.paths]
        df[POEM_LOCATION_DF_COL] = [_replace_ds_root_w_placeholder(str(p)) for p in self.paths]
        return df

    @classmethod
    def from_root_path(cls, root_path:Path, poem_titles_to_ignore:List[str]=None):
        if poem_titles_to_ignore is None: poem_titles_to_ignore = []
        paths = [p for p in get_text_files(root_path) if p.name not in poem_titles_to_ignore]
        return cls(paths)


class PoemsDf:
    def __init__(self, df, poems_column, name_column, origin):
        self.df = df
        self.poems_column = poems_column
        self.name_column = name_column
        self.origin = origin
    
    def __len__(self):
        return len(self.df)
    
    def to_metadata_df(self):
        df = pd.DataFrame(columns=[POEM_NAME_DF_COL, POEM_LOCATION_DF_COL])
        df[POEM_NAME_DF_COL] = self.df[self.name_column]
        df[POEM_LOCATION_DF_COL] = [f'{_replace_ds_root_w_placeholder(str(self.origin))}[{self.poems_column}:{i}]' 
                                    for i in self.df.index]
        return df

    @classmethod
    def from_csv_path(cls, csv_path, poems_column, name_column):
        df = pd.read_csv(csv_path)
        return cls(df, poems_column, name_column, csv_path)
    
    
def get_data_sources(lang:Lang, source:DataSource):
    path = get_ds_path(lang, source)
    
    # TODO: maybe the type (files, DataFrame, ...) could be inferred from the path, 
    # but it'd be just a heuristic at best
    if lang == Lang.Spanish:
        if source == DataSource.Marcos:
            poems_to_ignore = ['other_authors.es.txt']
            return PoemsFileList.from_root_path(path, poem_titles_to_ignore=poems_to_ignore)
        else:
            return PoemsDf.from_csv_path(path, 'content', 'title')
    elif lang == Lang.English:
        return PoemsFileList.from_root_path(path)


class PoemsFileListSplitterByParent:
    def split(self, poems_list:PoemsFileList, valid_pct=0.2):
        all_train_files, all_valid_files = [], []
        parent_paths = set(fp.parent for fp in poems_list)
        poems_set = set(poems_list)
        rng = np.random.default_rng(seed=get_config_value('RNG_SEED'))

        for parent in parent_paths:
            # For any directory with at least two children, we choose at least 
            # one file for validation set, independently of valid_pct
            txt_files_in_parent = set(get_text_files(parent)).intersection(poems_set)# , poems_list.poem_titles_to_ignore)
            num_valid = (0 if len(txt_files_in_parent) <= 1 
                         else max(1, round(valid_pct * len(txt_files_in_parent))))
            valid_files_idxs = rng.choice(len(txt_files_in_parent), size=num_valid, replace=False)
            files_arr = np.array(list(txt_files_in_parent))
            valid_files = files_arr[valid_files_idxs]
            train_files = txt_files_in_parent - set(valid_files)
            all_valid_files.extend(valid_files)
            all_train_files.extend(train_files)
        
        return PoemsFileList(all_train_files), PoemsFileList(all_valid_files)
    
    
class PoemsDfSplitter:        
    def split(self, poems_list:PoemsDf, valid_pct=0.2):
        df = poems_list.df
        all_idxs = list(range(len(df)))
        num_valid = round(valid_pct * len(all_idxs))
        rng = np.random.default_rng(seed=get_config_value('RNG_SEED'))
        valid_idxs = rng.choice(len(all_idxs), size=num_valid, replace=False)
        train_idxs = list(set(all_idxs) - set(valid_idxs))
        train_rows = df.iloc[train_idxs]
        valid_rows = df.iloc[valid_idxs]
        poems_df_args = [poems_list.poems_column, poems_list.name_column, poems_list.origin]
        return PoemsDf(train_rows, *poems_df_args), PoemsDf(valid_rows, *poems_df_args)

    
class SplitterFactory:
    def get_splitter_for(self, data):
        if isinstance(data, PoemsFileList):
            return PoemsFileListSplitterByParent()
        elif isinstance(data, PoemsDf):
            return PoemsDfSplitter()
        return None


def data_splits_to_df(data_splits:Tuple[List,str]):
    dfs = []
    for split, split_name in data_splits:
        for ds in split:
            ds_df = ds.to_metadata_df()
            ds_df['Split'] = [split_name] * len(ds_df)
            dfs.append(ds_df)
    return pd.concat(dfs, ignore_index=True)


class PoemsDfReader():
    def __init__(self, poems_df:PoemsDf):
        self.poems_df = poems_df
    
    def __iter__(self):
        poem_getter = lambda row: row[1][self.poems_df.poems_column]
        df = self.poems_df.df
        return (poem_getter(row).split('\n') for row in df.iterrows() if isinstance(poem_getter(row), str))

    
class PoemsFileReader():
    def __init__(self, poems_list:PoemsFileList, encoding='utf-8'):
        self.poems_list = poems_list
        self.encoding = encoding
    
    def __iter__(self):
        return FilesIterator(list(self.poems_list), self.encoding)
    
    
class FilesIterator():
    def __init__(self, paths, encoding='utf-8'):
        self.paths = paths
        self.encoding = encoding
        self.idx = 0
        
    def __next__(self):
        if self.idx < len(self.paths):
            path = self.paths[self.idx]
            self.idx += 1
            with open(path, 'r', encoding=self.encoding) as f:
                text = f.readlines()
            return text    
        raise StopIteration 

        
class ComposedPoemsReader:
    def __init__(self, readers):
        self.readers = readers
        
    def __iter__(self):
        return (poem for reader in self.readers for poem in reader)


class ReaderFactory:
    def get_reader_for(self, data):
        if isinstance(data, PoemsFileList):
            return PoemsFileReader(data)
        elif isinstance(data, PoemsDf):
            return PoemsDfReader(data)
        return None


class VerseGrouping(Enum):
    OneVerseBySequence = auto()
    OnePoemBySequence = auto()


@dataclass
class PoemsFileConfig:
    remove_multispaces:bool = True
    beginning_of_verse_token:str = ''
    end_of_verse_token:str = '\\n'
    end_of_poem_token:str = ''
    n_prev_verses_terminations:int = 0
    verse_grouping:VerseGrouping = VerseGrouping.OneVerseBySequence

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, default=lambda x: x.value)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            attrs_dict = json.load(f)
        attrs_dict['verse_grouping'] = VerseGrouping(attrs_dict['verse_grouping'])
        return cls(**attrs_dict)


class PoemsIOWriter():
    def __init__(self, open_file, conf:PoemsFileConfig):
        self.file = open_file
        self.conf = conf
        self.meaningful_chars_pattern = re.compile("[a-zA-Z0-9\?]")
        self.multispace_pattern = re.compile(" {2,}") if conf.remove_multispaces else None
        self.eol_punctuation_pattern = re.compile('[ .,;\?\!\-\\\:]+$')
        
    def write_verse(self, verse, endofpoem=False, prev_verses_last_word:List[str]=None):
        last_word = ''
        if (not endofpoem) and (self.meaningful_chars_pattern.search(verse) is None):
            return last_word
        if self.multispace_pattern is not None:
            verse = self.multispace_pattern.sub(" ", verse)
        verse = verse.strip()

        if (len(verse) > 0) or endofpoem:
            last_word = self.eol_punctuation_pattern.sub('', verse).split(' ')[-1]
            verse_end = self.conf.end_of_verse_token
            if endofpoem: verse_end += self.conf.end_of_poem_token
            if endofpoem or (self.conf.verse_grouping == VerseGrouping.OneVerseBySequence):
                verse_end += os.linesep
            prev_ends = (' '.join(prev_verses_last_word[-self.conf.n_prev_verses_terminations:])
                         if self.conf.n_prev_verses_terminations > 0
                         else '')
            encoded_verse = f'{prev_ends}{self.conf.beginning_of_verse_token}{verse} {verse_end}'
            self.file.write(encoded_verse)

        return last_word

    def write_poem(self, poem_lines:List[str]):
        verses_endings = []
        for line in poem_lines[:-1]:
            last_word = self.write_verse(line, prev_verses_last_word=verses_endings)
            if last_word != '': verses_endings.append(last_word)
        if len(poem_lines) > 0:
            self.write_verse(poem_lines[-1], endofpoem=True, prev_verses_last_word=verses_endings)

                    
def merge_poems(poems_reader, poems_writer):
    for poem_lines in poems_reader:
        poems_writer.write_poem(poem_lines)


class DfCache():
    def __init__(self):
        self.dfs_dict = dict()
        
    def get(self, path:Union[str,Path]):
        path = str(path)
        if path not in self.dfs_dict:
            self.dfs_dict[path] = pd.read_csv(path)
        return self.dfs_dict[path]
        
    def clear(self):
        self.dfs_dict.clear()


class LabelsType(Enum):
    Forms = "forms"
    Topics = "topics"
    All = "form_and_topic"


def label_type_to_str(label_type:LabelsType):
    return label_type.value[:-1] if label_type != LabelsType.All else label_type.value


class LabeledPoem:
    def __init__(self, poem_lines:List[str], labels:dict):
        self.poem_lines = poem_lines
        self.labels = labels
        
    def __repr__(self):
        return f'[Labels]: {self.labels}\n[Content]: {self.poem_lines}'


class PoemsSplitsDfContentReader:
    def __init__(self):
        self.df_cache = DfCache()

    def _location_is_just_path(self, location:str):
        return not location.endswith(']')
    
    def _location_to_path(self, location:str) -> Path:
        location_is_just_path = self._location_is_just_path(location)
        if location_is_just_path:
            return Path(location)
        else:
            assert '[' in location, f"Missing character '[' or unexpected character ']' in splits DataFrame for location {location}"
            return Path(location[:location.rindex('[')])
        
    def extract_poem_lines(self, location:str) -> List[str]:
        path = self._location_to_path(location)
        if self._location_is_just_path(location):
            return get_file_lines(path)
        else:
            assert path.suffix.lower() == '.csv',(
                f'Found location with [identifier] {location} in splits DataFrame, but only csv extension is supported for locations with identifiers'
            )
            df = self.df_cache.get(path)
            location_id = location[location.rindex('[')+1:-1]
            df_col, df_row = location_id.split(':')
            poem_content = df.iloc[int(df_row)][df_col]
            return poem_content.split('\n')


class LabeledPoemsSplitsDfReader:
    def __init__(self, df:pd.DataFrame, label_func:Callable[[str], dict]=None):
        self.df = df
        self.label_func = label_func if label_func is not None else self._default_label_func
        self.content_reader = PoemsSplitsDfContentReader()
        
    def _default_label_func(self, location:str) -> dict:
        if isinstance(location, str): location = Path(location)
        return {'cat1': location.parent.name}
            
    def __iter__(self):
        locations = [row[POEM_LOCATION_DF_COL] for _, row in self.df.iterrows()]
        return (LabeledPoem(self.content_reader.extract_poem_lines(loc), self.label_func(loc)) 
                for loc in locations if self.content_reader._location_to_path(loc).exists())


class BaseLabelsWriter(ABC):
    @abstractmethod
    def write_labels(self, labels:dict, poems_writer:PoemsIOWriter):
        pass

    @abstractmethod
    def num_verses_needed(self, n_total_categories:int) -> int:
        pass


class LabelsWriterStd(BaseLabelsWriter):
    def write_labels(self, labels:dict, poems_writer:PoemsIOWriter):
        for label in labels.values():
            if label == '': label = '?'
            poems_writer.write_verse(label)

    def num_verses_needed(self, n_total_categories:int) -> int:
        return n_total_categories


class LabelsWriterKeyValue(BaseLabelsWriter):   
    def write_labels(self, labels:dict, poems_writer:PoemsIOWriter):
        def qm_if_empty(label): return '?' if label == '' else label
        labels_verse = ', '.join(f'{k}: {qm_if_empty(v)}' for k, v in labels.items())
        poems_writer.write_verse(labels_verse)

    def num_verses_needed(self, n_total_categories:int) -> int:
        return 1


class LabelsWriterKeyValueMultiverse(BaseLabelsWriter):
    def write_labels(self, labels:dict, poems_writer:PoemsIOWriter):
        def qm_if_empty(label): return '?' if label == '' else label
        for k, v in labels.items():
            poems_writer.write_verse(f'{k}: {qm_if_empty(v)}')

    def num_verses_needed(self, n_total_categories:int) -> int:
        return n_total_categories


class LabelsWriterExplained(BaseLabelsWriter):
    def __init__(self, omit_empty=False):
        self.omit_empty = omit_empty
        
    def write_labels(self, labels:dict, poems_writer:PoemsIOWriter):
        labels_verse = 'This is a poem'
        for cat, label in labels.items():
            if label == '': 
                if self.omit_empty: 
                    continue
                else: 
                    label = '?'
            if cat == label_type_to_str(LabelsType.Topics):
                labels_verse += f' about {label}'
            elif cat == label_type_to_str(LabelsType.Forms):
                labels_verse += f' with {label} form'
        labels_verse += ':'
        poems_writer.write_verse(labels_verse)

    def num_verses_needed(self, n_total_categories:int) -> int:
        return 1


class BaseLabelsDecoder(ABC):
    """Child classes must define a method of extracting the labels from a poem(s) text."""
    @abstractmethod
    def decode_labels(self, text:str, file_config:PoemsFileConfig) -> Dict[LabelsType, str]:
        pass


class LabelsDecoderKeyValue(BaseLabelsDecoder):
    """Extracts the labels from a poem(s) text with the format used by `LabelsWriterKeyValue`.
    
    It assumes that `file_config.end_of_verse_token` and `file_config.end_of_poem_token` are
    defined (not empty).
    """
    def decode_labels(self, text:str, file_config:PoemsFileConfig) -> Dict[LabelsType, str]:
        labels = []
        text = text.replace(file_config.beginning_of_verse_token, '')
        first_verse_by_poem = [
            poem_text.split(file_config.end_of_verse_token)[0]
            for poem_text in text.split(file_config.end_of_poem_token)
            if len(poem_text) > 0
        ]

        topic_key = label_type_to_str(LabelsType.Topics)#+":"
        form_key = label_type_to_str(LabelsType.Forms)#+":"
        unknown_str = "?"

        for labels_verse in first_verse_by_poem:
            key_value_list = labels_verse.split(', ')
            poem_labels = dict()
            labels.append(poem_labels)
            
            for key_value_str in key_value_list:
                if key_value_str.startswith(topic_key):
                    value_str = key_value_str[len(topic_key)+1:].strip()
                    poem_labels[topic_key] = value_str if value_str != unknown_str else ''
                elif key_value_str.startswith(form_key):
                    value_str = key_value_str[len(form_key)+1:].strip()
                    poem_labels[form_key] = value_str if value_str != unknown_str else ''
            
            if topic_key not in poem_labels: poem_labels[topic_key] = ''
            if form_key not in poem_labels: poem_labels[form_key] = ''

        return labels


class LabelsDecoderExplained(BaseLabelsDecoder):
    """Extracts the labels from a poem(s) text with the format used by `LabelsWriterExplained`.
    
    It assumes that `file_config.end_of_verse_token` and `file_config.end_of_poem_token` are
    defined (not empty).
    """
    def decode_labels(self, text:str, file_config:PoemsFileConfig) -> Dict[LabelsType, str]:
        labels = []
        text = text.replace(file_config.beginning_of_verse_token, '')
        first_verse_by_poem = [
            poem_text.split(file_config.end_of_verse_token)[0]
            for poem_text in text.split(file_config.end_of_poem_token)
            if len(poem_text) > 0
        ]
        topic_key = label_type_to_str(LabelsType.Topics)
        form_key = label_type_to_str(LabelsType.Forms)

        for labels_verse in first_verse_by_poem:
            labels_verse = labels_verse.strip()
            if labels_verse[-1] == ":": 
                labels_verse = labels_verse[:-1]
            verse_tokens = labels_verse.split(' ')
            poem_labels = dict()
            labels.append(poem_labels)
            
            try:
                about_idx = verse_tokens.index('about')
            except ValueError:
                about_idx = -1
            poem_labels[topic_key] = (
                self._empty_label_if_unknown(verse_tokens[about_idx + 1].strip())
                if -1 < about_idx < (len(verse_tokens) - 1)
                else ''
            )
            try:
                with_idx = verse_tokens.index('with')
            except ValueError:
                with_idx = -1
            poem_labels[form_key] = (
                self._empty_label_if_unknown(verse_tokens[with_idx + 1].strip())
                if -1 < with_idx < (len(verse_tokens) - 1)
                else ''
            )
            
        return labels
    
    def _empty_label_if_unknown(self, label):
        return '' if label == '?' else label


class LabeledPoemsIOWriter():
    def __init__(self, open_file, conf:PoemsFileConfig, labels_writer:BaseLabelsWriter=None):
        self.poems_file_writer = PoemsIOWriter(open_file, conf)
        self.labels_writer = labels_writer if labels_writer is not None else LabelsWriterStd()
        
    def write_poem(self, labeled_poem:LabeledPoem):
        self.labels_writer.write_labels(labeled_poem.labels, self.poems_file_writer)
        self.poems_file_writer.write_poem(labeled_poem.poem_lines)


def filter_splits_df(df, labels_type, split):
    kaggle_ds_root_placeholder = get_ds_root_placeholder(DataSource.Kaggle)
    labels_type_filter = f'/{labels_type.value}/' if labels_type != LabelsType.All else ''
    # Only the Kaggle datasets have labels
    kaggle_filter = kaggle_ds_root_placeholder if labels_type != LabelsType.All else ''
    return df.copy()[
        df.Location.str.contains(labels_type_filter, regex=False)
        & df.Location.str.contains(kaggle_filter, regex=False)
        & (df.Split == split)        
    ]


def replace_location_placeholder(splits_df):
    kaggle_ds_root_placeholder = get_ds_root_placeholder(DataSource.Kaggle)
    own_ds_root_placeholder = get_ds_root_placeholder(DataSource.Marcos)
    kaggle_ds_root = get_ds_root_path(DataSource.Kaggle)
    own_ds_root = get_ds_root_path(DataSource.Marcos)
    splits_df.Location = splits_df.Location.str.replace(kaggle_ds_root_placeholder, 
                                                        kaggle_ds_root,
                                                        regex=False)
    splits_df.Location = splits_df.Location.str.replace(own_ds_root_placeholder, 
                                                        own_ds_root,
                                                        regex=False)
    return splits_df


def build_labeled_dfs_from_splits(splits_df:pd.DataFrame, labels_type:LabelsType):
    train_split_df = replace_location_placeholder(filter_splits_df(splits_df, labels_type, 'Train'))
    valid_split_df = replace_location_placeholder(filter_splits_df(splits_df, labels_type, 'Validation'))
        
    def _get_content_of_file_path(path:str):
        if not Path(path).exists():
            # Some poems contain strange characters in the title that don't match 
            # the original poem name, but they are about 1% and some are in french 
            # or other languages, so we don't mind discarding them
            #print('skipped', path)
            return ''
        with open(path) as f:
            return f.read()
        
    def _split_to_labeled_df(split_df):
        labeled_df = pd.DataFrame({
            'text': split_df.Location.map(_get_content_of_file_path), 
            'labels': split_df.Location.map(lambda path: Path(path).parent.name), 
        })
        return labeled_df
    
    train_df = _split_to_labeled_df(train_split_df)
    valid_df = _split_to_labeled_df(valid_split_df)
    train_empty_selector = train_df.text == ''
    valid_empty_selector = valid_df.text == ''
    train_df = train_df[~train_empty_selector]
    valid_df = valid_df[~valid_empty_selector]
    
    return train_df, valid_df


def build_datasets_for_encoder_decoder(
    splits_df:pd.DataFrame, file_conf:PoemsFileConfig, label_func:Callable[[str], Dict[str, str]]
) -> Tuple[Dataset, Dataset]:
    """Construct the datasets needed to train a HuggingFace Encoder-Decoder using the data in `splits_df`.

    Args:
        splits_df: DataFrame that has one row by poem and assigns each one to an split. It must contain, at least, 
            the following columns:
                `POEM_LOCATION_DF_COL`: location specification of a poem, with the same format as the columns of the
                    same name filled by `PoemsFileList` and `PoemsDf`.
                'Split': it must have the value 'Train' or 'Validation' depending on whether the poem belongs to the
                    training set or the validation set.
            You'd normally use a DataFrame that has been previously stored as a csv by the notebook 
            'txt_inputs_generator'.
        file_conf: formatting options that must be applied to the poems before training.
        label_func: function that receives the location specification of a poem as input and returns a dictionary
            that contains the labels (values) assigned to each category (keys). The text corresponding to each 
            categories is given by `label_type_to_str`.
    Returns:
        Tuple thats has the training set as the first element and the validation set as the second one.
        Each of these two datasets has the columns 'text' (text of the poem) and 'labels_text' (labels as encoded
        by `LabelsWriterExplained`).
    """
    splits_df = replace_location_placeholder(splits_df)
    train_split_df = filter_splits_df(splits_df, LabelsType.All, 'Train')
    valid_split_df = filter_splits_df(splits_df, LabelsType.All, 'Validation')
    
    readers = [LabeledPoemsSplitsDfReader(df, label_func) for df in (train_split_df, valid_split_df)]
    encoded_dfs = []
    labels_writer = LabelsWriterExplained()
    
    encoded_dfs = []
    for reader in readers:
        encoded_df_columns = {'text': [], 'labels_text': []}
        for poem in reader:
            with io.StringIO() as labels_io:
                # This writes endofverse token, which shouldn't be needed for the encoder
                labels_io_writer = PoemsIOWriter(labels_io, file_conf)  
                labels_writer.write_labels(poem.labels, labels_io_writer)
                labels_text = labels_io.getvalue()
            with io.StringIO() as content_io:
                content_io_writer = PoemsIOWriter(content_io, file_conf)
                content_io_writer.write_poem(poem.poem_lines)
                poem_text = content_io.getvalue()

            encoded_df_columns['text'].append(poem_text)
            encoded_df_columns['labels_text'].append(labels_text)

        encoded_dfs.append(pd.DataFrame(encoded_df_columns))
        
    encoded_train_df, encoded_valid_df = encoded_dfs
    train_ds = Dataset.from_pandas(encoded_train_df)
    valid_ds = Dataset.from_pandas(encoded_valid_df)
    
    return train_ds, valid_ds


class LabelsEstimator:
    """Class that packs a trained classifier for each `LabelsType` category.

    It assumes there's a classifier in the HuggingFace Hub of the user `hf_user`
    for each category and with a name that follows the same pattern as the models
    trained with the notebook 'train_poems_classifier.ipynb'.

    Args:
        base_model_name: name of the architecture of the model, as defined by Hugging
            Face. For instance: gpt2, distilbert, big_bird, ...
        hf_user: Hugging Face username of the owner of the models
        hf_token (optional): Hugging Face access token of the owner of the models, with
            read permission. Only needed if any of the models is private.
        device: Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, 
            a positive will run the model on the associated CUDA device id.
    """
    def __init__(self, base_model_name:str, hf_user, hf_token:Optional[str]=None, device=-1):
        self.clf_pipelines = dict()
        for cat in LabelsType:
            if cat == LabelsType.All: continue
            checkpoint_name = f'{base_model_name}-poems-clf-by-{label_type_to_str(cat)}.en'
            model_id = get_model_id(checkpoint_name, hf_user)
            clf = AutoModelForSequenceClassification.from_pretrained(model_id, use_auth_token=hf_token)
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
            self.clf_pipelines[cat] = TextClassificationPipeline(
                model=clf, 
                tokenizer=tokenizer, 
                device=device,     
            )

    def predict(self, cat:LabelsType, poem_lines:List[str]) -> str:
        poem_content = '\n'.join(poem_lines)
        return self.clf_pipelines[cat](poem_content, truncation=True)[0]['label']
