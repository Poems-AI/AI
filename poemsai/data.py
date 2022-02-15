from dataclasses import asdict, dataclass
from poemsai.config import get_config_value
from enum import auto, Enum
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from typing import Callable, List, Tuple, Union


__all__ = [
    'Lang', 'lang_to_str', 'DataSource', 'get_ds_path', 'get_text_files', 'get_file_lines',
    'get_data_sources', 'SplitterFactory', 'data_splits_to_df', 'PoemsDfReader', 'PoemsFileReader', 
    'ComposedPoemsReader', 'ReaderFactory', 'PoemsFileConfig', 'PoemsFileWriter', 'merge_poems', 
    'DfCache', 'LabelsType', 'LabeledPoem', 'LabeledPoemsSplitsDfReader', 'LabeledPoemsFileWriter',
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


def get_ds_path(lang:Lang, source:DataSource):
    assert source in (DataSource.Marcos, DataSource.Kaggle), 'Not implemented for given DataSource'
    if source == DataSource.Marcos:
        path = Path(f'dataset/marcos_de_la_fuente.txt/{lang_to_str(lang)}.txt')
    elif source == DataSource.Kaggle:
        relative_path = 'poemsdataset' if lang == Lang.English else 'spanish-poetry-dataset/poems.csv'
        path = Path(get_config_value('KAGGLE_DS_ROOT'))/relative_path
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


def _replace_ds_root_w_placeholder(path_str:str):
    path_str = path_str.replace(os.sep, '/')
    ds_roots_keys = ['KAGGLE_DS_ROOT', 'OWN_DS_ROOT']
    ds_roots = [(get_config_value(key), key) for key in ds_roots_keys]
    for ds_root, ds_root_key in ds_roots:
        ds_root = str(Path(ds_root).resolve()).replace(os.sep, '/')
        try:
            ds_root_idx = path_str.index(ds_root)
        except ValueError:
            ds_root_idx = None
        if ds_root_idx == 0:
            return path_str.replace(ds_root, f'[{ds_root_key}]')
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
    def __init__(self, poems_list:PoemsFileList):
        self.poems_list = poems_list
    
    def __iter__(self):
        return FilesIterator(list(self.poems_list))
    
    
class FilesIterator():
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        
    def __next__(self):
        if self.idx < len(self.paths):
            path = self.paths[self.idx]
            self.idx += 1
            with open(path, 'r') as f:
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


class PoemsFileWriter():
    def __init__(self, open_file, conf:PoemsFileConfig):
        self.file = open_file
        self.conf = conf
        self.meaningful_chars_pattern = re.compile("[a-zA-Z0-9]")
        self.multispace_pattern = re.compile(" {2,}") if conf.remove_multispaces else None
        self.eol_punctuation_pattern = re.compile('[ .,;\?\!\-\\\:]+$')
        
    def write_verse(self, verse, endofpoem=False, prev_verses_last_word:List[str]=None):
        last_word = ''
        if self.meaningful_chars_pattern.search(verse) is None:
            return last_word
        if self.multispace_pattern is not None:
            verse = self.multispace_pattern.sub(" ", verse)
        verse = verse.strip()

        if len(verse) > 0:
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


class LabeledPoem:
    def __init__(self, poem_lines:List[str], label:str):
        self.poem_lines = poem_lines
        self.label = label
        
    def __repr__(self):
        return f'[Label]: {self.label}\n[Content]: {self.poem_lines}'


class LabeledPoemsSplitsDfReader:
    def __init__(self, df:pd.DataFrame, label_func:Callable[[str], str]=None):
        self.df = df
        self.label_func = label_func if label_func is not None else self._default_label_func
        self.df_cache = DfCache()
        
    def _default_label_func(self, location:str) -> str:
        if isinstance(location, str): location = Path(location)
        return location.parent.name
    
    def _location_is_just_path(self, location:str):
        return not location.endswith(']')
    
    def _location_to_path(self, location:str) -> Path:
        location_is_just_path = self._location_is_just_path(location)
        if location_is_just_path:
            return Path(location)
        else:
            assert '[' in location, f"Missing character '[' or unexpected character ']' in splits DataFrame for location {location}"
            return Path(location[:location.rindex('[')])
        
    def _extract_poem_lines(self, location:str) -> List[str]:
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
            
    def __iter__(self):
        locations = [row[POEM_LOCATION_DF_COL] for _, row in self.df.iterrows()]
        return (LabeledPoem(self._extract_poem_lines(loc), self.label_func(loc)) 
                for loc in locations if self._location_to_path(loc).exists())


class LabeledPoemsFileWriter():
    def __init__(self, open_file, conf:PoemsFileConfig):
        self.poems_file_writer = PoemsFileWriter(open_file, conf)
        
    def write_poem(self, labeled_poem:LabeledPoem):
        self.poems_file_writer.write_verse(labeled_poem.label)
        self.poems_file_writer.write_poem(labeled_poem.poem_lines)
