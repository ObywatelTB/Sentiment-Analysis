import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm   #progress bar
from typing import Tuple, List, Dict, Any
from rich import print as rprint
import gc   #garbage collector

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

from sourcing import utils as su
import sourcing.utils as sut
from sourcing.SQLiteService import SQLiteService as sqlserv
from sourcing.CSVService import CSVService as csvserv


def load_encoded_tweets(dirs: Dict[str, str], dataset_name: str, 
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Load two ndarrays of: previously encoded selected tweets,
    and corresponding target values ({0,1}).
    
    Args:
        dataset_name (str) : 'db'/'kaggle'/'selection'
    """
    dirpath = {
        'kaggle': dirs['kaggle_encoded'],
        'selection' : dirs['selected_tweets_encoded'],
        'db': dirs['fil_opinions_encoded']
    }[dataset_name]
    
    filepath = su.get_most_recent_filepath(dirpath, '.', 'int_format')
    loaded = np.load(filepath)
    return loaded['X'], loaded['Y'] 


def vectorize_dataset(model_parameters: Dict[str, Any], dirs: Dict[str, str], 
                    dbtablename: str, dataset_name: str = 'selection') -> None:
    """
    Encode texts of choice using Google USE and save in the files.

    Args:
        model_parameters (dict[str, Any]) : Parameters concerning model 
        structure, training parameters.
        dirs (dirs[str, str]) : A set of needed directory paths.
        which_df (str) : 'kaggle'/'selection' - Choose between two different 
        sets of texts to vectorize.

    Returns:
        None

    Raises:
    """

    rprint('[italic red] Loading the encoder... [/italic red]')
    embed = hub.load(dirs.get('universal_sentence_encoder', ''))
    
    DataService, batch_generator, import_path, export_dirpath = {
        'db': (sqlserv, sqlserv.setup_batch_generator,
            dirs['fil_opinions_encoded'], dirs['fil_opinions_encoded']),
        'kaggle': (csvserv, csvserv.kaggle_batch_generator,
                dirs['kaggle_encoded'], dirs['kaggle_encoded']),
        'selection': (csvserv, csvserv.selected_batch_generator,
                    dirs['selected_tweets'], dirs['selected_tweets_encoded'])
    }[dataset_name]

    batch_getter = getattr(DataService, batch_generator.__name__)

    data_service = DataService(import_path, dbtablename)
    batch_gen, batch_amount = batch_getter(data_service)

    vectorize_batches(embed, batch_gen, batch_amount, export_dirpath)


def vectorize_batches(embed, batch_gen, batch_amount: int, export_dirpath: str
                        ) -> None:
    """Encode the Kaggle dataset, splitting the process into batches."""
    rprint('[italic red] Loading kaggle data... [/italic red]')

    vectorized_tweets = []
    targets = []
    rprint(f'Encoding [italic red]{batch_amount}[/italic red] batches of data...')

    # Iterating over small (eg 256-long) batches, digestible for embed:
    progress_bar = tqdm(total=batch_amount)
    for _ in range(batch_amount):
        df_batch = next(batch_gen)
        if len(df_batch) > 0:
            batch_vect_tweets, batch_targets = vectorize_phrases(df_batch, embed)
            del df_batch; gc.collect()
            vectorized_tweets = [*vectorized_tweets, *batch_vect_tweets]
            targets = [*targets, *batch_targets]
        progress_bar.update(1)

    vectorized_tweets = np.array(vectorized_tweets)
    targets = np.array(targets)
    save_vectorized_dataset(export_dirpath, vectorized_tweets, targets)


def vectorize_all_at_once(embed, tweets, export_dirpath: str) -> None:
    """Encode at once dataset and save in a file.""" 
    vect_tweets, targets = vectorize_phrases(tweets, embed)
    save_vectorized_dataset(export_dirpath, vect_tweets, targets)


def vectorize_phrases(phrases: pd.DataFrame, embed
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorize using Google USE."""
    encoded = embed(phrases['content'].values.tolist()).numpy()

    if 'sentiment' in phrases.columns:
        targets = phrases['sentiment'].values
    else:
        targets = np.full((len(encoded)), -1)
    return encoded, targets


def save_vectorized_dataset(export_dirpath: str, vect_tweets: np.ndarray,
                            targets: np.ndarray) -> None:
    """Saves encoded tweets and targets to numpy's .npz file."""
    if not os.path.exists(export_dirpath):
        os.makedirs(export_dirpath, exist_ok=True)
    
    export_filename = f'{int(time.time())}.npz'
    export_filepath = os.path.join(export_dirpath, export_filename)
    np.savez(export_filepath, X=vect_tweets, Y=targets) #X,Y - dictionary keys
    print(f'Saved to file: {export_filepath}.')