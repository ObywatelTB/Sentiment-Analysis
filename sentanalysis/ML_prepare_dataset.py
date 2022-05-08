import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm   #progress bar
from typing import Tuple, List, Dict, Any
from rich import print as rprint
import gc #garbage collector

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

#-----------------------------------------------------------------------            72

def load_datasets(dirs: Dict[str, str]) -> Tuple:
    """Load train and validation datasets. Load kaggle and """
    df = get_kaggle_encoded_tweets(dirs)
    df_ours = pd.read_csv(dirs['selected_tweets_encoded'],header=None,
                        usecols=[1,2],names=['tweet','sentiment'])
    df_ours['tweet'] = df_ours['tweet'].apply(string_to_ndarray)

    # df = pd.concat([df,df,df,df,df])    # worked for 500k
    df_train, df_val, df_test_kaggle = split_datasets(df)
    all_ds = [df_train, df_val, df_test_kaggle,  df_ours]
    train_ds, val_ds, test_ds_kaggle,  ds_ours = \
        [switch_to_numpy_tuple(ads) for ads in all_ds]
    return train_ds, val_ds, test_ds_kaggle, ds_ours


def get_kaggle_encoded_tweets(dirs: Dict[str, str], dirname: str ='kaggle1638301819'
                            ) -> pd.DataFrame:
    """
    Load previously performed encodings of the kaggle 1.6mln tweets dataset.

    Args:
        dirs (dict[str, str]) : The main project directories.
        dirname (str) : The path to directory with the Kaggle dataset.

    Returns:
        dfs_arr (DataFrame) : Texts encoded with the Gooogle USE encoder.
    Raises:
    """
    dirpath = dirs.get('kaggle_encoded')
    dfs_arr = []
    filenames = os.listdir(dirpath)[:5]
    with tqdm(total= len(filenames) ) as progress_bar:
        for fn in filenames:
            filepath = os.path.join(dirpath, fn)
            dfs_arr.append( pd.read_csv(filepath, header=None,usecols=[1,2],
                                        names=['sentiment','tweet']) )
            progress_bar.update(1)
    return pd.concat(dfs_arr)


def vectorize_dataset(model_parameters: Dict[str, Any], dirs: Dict[str, str], 
                        dataset_name: str = 'selection') -> None:
    """
    Encode texts of choice using Google USE and save in the files.

    Args:
        model_parameters (dict[str, Any]) : Parameters concerning model 
        structure, training parameters.
        dirs (dirs[str, str]) : A set of needed directory paths.
        which_df (str) : 'kaggle'/'selection' - Chose between two different sets
        of texts to vectorize.

    Returns:
        None

    Raises:
    """
    dataset_encoding = model_parameters.get('dataset_encoding', '')
    rprint('[italic red] Loading the encoder... [/italic red]')
    embed = hub.load(dirs.get('universal_sentence_encoder', ''))

    export_dirpath = os.path.join(dirs['opinions_encoded'], 
                                'selected_tweets_encoded')
    # export_dirpath = {
    #     'selection': os.path.join(dirs['opinions_encoded'], 
    #                             'selected_tweets_encoded'),
    #     'kaggle': os.path.join(dirs['opinions_encoded'],
    #                             f'kaggle{int(time.time())}')
    # }[dataset_name]

    if not os.path.exists(export_dirpath):
        os.makedirs(export_dirpath, exist_ok=True)
    
    if dataset_name == 'selection':
        vectorize_selected_tweets(embed, dirs, export_dirpath)
    if dataset_name == 'kaggle':
        vectorize_kaggle_batches(embed, dirs, dataset_encoding, export_dirpath)


def vectorize_selected_tweets(embed, dirs: Dict[str, str], export_dirpath: str,
                            sentiment_treshold: float = 0.3) -> None:
    """
    Encode the selected tweets dataset. Cuts the dataset wrt. sentiment,
    to outliers only: (0, tres)U(1-tres, 1). Thus leaving only the extreme
    values of the scores (definitely positive or def. negative).

    Args:
        sentiment_treshold (float) : It is a percentage of values to leave
        from boths sides. So it leaves twice the amount (eg. 0.3 -> 60% left).
    """ 
    old_limits = [-9, 9]  # Defines the scale in which sentiment is graded.
    old_range = np.sum(np.abs(old_limits))  # eg. (-9, 9) -> 18

    tweets = pd.read_csv(dirs['selected_tweets'], skiprows=[0,1],
                    names=['tweet','sentiment'] )
    tweets['sentiment'] = tweets['sentiment'].apply(lambda x: x/old_range + 0.5)
    tres_opposite = 0.5 - sentiment_treshold
    tweets = tweets[abs(tweets['sentiment'] - 0.5) > tres_opposite] # ~600 -> 270
    tweets['sentiment'] = tweets['sentiment'].apply(round)

    vect_tweets, targets = vectorize_phrases(tweets, embed)
    save_vectorized_dataset(export_dirpath, 'selection', vect_tweets, targets)


def vectorize_kaggle_batches(embed, dirs: Dict[str, str], dataset_encoding: str, 
                            export_dirpath: str) -> None:
    """Encode the Kaggle dataset, splitting the process into batches."""
    rprint('[italic red] Loading kaggle data... [/italic red]')
    tweets = get_kaggle_tweets(dirs['kaggle_tweets'], dataset_encoding) 

    tweets = shuffle(tweets).reset_index(drop=True)
    tweets = tweets.iloc[:921600]
    max_batch = 256     # near ~4k embed definitely stops
    batch_amount = int(len(tweets)/max_batch)
    batch_gen = batch_generator(tweets, batch_size = max_batch)
    del tweets; gc.collect()
    vectorized_tweets = []
    targets = []
    rprint(f'[italic red] Encoding {len(tweets)} tweets...[/italic red]')
    with tqdm(total= batch_amount ) as progress_bar:
        # Iterating over small 256-long-batches, digestible for embed:
        for inx in range(batch_amount):
            df_batch = next(batch_gen)
            batch_vect_tweets, batch_targets = vectorize_phrases(df_batch, embed)
            del df_batch; gc.collect()
            vectorized_tweets = [*vectorized_tweets, *batch_vect_tweets]
            targets = [*targets, *batch_targets]
            progress_bar.update(1)
    vectorized_tweets = np.array(vectorized_tweets)
    targets = np.array(targets)
    save_vectorized_dataset(export_dirpath, 'kaggle', vectorized_tweets, targets)


def save_vectorized_dataset(export_dirpath: str, filename:str, vect_tweets: np.ndarray,
                            targets: np.ndarray) -> None:
    """Saves encoded tweets and targets to numpy's .npz file."""
    export_filename = f'{filename}{int(time.time())}.npz'
    export_filepath = os.path.join(export_dirpath, export_filename)
    np.savez(export_filepath, X=vect_tweets, Y=targets) #X,Y - dictionary keys
    print(f'Saved to file: {export_filepath}.')


def vectorize_phrases(phrases: pd.DataFrame, embed
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorize using Google USE."""
    encoded = embed(phrases['tweet'].values.tolist()).numpy()
    # phrases['tweet'] = encoded
    return encoded, phrases['sentiment'].values  # encoded[0].shape -> (512,)


def get_kaggle_tweets(dataset_path: str, dataset_encoding: str, start: int = 0, 
                        amount: int = 0) -> pd.DataFrame:
    """Load from file the Kaggle dataset of 1.6mln tweets. Their sentiment
    is in {0, 4} set - rated as 0 (negative) and 4 (positive)."""
    tweets = pd.read_csv(dataset_path, encoding = dataset_encoding)
    if amount:
        tweets = tweets.iloc[start:amount]
    tweets= tweets.iloc[:,[0,-1]]
    tweets.columns = ['sentiment','tweet']
    tweets.sentiment = tweets.sentiment.map({0:0,4:1})
    print(f'Got {len(tweets)} tweets from kaggle set.')
    return tweets


def batch_generator(data: pd.DataFrame, batch_size: int):
    """Usage:   a=batch_generator(df,n);   next(a); next(a)."""
    while True:
        for i in range(0, len(data), batch_size):
            data_batch = data.iloc[i: i+batch_size]
            yield data_batch


def split_datasets(data: pd.DataFrame, train_size: int = 1024*70, 
            test_size: int = 2048) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Divide into 3 datasets for training. The amount of all of the returned
    samples is MAX_TWEETS + TEST_SIZE. It works for max 4000 (GPU 0.33).
    For 0.5 GPU it works for 5000.

    Args:
        data (DataFrame) : The dataset to train.
        train_size (int) : Size of a train dataset. Divisible by 8.
        test_size (inte) : a 
    
    Returns:
        df_train, df_val, df_test (tuple[DataFrame,DataFrame,Series]) : A
        tuple of sets of data to train and test.
    Raises:
    """
    rprint('[italic red]Getting the data... [/italic red]')
    data['tweet'] = data['tweet'].apply(string_to_ndarray)
     
    train_size = int(int(len(data)/8)*6)
    max_tweets = int(train_size / 8 * 10) 
    
    sent0 = data.query("sentiment==0")
    sent1 = data.query("sentiment==1")
    assert max_tweets+test_size <= len(data)
    
    tres1 = int(max_tweets/2)  
    tres2 = int(max_tweets/2) + int(test_size/2)
    new_df = pd.concat([sent0.iloc[:tres1], sent1.iloc[:tres1]])
    df_test = pd.concat([ sent0.iloc[tres1:tres2], 
                          sent1.iloc[tres1:tres2] ])
    new_df = shuffle(new_df).reset_index(drop=True)
    assert len(new_df) == max_tweets
    assert len(new_df.query("sentiment==0")) == tres1
    assert len(new_df.query("sentiment==1")) == tres1

    rprint(f'The number of analysed tweets: \
            [italic red] {max_tweets+test_size}[/italic red].')
    df_train, df_val = train_test_split(new_df, test_size=0.2)
    return df_train, df_val, df_test


def switch_to_numpy_tuple(data: pd.DataFrame):
    X = data.tweet.to_numpy()
    Y = data.sentiment.to_numpy()
    X = [np.asarray(x).astype('float32') for x in X]
    Y = [x.astype('float32') for x in Y]
    X = np.array(X)
    Y = np.array(Y)
    return (X,Y)


def string_to_ndarray(a_string: str) -> np.ndarray:
    """Get an ndarray from a string loaded from a file."""
    a_list = [float(x) for x in a_string.replace('[','').replace(']','').split(',')]
    return np.asarray(a_list, dtype=np.float)


def prepare_data_set0(data: pd.DataFrame):
    X = data.tweet.to_numpy()
    Y = data.sentiment.to_numpy()
    # numeric_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    # ds_final = numeric_dataset.shuffle(1000).batch(BATCH_SIZE) #numeric_batches

    AUTOTUNE = tf.data.AUTOTUNE
    # ds_final = ds_final.cache().prefetch(buffer_size=AUTOTUNE)
    return X,Y #ds_final
