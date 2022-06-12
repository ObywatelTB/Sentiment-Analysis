import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from sourcing.utils import prepare_df_info
from sourcing.SQLiteService import SQLiteService


class CSVService:
        
    def __init__(self, filepath, dbtablename=None):
        self.filepath = filepath
    

    def kaggle_batch_generator(self, last_percents = 0.5, batch_size: int = 256):
        """
        Create a batch generator getting data from a csv file.

        Args:
            batch_size (int) : The nr of samples per batch. (Near 4k) the
            embed definitely stops working.
        """
        tweets = self.load_kaggle_tweets(last_percents)

        batch_amount = int(len(tweets)/batch_size)
        batch_gen = self.dataframe_batch_generator(tweets, batch_size) 

        return batch_gen, batch_amount


    def selected_batch_generator(self):
        """Create a batch generator getting data from a csv file."""
        tweets = self.load_selected_tweets(last_percents=1.0)

        batch_size = len(tweets)
        batch_amount = 1
        
        batch_gen = self.dataframe_batch_generator(tweets, batch_size) 

        return batch_gen, batch_amount


    def dataframe_batch_generator(self, data: pd.DataFrame, batch_size: int):
        """Used when getting data from a csv file.
        Usage:   a=batch_generator(df,n);   next(a); next(a)."""
        for i in range(0, len(data), batch_size):
            data_batch = data.iloc[i: i+batch_size]
            yield data_batch


    def load_kaggle_tweets(self, last_percents: int, dataset_encoding: str = 'ISO-8859-1'
                        ) -> pd.DataFrame:
        """Load from file the Kaggle dataset of 1.6mln tweets. Their sentiment
        is in {0, 4} set - rated as 0 (negative) and 4 (positive)."""
        csv_path = self.filepath

        records = pd.read_csv(csv_path, encoding = dataset_encoding)
        if last_percents != 1.0:
            amount = round(len(records)*last_percents) 
            records = records.tail(amount) 
        records= records.iloc[:,[0,-1]]
        records.columns = ['sentiment','content']
        records.sentiment = records.sentiment.map({0:0,4:1})

        # opinions = shuffle(opinions).reset_index(drop=True)
        print(f'Got {len(records)} tweets from Kaggle set.')
        return records


    def load_selected_tweets(self) -> pd.DataFrame:
        """
        Load selected rated tweets from a csv file. Cuts the dataset wrt. 
        sentiment, to outliers only: (0, tres)U(1-tres, 1). Thus leaving 
        only the extreme values of the scores (definitely positive or def. 
        negative).

        Args:
            sentiment_treshold (float) : It is a percentage of values to leave
            from boths sides. So it leaves twice the amount (eg. 0.3 -> 60% left).
        """
        csv_path = self.filepath
        old_limits = [-9, 9]  # Defines the scale in which sentiment is graded.
        old_range = np.sum(np.abs(old_limits))  # eg. (-9, 9) -> 18

        tweets = pd.read_csv(csv_path, skiprows=[0,1],
                        names=['tweet','sentiment'] )
        tweets.dropna(inplace=True)
        tweets['sentiment'] = tweets['sentiment'].apply(lambda x: x/old_range + 0.5)
        tweets['sentiment'] = tweets['sentiment'].apply(round)
        return tweets


    def save_scores(self, dir_path: str, filename: str, scores: pd.DataFrame, 
                    parameters: Dict[str, str]) -> None:
        """
        Save sentiment scores to a file.

        Args:
            dir_path (str) : The directory's path to the file being saved.
            filename (str) : The name of the file.
            scores (DataFrame) : Set of sentiment scores.
            parameters (dict[str, str]) : All parameters, printed out in the
            file as an information.
        Returns:
            None
        Raises:
        """
        export_file_path = os.path.join(dir_path, filename)
        prepare_df_info(scores, parameters)

        if not os.path.exists(os.path.dirname(export_file_path)):
            os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        scores.to_csv(export_file_path)
        print('Sentiment scores saved in the file:', export_file_path)
