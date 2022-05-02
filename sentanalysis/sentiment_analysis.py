import copy
import time
import numpy as np
import pandas as pd
import os
from tqdm import tqdm   #progress bar
from datetime import datetime  #used for file name
from datetime import timedelta
import re
from typing import List, Dict, Any, Tuple, Callable

import sentanalysis.nltk_utils as vdr
from dbservices.DBInterface import DBInterface


def analyse_opinions(MCparams: Dict[str, Any], dp: Dict[str, str],
        dbtable: DBInterface, last_percents: float = 100.0) -> None:
    """
    Perform a sentiment analysis on a dataset of opinions, save the 
    obtained scores.

    Args:
        MCparams (dict[str,Any]) : Monte Carlo parameters. These are the
        arguments that change with each MC method execution. They specify 
        eg. the topic and how the sentiment is evaluated.
        dp (dict[str,str]) : Directories paths.
        dbtable (TableInterface) : An object providing access a DB table, 
        and functions getting DB records.
        last_percents (float) : How much % of the entire DB table to give 
        back, that is the last records in terms of date.
    Returns:
        None
    Raises:
        Exceptions if the topic or DB path are not specified.
    """
    topic = MCparams.get('topic', '')       # needed for the DB table name
    analysed_period = MCparams.get('analysed_period', '2h')
    is_db_filtered = MCparams.get('is_db_filtered', True)

    dbtablename = f'{topic}_table'
    if is_db_filtered:
        dbpath = dp.get('filtered_opinions_db','')
    else :
        dbpath = dp.get('opinions_db','')

    if topic == '':
        raise Exception('You have to specify the topic in the program parameters!')
    if dbpath == '':
        raise Exception('The chosen DB (filtered or not) not provided!')

    dbtable.setvalues(dbpath)
    start_date, end_date, timeperiods = dbtable.specify_data_range(analysed_period, 
                                                                    dbtablename,
                                                                    last_percents)
    scores = count_scores(MCparams, dbtable, dbtablename, analysed_period, 
                            timeperiods, start_date, end_date) 
    
    if not os.path.exists(os.path.dirname(dp['scores'])):
        os.makedirs(os.path.dirname(dp['scores']), exist_ok=True)
    filename = f"{int(time.time())}scores{analysed_period}.csv"
    save_scores(dp['scores'], filename, scores, MCparams)


#===Encapsulated functions
def count_scores(analysis_params: Dict[str, Any], dbtable: DBInterface, tablename:str,
                analysed_period: str, timeperiods: int, start_date: datetime, 
                end_date: datetime) -> pd.DataFrame:
    """
    Create a DataFrame of sentiment analysis scores, from given opinions
    on a specific topic.

    Args:
        analysis_params (dict[str, Any]) : Program parameters used in the 
        sentiment evaluation (treshold, is_score_binary, etc.).
        dbtable (TableInterface) : An object providing access to a DB 
        table, and functions to get DB records.
        tablename (str) : Name of chosen DB table.
        analysed_period (str) : A parameter in the xn form, where x is 
        number, n is min/h/H/d.
        timeperiods (int) : A number of time periods into which we split 
        the DB table records, in order to compute faster.
        start_date (datetime) : The first timeframe marking the beginning 
        of the chosen dataset.
        end_date (datetime) : The last timeframe marking the end of the 
        chosen dataset.
    Returns:
        scores (DataFrame) : A dataframe with evaluated sentiment scores.
    Raises:
    """
    datatime_shift = get_period_timedelta(analysed_period)
    
    interval_time = start_date
    scores = pd.DataFrame(columns=['mean','count'])
    with tqdm(total=timeperiods) as progress_bar:     # progress bar
        while interval_time < end_date:    
            last_timestamp = interval_time + datatime_shift - timedelta(seconds=1)
            interval_of_opinions = dbtable.get_data_batch(interval_time, tablename, 
                                                            last_timestamp)

            scores_period = vdr.evaluate_sentiment(interval_of_opinions, analysis_params)
            scores_period = pd.concat([scores_period, pd.Series(index=[start_date], 
                                                                dtype=np.float64)])
            scores_period = scores_period.groupby(pd.Grouper(freq=analysed_period, 
                                                            origin='start')
                                                ).agg(['mean','count'])
            scores_period.dropna(inplace=True)

            scores = pd.concat([scores, scores_period])
            interval_time += datatime_shift
            progress_bar.update(1) 
    return scores


def get_period_timedelta(analysed_period: str) -> timedelta:
    """
    Give back 'analysed period' in the timedelta format.

    Args:
        analysed_period (str) : The parameter in the xn form, where x is
        number, n is min/h/H/d.
    Returns:
        datatime_shift (timedelta) : The equivalent of the input conversed 
        to the needed type.
    Raises:
    """
    timeshift, timeunit = re.split('([0-9]+)', analysed_period)[1:]
    timeshift = int(timeshift)

    if timeunit == 'd':
        period_timedelta = timedelta(days=timeshift)
    elif timeunit in ('h','H'):
        period_timedelta = timedelta(hours=timeshift) 
    else:
        period_timedelta = timedelta(minutes=timeshift)

    return period_timedelta


def save_scores(dir_path: str, filename: str, scores: pd.DataFrame, 
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


def prepare_df_info(scores: pd.DataFrame, const_params: Dict[str,str]
                    ) -> None:
    """
    Modify the DataFrame, so that it contains a column with enlisted MC 
    parameters.

    Args:
        scores (DataFrame) : A df with scores of opinions' sentiment.
        const_params (dict[str, str]) : The program parameters that do
        not change during one analysis execution. Put in the info column.
    Returns:
        None
    Raises:
    """
    info_list = []
    for key in const_params:
        info_list.append(f'{key}:{const_params[key]}')
    if len(info_list) < len(scores):    # Param length lesser than amount of periods.
        scores['info'] = np.nan         # Creates a column full of NaN.
        scores['info'].iloc[0:len(info_list)] = info_list
    else:                               # Param. length greater than periods amount.
        rows_to_add = len(info_list) - len(scores)
        for r in range(rows_to_add):
            scores = scores.append(pd.Series(),ignore_index=True)
        scores['info'] = pd.Series(info_list)