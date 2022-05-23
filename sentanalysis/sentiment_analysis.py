import copy
import time
import numpy as np
import pandas as pd
import os
from tqdm import tqdm   #progress bar
from datetime import datetime  #used for file name
from typing import List, Dict, Any, Tuple, Callable

import sentanalysis.nltk_utils as vdr
from sourcing.DBInterface import DBInterface
from MonteCarlo import monte_carlo


# @monte_carlo
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

    dbtablename = str(topic)
    if is_db_filtered:
        dbpath = dp.get('filtered_opinions_db','')
    else :
        dbpath = dp.get('opinions_db','')

    if topic == '':
        raise Exception('You have to specify the topic in the program parameters!')
    if dbpath == '':
        raise Exception('The chosen DB (filtered or not) not provided!')

    dbtable.set_dbpath(dbpath)
    start_date, end_date, timeperiods = \
        dbtable.specify_data_range(dbtablename, analysed_period, last_percents)

    scores = count_scores(MCparams, dbtable, dbtablename, analysed_period, 
                            timeperiods, start_date, end_date) 
    
    if not os.path.exists(os.path.dirname(dp['scores'])):
        os.makedirs(os.path.dirname(dp['scores']), exist_ok=True)
    filename = f"{int(time.time())}scores{analysed_period}.csv"
    save_scores(dp['scores'], filename, scores, MCparams)


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
    period_timedelta = get_period_timedelta(analysed_period)
    
    batch_gen = \
        dbtable.data_batch_generator(tablename, timeperiods, start_date,
                                    period_timedelta)

    scores = pd.DataFrame(columns=['mean','count'])
    interval_time = start_date
    with tqdm(total=timeperiods) as progress_bar:     # progress bar
        # while interval_time < end_date:     
        for _ in range(timeperiods):    
            opinions_batch = next(batch_gen)
            scores_period = vdr.evaluate_sentiment(opinions_batch, analysis_params)
            scores_period = pd.concat([scores_period, pd.Series(index=[start_date], 
                                                                dtype=np.float64)])
            scores_period = scores_period.groupby(pd.Grouper(freq=analysed_period, 
                                                            origin='start')
                                                ).agg(['mean','count'])
            scores_period.dropna(inplace=True)

            scores = pd.concat([scores, scores_period])
            progress_bar.update(1) 
    return scores
