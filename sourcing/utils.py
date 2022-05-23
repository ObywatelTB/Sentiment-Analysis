import numpy as np
import pandas as pd
import re
from datetime import timedelta
from typing import Tuple, List, Dict, Any

from sourcing.CSVService import load_kaggle_tweets
        

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
    elif timeunit == 'min':
        period_timedelta = timedelta(minutes=timeshift)
    else:
        raise('Wrong period!')

    return period_timedelta


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