import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
from datetime import timedelta
from typing import Tuple, List, Dict, Any


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


def get_most_recent_filepath(dirpath: str, name_splitter: str, case, 
                        files_only=True) -> Any:
    """
    Gives back the date of the file in the folder, that has the most 
    recent date.

    Args:
        dir_path (str): Full path to the directory.
        name_splitter (str): b
        case (): 'date_format' or 'int_format'. Specifies how the  
        timestamps in files' names are formatted.
    
    Returns:
        filepath (Any): A full path to the file which has in its name
        the most recent timestamp for a chosen directory. 
    Raises:
    """
    if case == 'date_format':
        most_recent = datetime.strptime('2000-01-01-15', '%Y-%m-%d-%H') 
        getdate = lambda fn, namespl: \
                datetime.strptime(fn.split(namespl)[0], '%Y-%m-%d-%H')
    elif case == 'int_format':
        most_recent = 0
        getdate = lambda fn, namespl: int(fn.split(namespl)[0])

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for filename in os.listdir(dirpath):
        file_validation = files_only and os.path.isfile(os.path.join(dirpath,filename))
        dir_validation = (not files_only) and (
                        not os.path.isfile(os.path.join(dirpath,filename)))
        if file_validation or dir_validation:
            date = getdate(filename, name_splitter)
            if date > most_recent:
                most_recent = date

    for fn in os.listdir(dirpath):
        if str(most_recent) in fn:
            filename = fn
    filepath = os.path.join(dirpath, filename)
    return filepath


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