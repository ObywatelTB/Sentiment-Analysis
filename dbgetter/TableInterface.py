import abc
from typing import Tuple #, Callable, List, Dict, Any, 
from datetime import datetime
import pandas as pd


class TableInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'specify_data_range') and
                callable(subclass.specify_data_range) and
                hasattr(subclass, 'get_data_batch') and
                callable(subclass.get_data_batch) and
                NotImplemented)


    @abc.abstractmethod
    def specify_data_range(self, analysed_period: str, last_percents: bool,
                            ) -> Tuple[datetime, datetime, int]:
        """Set the time period of desired data"""
        raise NotImplementedError


    @abc.abstractmethod
    def get_data_batch(self, period_start: datetime, period_end: datetime
                        ) -> pd.DataFrame:
        """Get a batch of records within specified timestamps"""
        raise NotImplementedError
