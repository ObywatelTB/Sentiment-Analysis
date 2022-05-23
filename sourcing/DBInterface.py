import abc
from typing import Tuple #, Callable, List, Dict, Any, 
from datetime import datetime
import pandas as pd


class DBInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'specify_data_range') and
                callable(subclass.specify_data_range) and
                hasattr(subclass, 'data_batch_generator') and
                callable(subclass.data_batch_generator) and
                hasattr(subclass, 'delete_duplicates') and
                callable(subclass.delete_duplicates) and
                hasattr(subclass, 'post_list_of_data') and
                callable(subclass.post_list_of_data) and
                NotImplemented)


    @abc.abstractmethod
    def set_dbpath(self, dbpath: str) -> None:
        """Initializes the values of the parameter."""
        raise NotImplementedError


    @abc.abstractmethod
    def specify_data_range(self, tablename: str, analysed_period: str, 
                        last_percents: bool, ) -> Tuple[datetime, datetime, int]:
        """Set the time period of desired data"""
        raise NotImplementedError


    @abc.abstractmethod
    def data_batch_generator(self, tablename: str, period_start: datetime, 
                        period_end: datetime) -> pd.DataFrame:
        """Get a batch of records within specified timestamps"""
        raise NotImplementedError


    @abc.abstractmethod
    def delete_duplicates(self, tablename: str):
        """Delete duplicate data from specified table"""
        raise NotImplementedError


    @abc.abstractmethod
    def post_list_of_data(self, tablename: str, data: pd.DataFrame):
        """pasts data from data frame to specified table"""
        raise NotImplementedError
