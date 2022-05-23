
import pandas as pd 
from datetime import datetime
from datetime import timedelta
import sqlite3
import re
from typing import List, Dict, Any, Tuple, Callable

from sourcing.DBInterface import DBInterface
from sourcing.utils import get_period_timedelta


class SQLiteService(DBInterface):

    def __init__(self, dbpath='', dbtablename = ''):
        self.dbpath = dbpath
        self.tablename = dbtablename
    

    def set_dbpath(self, dbpath: str, dbtablename = None) -> None: 
        """Initializes the values of the parameters."""
        self.dbpath = dbpath
        if dbtablename:
            self.tablename = dbtablename


    def setup_batch_generator(self, last_percents: float = 0.1,
                    batch_period: str = '1h') -> pd.DataFrame:
        """Seat up the generator allowing to read tweets from local SQL DB."""
        tablename = self.tablename
        start_date, end_date, timeperiods = \
            self.specify_data_range(tablename, batch_period, last_percents)
        period_timedelta = get_period_timedelta(batch_period)

        batch_gen = \
            self.data_batch_generator(tablename, timeperiods, start_date,
                                            period_timedelta)
        return batch_gen, timeperiods


    def specify_data_range(self, tablename: str, analysed_period: str,
                            last_percents: float = 1.0) -> Tuple[datetime, datetime, int]:
        """
        Give periods number, the first and the last timestamps and the number
        of records. Periods here are understood as batches of records sampled 
        with constant frequency.

        Args:
            DBparams (Dict[str,str]) : Parameters like dbpath, db table name.
            analysed_period (str): A period duration for batches of records 
            sampled with constant frequency.
            last_percents (float) : How much % of the entire DB table to give 
            back, that is the last records in terms of date.

        Returns:
            (int(periods), start_date, end_date, nr_of_records) : A tuple 
            with 4 values - number of periods, the first timestamp 
            of dataset, the last one, the nr of records.
        Raises:
        """
        dbpath = self.dbpath
        
        conn = sqlite3.connect(dbpath) 
        command_oldest = f"SELECT* from {tablename} order by date asc limit 1"
        command_newest = f"SELECT* from {tablename} order by date desc limit 1"
        command_count = f"SELECT Count(*) FROM {tablename}"
        oldest_record = pd.read_sql(command_oldest, conn)
        newest_record = pd.read_sql(command_newest, conn)
        nr_of_records = pd.read_sql(command_count, conn).values[0][0]
        conn.close()

        # timeframes:
        oldest_timef = datetime.strptime(oldest_record.iloc[0]['date'], 
                                                '%Y-%m-%d %H:%M:%S')
        newest_timef =  datetime.strptime(newest_record.iloc[0]['date'], 
                                                '%Y-%m-%d %H:%M:%S')

        if last_percents != 100.0:
            oldest_timef = newest_timef - (newest_timef-oldest_timef)*last_percents

        timeshift, timeunit = re.split('([0-9]+)', analysed_period)[1:]
        timeshift = int(timeshift)

        all_minutes = (newest_timef-oldest_timef).total_seconds()/60
        periods = all_minutes/timeshift
        if timeunit == 'd':
            periods = int(periods/(24*60))
        elif timeunit in ('h','H'):
            periods = int(periods/60)
        elif timeunit == 'min':
            periods = int(periods)

        start_date = oldest_timef.replace(microsecond=0, second=0, minute=0)
        end_date = newest_timef.replace(microsecond=0, second=0, minute=0)
        
        return start_date, end_date, periods #, nr_of_records


    def data_batch_generator(self, tablename: str, timeperiods: int, 
                        records_start: datetime,
                        period_timedelta: datetime) -> pd.DataFrame:
        """
        Give back data from a MySQL DB, for a specified time period.

        Args:
            DBparams (Dict[str,str]) : Parameters like dbpath, db table name.
            records_start (datetime) : The first timeframe marking the beginning 
            of the chosen period.
            period_end (datetime) : The last timeframe marking the end of the 
            chosen period.

        Returns:
            period_records (DataFrame) : Records of the time period got from the DB.
        Raises:
        """
        dbpath = self.dbpath

        period_start = records_start - period_timedelta
        for _ in range(timeperiods):
            period_start += period_timedelta
            period_end = period_start + period_timedelta - timedelta(seconds=1)

            period_start_str = datetime.strftime(period_start,'%Y-%m-%d %H:%M:%S')
            period_end_str = datetime.strftime(period_end,'%Y-%m-%d %H:%M:%S')

            conn = sqlite3.connect(dbpath)
            command = 'SELECT* FROM {} WHERE date BETWEEN \'{}\' AND \'{}\';'.format(
                                                tablename, period_start_str, period_end_str)
            period_records = pd.read_sql(command, conn)
            conn.close()

            period_records.index = period_records['date'].apply(
                                        lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S'))
            # period_records = period_records.set_index( pd.DatetimeIndex(period_records['date']) )
            period_records.drop(columns='date',inplace=True)
            period_records.sort_index(inplace=True)

            yield period_records


    def get_sql_table_lenght(self, dbpath: str):  
        """
        Get the nr of records in the SQL DB table.

        Args:
            dbpatch (str) : A filepath to the local DB.

        Returns:
        Raises:
        """
        conn = sqlite3.connect(dbpath)
        cur = conn.cursor()
        cur_result = cur.fetchone()
        conn.close()

        return cur_result[0]


    def get_all_records(self, table_name: str) -> pd.DataFrame:
        """
        Get all records of the DB table.

        Args:
            table_name (str) : Name of the DB table.
        
        Returns:
           
        Raises:
        """
        conn = sqlite3.connect(self.dbpath)
        records = pd.read_sql(f"SELECT* FROM {table_name}; ", conn)
        conn.close()

        records.drop('index', axis=1, inplace=True)
        records['date'] = records['date'].apply(lambda d: datetime.strptime(
                                                    d,'%Y-%m-%d %H:%M:%S'))
        records = records.set_index( pd.DatetimeIndex(records['date']) )
        records.drop('date',axis=1, inplace=True)
        records.sort_index(inplace=True)
        return records


    def delete_duplicates(self, tablename: str) -> None:
        """
        Delete duplicates from the given tabele.

        Args:
            table_name (str) : Name of the table where duplicates should be deleted.
        
        Returns:
            None
        Raises:
        """
        conn = sqlite3.connect(self.dbpath)
        cur = conn.cursor()
        command = 'DELETE FROM ' + tablename +' WHERE rowid NOT IN (SELECT \
                    min(rowid) FROM ' + tablename + ' GROUP BY status_id);'
        cur.execute(command)
        conn.commit()
        conn.close()

    
    def post_list_of_data(self, tablename: str, data_to_post: pd.DataFrame) -> None:
        """
        Write data to the given table of the database.

        Args:
            table_name (str) : Name of the DB table.
            data (DataFrame): Data to be written to the DB.
        Returns:
            None
        Raises:
        """
        conn = sqlite3.connect(self.dbpath)
        data_to_post.to_sql(tablename, conn, schema=None, if_exists='append')
        conn.close()

    