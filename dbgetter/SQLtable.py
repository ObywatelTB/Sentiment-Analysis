
import pandas as pd 
from datetime import datetime
from datetime import timedelta
import sqlite3
import re
from typing import List, Dict, Any, Tuple, Callable

from dbgetter.TableInterface import TableInterface


class SQLtable(TableInterface):

    def __init__(self):
        pass
    

    def setvalues(self, dbpath: str, tablename: str) -> None: 
        """Initializes the values of the parameters."""
        self.dbpath = dbpath
        self.tablename = tablename


    def specify_data_range(self, analysed_period: str,
                            last_percents: float = 100.0) -> Tuple[datetime, datetime, int]:
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
            with of 4 parameters - number of periods, the first timestamp 
            of dataset, the last one, the nr of records.
        Raises:
        """
        dbpath = self.dbpath
        tablename = self.tablename
        
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
        # do dokonczenia-^

        timeshift, timeunit = re.split('([0-9]+)', analysed_period)[1:]
        timeshift = int(timeshift)

        all_minutes = (newest_timef-oldest_timef).total_seconds()/60
        periods = all_minutes/timeshift
        if timeunit == 'd':
            periods = int(periods/(24*60))
        elif timeunit in ('h','H'):
            periods = int(periods/60)

        start_date = oldest_timef.replace(microsecond=0, second=0, minute=0)
        end_date = newest_timef.replace(microsecond=0, second=0, minute=0)
        
        return start_date, end_date, periods #, nr_of_records


    def get_data_batch(self, period_start: datetime, 
                        period_end: datetime) -> pd.DataFrame:
        """
        Give back data from a MySQL DB, for a specified time period.

        Args:
            DBparams (Dict[str,str]) : Parameters like dbpath, db table name.
            period_start (datetime) : The first timeframe marking the beginning 
            of the chosen period.
            period_end (datetime) : The last timeframe marking the end of the 
            chosen period.

        Returns:
            period_records (DataFrame) : Records of the time period got from the DB.
        Raises:
        """
        dbpath = self.dbpath
        tablename = self.tablename

        period_start = datetime.strftime(period_start,'%Y-%m-%d %H:%M:%S')
        period_end = datetime.strftime(period_end,'%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(dbpath)
        command = f"""SELECT* FROM {tablename} WHERE date BETWEEN \'{period_start}\' 
                    AND \'{period_end}\';"""
        period_records = pd.read_sql(command, conn)
        conn.close()

        period_records['date'] = period_records['date'].apply(
                                    lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S'))
        period_records = period_records.set_index( pd.DatetimeIndex(period_records['date']) )
        period_records.sort_index(inplace=True)

        return period_records


    def get_sql_table_lenght(dbpath: str):  
        """
        Get the nr of records in the SQL DB table.

        Args:
            dbpatch (str) : A filepath to the local DB.

        Returns:
        Raises:
        """
        conn = sqlite3.connect(dbpath)
        cur = conn.cursor()
        data = cur.execute("SELECT COUNT(*) FROM tweets_table") 
        cur_result = cur.fetchone()
        conn.close()
        return cur_result[0]


    def read_all_opinions(db_name: str, table_name: str) -> pd.DataFrame:
        """
        Get all records of the DB table.
        """
        conn = sqlite3.connect(db_name)
        opinions = pd.read_sql(f"SELECT* FROM {table_name}; ", conn)
        conn.close()
        opinions.drop('index', axis=1, inplace=True)
        opinions['date'] = opinions['date'].apply(lambda d: datetime.strptime(
                                                    d,'%Y-%m-%d %H:%M:%S'))
        opinions = opinions.set_index( pd.DatetimeIndex(opinions['date']) )
        opinions.drop('date',axis=1, inplace=True)
        opinions.sort_index(inplace=True)
        return opinions