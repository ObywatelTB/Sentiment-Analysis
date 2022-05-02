import pandas as pd 
import pyrebase


class FirebaseService:

    def __init__(self):
        pass


    def set_firebase_config(self, config):
        """Initialize database based on a given configuration."""
        firebase = pyrebase.initialize_app(config)
        self.f_db = firebase.database()


    def get_data(self, tablename:str, string_date:str) -> pd.DataFrame:
        """
        Get tweets from firebase for specified currency and date.

        Args:
            string_date (str) : Date of the day for which records should
            be downloaded (string formatted as %Y%m%d).
            table_name (str) : DB table for which records should be got.
           
        Returns:
            data (DataFrame) : Records for a given day.
        
        Raises:
        """
        records = self.f_db.child(tablename + string_date).get()  
        data_inverted = pd.DataFrame(records.val())
        data = data_inverted.T
        return data


    def delete_from_firebase(self, tablename:str, chosen_day:str) -> None:
        """
        Delete from firebase records of specified tablename and date.

        Args:
            tablename (str) : DB table for which records should be deleted.
            chosen_day (str) : Date of the day for which records should
            be deleted (string formatted as %Y%m%d).
        
        Returns:
            None
        Raises:
        """
        self.f_db.child(tablename + chosen_day).remove() 