"""
Sourcing
===
The package provides with services to get records to and from databases.
Also it provides with a module to get the .csv data.

It supports:
- the Firebase cloud DB
- local MySQL DB
- csv files

The intent is to upload the data (opinions on specified subjects) to the
cloud DB periodically. 
Then to get the data from the cloud and store it temporarily in the local
database in order to perform data processing on the specified records.

Gives access to classes:
- SQLiteService
- FirebaseService
- CSVService
"""

from sourcing.SQLiteService import SQLiteService
from sourcing.FirebaseService import FirebaseService
from sourcing.CSVService import CSVService