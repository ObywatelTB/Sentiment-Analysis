"""
DbGetter
===
The package provides services to get records to and from databases.
It supports:
- the Firebase cloud DB 
- local MySQL

The intent is to upload the data (opinions on specified subjects) to the
cloud DB periodically. 
Then to get the data from the cloud and store it temporarily on the local
machine in order to perform data processing on samples of the data.

Gives access to classes:
- from dbgetter.SQLtable.SQLtable

"""

# from dbgetter.firebase_support import fun1

from dbservices.SQLiteService import SQLiteService