#!/usr/bin/python
import psycopg2
from config import config
import datetime
conn = None
def connect():
    """ Connect to the PostgreSQL database server """
    params = config()
    print('Connecting to the PostgreSQL database...')
    global conn
    conn = psycopg2.connect(**params)

def disconnect():
    global conn
    conn.close()
    print('Database connection closed.')

def getsubscribercount():
    cur = conn.cursor()
    insert_stmt = (
        "SELECT * from subreddit_infonew"
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    return dict(result)
    cur.close()

def getdeletedsubreddits():
    cur = conn.cursor()
    insert_stmt = (
        "SELECT subreddit from subreddit_infonew where subscribers = 0"
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    result = [x for (x,) in result]
#    print(result)
    return result

def getdeminfo():
    print("collecting dem info")
    cur = conn.cursor()
    insert_stmt = (
        "SELECT author,subreddit,score from demactivity2020 where (epoch > 1577883600)"
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    print("dem info collected")
    return result

def getrepinfo():
    print("collecting rep info")
    cur = conn.cursor()
    insert_stmt = (
        "SELECT author,subreddit,score from repactivity2020 where (epoch > 1577883600)"
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    print("rep info collected")
    return result
