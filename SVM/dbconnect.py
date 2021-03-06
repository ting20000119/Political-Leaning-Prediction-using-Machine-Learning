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
    cur = conn.cursor()
    insert_stmt = (
        "SELECT author,subreddit,score from testdemminusrep where (epoch > 1577883600) LIMIT 10000"
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    return result

def getrepinfo():
    cur = conn.cursor()
    insert_stmt = (
        "SELECT author,subreddit,score from testrepminusdem where (epoch > 1577883600) LIMIT 10000"
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    return result

def main():
    connect()
    insertdb('key5','testtitle', 'http://testurl', 'testauthor', 10, datetime.date(2012, 3, 23), '123456',11,'testperm','testflair')
    disconnect()
if __name__ == '__main__':
    main()
