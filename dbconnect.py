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

def getdeminfo():
    cur = conn.cursor()
    insert_stmt = (
        "SELECT author,subreddit from testdemminusrep LIMIT 10000"
        
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    return result

def getrepinfo():
    cur = conn.cursor()
    insert_stmt = (
        "SELECT author,subreddit from testrepminusdem  LIMIT 10000"
        
        )
    cur.execute(insert_stmt)
    result = cur.fetchall()
    cur.close()
    return result

def main():
    connect()
    #insertdb('key5','testtitle', 'http://testurl', 'testauthor', 10, datetime.date(2012, 3, 23), '123456',11,'testperm','testflair')
    disconnect()
if __name__ == '__main__':
    main()
