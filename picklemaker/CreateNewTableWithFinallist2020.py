import pandas as pd
import csv
import time
import psycopg2
import dbconnect
import numpy as np
import re
from tqdm import tqdm
def main():
    print("getting")
    getfinallist()

def getfinallist():
    dbconnect.connect()
    mylist = dbconnect.getdeminfo()
    demdf = pd.DataFrame (mylist,columns=['author','subreddit','score'])
    demdf["leaning"]="dem"

    mylist = dbconnect.getrepinfo()
    repdf = pd.DataFrame (mylist,columns=['author','subreddit','score'])
    repdf["leaning"]="rep"

    frames = [demdf, repdf]
    df = pd.concat(frames)
    df = df.drop_duplicates()

    authorlist = df.author.unique()
    subredditlist = df.subreddit.unique()

    subredditlist = ['leaning'] + list(subredditlist)

    finallist = pd.DataFrame (index = list(authorlist),columns= subredditlist)
    finallist = finallist.fillna(0)

    for row in tqdm(df.iterrows()): #each row is a tuple (index num, series)
        currentauthor = str(row[1]['author'])
        currentsubreddit = str(row[1]['subreddit'])
        currentleaning = str(row[1]['leaning'])
        currentscore = row[1]['score']
        #print(currentauthor)
        #print(currentsubreddit)
        finallist.loc[currentauthor, currentsubreddit] += float(currentscore)
        #finallist.loc[currentauthor, 'author'] = currentauthor
        finallist.loc[currentauthor, 'leaning'] = currentleaning

    for column in finallist:
        if column == 'leaning':
            continue
        max = finallist[column].max()
        if max == 0:
            continue
        max = float(max)
        mylist = finallist[column].astype('float')
        finallist[column] = mylist.divide(other = max).round(3)

    finallist.reset_index(drop = True, inplace = True)
    finallist = finallist.sample(frac=1)
    finallist.reset_index(drop = True, inplace = True)
#    print(finallist['politics'])
#    print(finallist)
    finallist = finallist.drop(columns=['democrats','Republican'])
    print(finallist)
    finallist.to_pickle("../activitydata2020.pkl")
    dbconnect.disconnect()

'''
    CreateTableStatement = "CREATE TABLE UserHistoryTable ( "
    for col in finallist.columns:
        CreateTableStatement += ("r" + re.sub(r'[^\w\s]', '', col)  + " varchar(20), ")
    CreateTableStatement = CreateTableStatement[:-2]
    CreateTableStatement+=")"
    print(CreateTableStatement)

    print("Initializaing Creation SQL statement")
    dbconnect.executeSQL(CreateTableStatement)
'''

if __name__ == "__main__":
    main()
