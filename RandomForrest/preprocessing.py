import pandas as pd
import csv
import time
import psycopg2
import dbconnect
import numpy as np
from tqdm import tqdm
def getfeaturesmax():
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

    scorelist = finallist
    for row in tqdm(df.iterrows()): #each row is a tuple (index num, series)
        currentauthor = str(row[1]['author'])
        currentsubreddit = str(row[1]['subreddit'])
        currentleaning = str(row[1]['leaning'])
        currentscore = row[1]['score']
        #print(currentauthor)
        #print(currentsubreddit)
        scorelist.loc[currentauthor, currentsubreddit] += float(currentscore)
        #finallist.loc[currentauthor, 'author'] = currentauthor
        scorelist.loc[currentauthor, 'leaning'] = currentleaning
    
    participationlist = finallist
    for row in tqdm(df.iterrows()): #each row is a tuple (index num, series)
        currentauthor = str(row[1]['author'])
        currentsubreddit = str(row[1]['subreddit'])
        currentleaning = str(row[1]['leaning'])
        currentscore = row[1]['score']
        #print(currentauthor)
        #print(currentsubreddit)
        participationlist.loc[currentauthor, currentsubreddit] += 1
        #finallist.loc[currentauthor, 'author'] = currentauthor
        participationlist.loc[currentauthor, 'leaning'] = currentleaning

    participationlist = participationlist.drop(columns=['leaning'])
    for subreddit_name in subredditlist:
        participationlist.rename(columns = {subreddit_name: subreddit_name + '_participation'}, inplace = True)
   #print(participationlist)


    finallist = pd.concat([scorelist,participationlist])
    print("participation and score merged! ")
    

    demlist = finallist[(finallist.leaning == 'dem')]
    print("dem author count: ",len(demlist.index))
    replist = finallist[(finallist.leaning == 'rep')]
    print("rep author count: ",len(replist.index))
    print("attempting to balance so that dem/rep have same amount of author...")
    finallist =pd.concat([replist.head(min(len(replist.index),len(demlist.index))),demlist.head(min(len(replist.index),len(demlist.index)))])
    demlist = finallist[(finallist.leaning == 'dem')]
    print("dem author count: ",len(demlist.index))
    replist = finallist[(finallist.leaning == 'rep')]
    print("rep author count: ",len(replist.index))
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
    dbconnect.disconnect()
    return finallist

def getfeaturesuser():
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

    subscriberlist = dbconnect.getsubscribercount()
#    print(subscriberlist)
    for row in tqdm(df.iterrows()): #each row is a tuple (index num, series)
        currentauthor = str(row[1]['author'])
        currentsubreddit = str(row[1]['subreddit'])
        currentleaning = str(row[1]['leaning'])
        currentscore = row[1]['score']
        currentsubscriber=subscriberlist[currentsubreddit]
#        print(currentsubscriber)
        if currentsubscriber != 0:
            try:
                finallist.loc[currentauthor, currentsubreddit] += (float(currentscore) / currentsubscriber) * 100000
            except:
                print(currentsubreddit, currentsubscriber)
        #finallist.loc[currentauthor, 'author'] = currentauthor
        finallist.loc[currentauthor, 'leaning'] = currentleaning


#    print(finallist)
    finallist.reset_index(drop = True, inplace = True)
    finallist = finallist.sample(frac=1)
    finallist.reset_index(drop = True, inplace = True)
#    print(finallist['politics'])
#    print(finallist)
    finallist = finallist.drop(columns=['democrats','Republican'])
    deletedlist = dbconnect.getdeletedsubreddits()
    for delsub in deletedlist:
        try:
            finallist = finallist.drop(columns = delsub)
        except:
            continue

    dbconnect.disconnect()
    return finallist
