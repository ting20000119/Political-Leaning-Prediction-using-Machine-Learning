import pandas as pd
import requests
import json
import csv
import time
import datetime
import psycopg2
import dbconnect
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython import get_ipython
ipy = get_ipython()
import math

#import Charts

def main():
    dbconnect.connect()
    mylist = dbconnect.getdeminfo()
    demdf = pd.DataFrame (mylist,columns=['author','subreddit'])
    demdf["leaning"]=float(0)

    mylist = dbconnect.getrepinfo()
    repdf = pd.DataFrame (mylist,columns=['author','subreddit'])
    repdf["leaning"]=float(100)

    frames = [demdf, repdf]
    df = pd.concat(frames)
    df = df.drop_duplicates()

    authorlist = df.author.unique()
    subredditlist = df.subreddit.unique()

    subredditlist = ['author','leaning'] + list(subredditlist)

    finallist = pd.DataFrame (index = list(authorlist),columns= subredditlist)
    finallist = finallist.fillna(0)
    #print(finallist)

    #print(df)
    
    for row in df.iterrows() : #each row is a tuple (index num, series)
        currentauthor = str(row[1]['author'])
        currentsubreddit = str(row[1]['subreddit'])
        currentleaning = str(row[1]['leaning'])
        #print(currentauthor)
        #print(currentsubreddit)
        finallist.loc[currentauthor, currentsubreddit] = 1
        #finallist.loc[currentauthor, 'author'] = currentauthor
        finallist.loc[currentauthor, 'leaning'] = currentleaning
    finallist.reset_index(drop = True, inplace = True) ##solve could not string convert float
    #print(finallist)
    finallist = finallist.drop(columns=['democrats','Republican'])
    features = finallist
    
    #_____
    
    labels = np.array(features['leaning'])
    features= features.drop('leaning', axis = 1)
    feature_list = list(features.columns)
    #print(features)
    features = np.array(features)
    #X = features
    #y = labels
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3,random_state=101)
    from sklearn.svm import SVC
    model = SVC()
    #print(type(X_train),type(y_train))
    model.fit(features_train,labels_train.ravel())
    #model.fit(X_train,y_train.ravel())
    predictions = model.predict(features_test)
    #print(predictions)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(labels_test,predictions))
    print('\n')
    print(classification_report(labels_test,predictions))
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
    grid = GridSearchCV(SVC(),param_grid,verbose=0)
    #grid.fit(X_train,y_train.ravel())
    grid.fit(features_train,labels_train.ravel())
    grid_predictions = grid.predict(features_test)
    #print(grid_predictions)
    print(confusion_matrix(labels_test,grid_predictions))
    print('\n')
    print(classification_report(labels_test,grid_predictions))
    '''labels_test =  labels_test.astype(np.float)
    grid_predictions =  grid_predictions.astype(np.float)
    errors = abs(grid_predictions - labels_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    print('accuracy based on MAE: ', 100-(round(np.mean(errors), 2)), '%')
    correctnum = list(filter(lambda number: number < 50, errors))
    print('accuracy only considering Correct/Incorrect: ', round(len(correctnum) / len(errors),4) * 100, '%')'''

    

    dbconnect.disconnect()
if __name__ == '__main__':
    main()
