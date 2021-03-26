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
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from sklearn import svm


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
    labels = np.array(features['leaning'])
    features= features.drop('leaning', axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)
    print(features)
    features_num=np.sum(features) # 計算features的個數
    print(features_num)
    print("\n")
    #---------------------------------------
    from sklearn import decomposition
    pca=decomposition.PCA()
    pca.fit(features) # 用PCA降維
    variances=pca.explained_variance_ #可以理解成該特徵的重要性，數字非常小，即特徵不重要
    #print(variances)  #列印降維後的新特徵
    #print("\n")
    thresh=0.04 # 故而可以為重要性設定一個閾值，小於該閾值的認為該特徵不重要，可刪除
    useful_features=variances > thresh
    #print(useful_features) # 標記為True的表示重要特徵，要保留，False則刪除
    useful_features_num=np.sum(useful_features) # 計算True的個數
    print(useful_features_num)
    pca.n_components=useful_features_num # 即設定PCA的新特徵數量為n_components
    new_features=pca.fit_transform(features)
    print(new_features)
   
    #----------------------- 資料降維完畢
    #X = features
    #y = labels
    new_features_train, new_features_test, labels_train, labels_test = train_test_split(new_features,labels,test_size=0.2,random_state=101)
    from sklearn.svm import SVC
    model = SVC()
    #print(type(X_train),type(y_train))
    model.fit(new_features_train,labels_train.ravel())
    predictions = model.predict(new_features_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(labels_test,predictions)) #Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    print('\n')
    print(classification_report(labels_test,predictions))
    print('\n')
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
    grid = GridSearchCV(SVC(),param_grid,verbose=0)
    #grid.fit(X_train,y_train.ravel())
    grid.fit(new_features_train,labels_train.ravel())
    grid_predictions = grid.predict(new_features_test)
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
