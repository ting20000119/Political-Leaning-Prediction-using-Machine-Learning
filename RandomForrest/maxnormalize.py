import pandas as pd
import requests
import json
import csv
import time
import datetime
import psycopg2
import dbconnect
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
finallist = []
def main():
    dbconnect.connect()
    mylist = dbconnect.getdeminfo()
    demdf = pd.DataFrame (mylist,columns=['author','subreddit','score'])
    demdf["leaning"]=1

    mylist = dbconnect.getrepinfo()
    repdf = pd.DataFrame (mylist,columns=['author','subreddit','score'])
    repdf["leaning"]=0

    frames = [demdf, repdf]
    df = pd.concat(frames)
    df = df.drop_duplicates()

    authorlist = df.author.unique()
    subredditlist = df.subreddit.unique()

    subredditlist = ['leaning'] + list(subredditlist)

    finallist = pd.DataFrame (index = list(authorlist),columns= subredditlist)
    finallist = finallist.fillna(0)

    for row in df.iterrows(): #each row is a tuple (index num, series)
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
        finallist[column] = mylist.divide(other = max).round(3) * 1000



    print(finallist)
    finallist.reset_index(drop = True, inplace = True)
    finallist = finallist.sample(frac=1)
    finallist.reset_index(drop = True, inplace = True)
#    print(finallist['politics'])
#    print(finallist)
    finallist = finallist.drop(columns=['democrats','Republican'])
    features = finallist
    while(True):
        ####start learning
        # Labels are the values we want to predict
        labels = np.array(features['leaning'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        features= features.drop('leaning', axis = 1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        # Convert to numpy array
        features = np.array(features)
        decompfeatures = decomp(features)

        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)
        # Import the model we are using
        # Instantiate model with 1000 decision trees
        test_estimators = [150]
        for estimator in test_estimators:
            rf = RandomForestClassifier(n_estimators = estimator, random_state = 42)
        #    RandomForestClassifier.get_params()
            # Train the model on training data
            rf.fit(train_features, train_labels);

            # Use the forest's predict method on the test data

            predictions = rf.predict(test_features)
            predictions = predictions.astype(np.int)
        #    print(predictions)
            # Calculate the absolute errors
            test_labels = test_labels.astype(np.int)
        #    print("test_labels: ", test_labels)
            errors = predictions - test_labels
        #    print("errors: ",errors)
            correctnum = list(filter(lambda number: number == 0, errors))
            print('accuracy only considering Correct/Incorrect: ', round(len(correctnum) / len(errors),4) * 100, '%')
        # Get numerical feature importances
            features = findimportance(rf,feature_list,finallist)

    dbconnect.disconnect()

def findimportance(rf,feature_list,finallist):
# Get numerical feature importances
    importances = list(rf.feature_importances_)
# List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
    #print(feature_importances) #list of tuples
#        filteredfeatures = list(filter(lambda number: number[1] > 0.0, feature_importances))

    filteredfeatures = feature_importances[:round(len(feature_importances)/1.1)]
    print("number of features left: ",len(feature_importances))
#            filteredfeatures = list(filter(lambda number: number[0], filteredfeatures))
    showfeatures = feature_importances[:10]
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in showfeatures]
    filteredfeatures = [f[0] for f in filteredfeatures]
    filteredfeatures+=['leaning']
    features = finallist[filteredfeatures]
    return features

def decomp(features):
    pca=decomposition.PCA()
    pca.fit(features) # 用PCA降維
    variances=pca.explained_variance_ #可以理解成該特徵的重要性，數字非常小，即特徵不重要
    #print(variances)  #列印降維後的新特徵
    #print("\n")
    thresh=0.05 # 故而可以為重要性設定一個閾值，小於該閾值的認為該特徵不重要，可刪除
    useful_features=variances > thresh
#    print(useful_features) # 標記為True的表示重要特徵，要保留，False則刪除
    useful_features_num=np.sum(useful_features) # 計算True的個數
    print("number of features after decomposition: ",useful_features_num)
    pca.n_components=useful_features_num # 即設定PCA的新特徵數量為n_components
    new_features=pca.fit_transform(features)
#    print(new_features)
    return new_features
if __name__ == '__main__':
    main()
