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
from sklearn.ensemble import RandomForestRegressor
finallist = []
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

    subredditlist = ['leaning'] + list(subredditlist)

    finallist = pd.DataFrame (index = list(authorlist),columns= subredditlist)
    finallist = finallist.fillna(0)

    #print(df)

    for row in df.iterrows(): #each row is a tuple (index num, series)
        currentauthor = str(row[1]['author'])
        currentsubreddit = str(row[1]['subreddit'])
        currentleaning = str(row[1]['leaning'])
        #print(currentauthor)
        #print(currentsubreddit)
        finallist.loc[currentauthor, currentsubreddit] += 1
        #finallist.loc[currentauthor, 'author'] = currentauthor
        finallist.loc[currentauthor, 'leaning'] = currentleaning
    finallist.reset_index(drop = True, inplace = True)
    print(finallist)
    finallist = finallist.drop(columns=['democrats','Republican'])
    features = finallist

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

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    # Import the model we are using
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    test_labels = test_labels.astype(np.float)
#    print("test_labels: ", test_labels)
    errors = abs(predictions - test_labels)
#    print("errors: ",errors)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    print('accuracy based on MAE: ', 100-(round(np.mean(errors), 2)), '%')

    correctnum = list(filter(lambda number: number < 50, errors))

    print('accuracy only considering Correct/Incorrect: ', round(len(correctnum) / len(errors),4) * 100, '%')
# Get numerical feature importances
    importances = list(rf.feature_importances_)
# List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
    #print(feature_importances) #list of tuples
    filteredfeatures = list(filter(lambda number: number[1] > 0.0, feature_importances))
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in filteredfeatures];

    filteredlist = [x[0] for x in filteredfeatures]
#    print(filteredlist)
    finallist = finallist.filter(items=filteredlist+['leaning'])
    for index in range(10):
        print("####################################2nd Phase: ################################################\n")
        features = finallist
        labels = np.array(features['leaning'])
        features= features.drop('leaning', axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf.fit(train_features, train_labels);
        predictions = rf.predict(test_features)
        test_labels = test_labels.astype(np.float)
        errors = abs(predictions - test_labels)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        print('accuracy based on MAE: ', 100-(round(np.mean(errors), 2)), '%')
        correctnum = list(filter(lambda number: number < 50, errors))
        print('accuracy only considering Correct/Incorrect: ', round(len(correctnum) / len(errors),4) * 100, '%')
        importances = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        filteredfeatures = list(filter(lambda number: number[1] > 0.0, feature_importances))
    #    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in filteredfeatures];
        filteredlist = [x[0] for x in filteredfeatures]
        filteredlist = filteredlist[:len(filteredlist)-5]
        finallist = finallist.filter(items=filteredlist+['leaning'])

    dbconnect.disconnect()

if __name__ == '__main__':
    main()
