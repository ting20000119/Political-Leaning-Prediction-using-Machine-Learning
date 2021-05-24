import pandas as pd
import requests
import json
import csv
import time
import datetime
import psycopg2
import dbconnect
import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
def main():
    dffeatures = pd.read_pickle("../activitydata.pkl")
    dffeatures = dffeatures.fillna(0)
    #dffeatures = preprocessing.getfeaturesmax()
    features = dffeatures #can change to getfeaturesuser for different normalization technique
    print(features.head())
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
    decompfeatures = decompSVD(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2,stratify = labels)
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
#        predictions = predictions.astype(np.int)
    #    print(predictions)
        # Calculate the absolute errors
    #    test_labels = test_labels.astype(np.int)
    #    print("test_labels: ", test_labels)
        AUC(test_labels, predictions)
        conf_mat = confusion_matrix(test_labels, predictions)
        print(conf_mat)
        correct = conf_mat[0][0]+conf_mat[1][1]
        total = correct + conf_mat[0][1]+conf_mat[1][0]
        print('accuracy only considering Correct/Incorrect: ',round(correct / total , 4) * 100, '%')
        #errors = predictions - test_labels
        #print("errors: ",errors)
        #correctnum = list(filter(lambda number: number == 0, errors))
        #print('accuracy only considering Correct/Incorrect: ', round(len(correctnum) / len(errors),4) * 100, '%')
    # Get numerical feature importances
        findimportance(rf,feature_list,dffeatures)

    dbconnect.disconnect()

def findimportance(rf,feature_list,dffeatures):
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
    showfeatures = feature_importances[:10] #list of tuples
    democratfeatures = dffeatures[dffeatures["leaning"] == 'dem']
    republicanfeatures = dffeatures[dffeatures["leaning"] == 'rep']
    newshowfeatures = []
    for (feature,importance) in showfeatures:
        demtotal=np.array(democratfeatures[feature]).sum()
        reptotal=np.array(republicanfeatures[feature]).sum()
#        print(feature,demtotal," ",reptotal)
        newshowfeatures.append((feature,importance,demtotal.round(2),reptotal.round(2)))

    [print('Variable: {:20} Importance: {}           Democrat total: {} Republican total: {}'.format(*pair)) for pair in newshowfeatures]
    filteredfeatures = [f[0] for f in filteredfeatures]
    filteredfeatures+=['leaning']

def decompPCA(features):
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
def decompSVD(features):
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    svd.fit(features)
    new_features=svd.fit_transform(features)
    return new_features
def AUC(labels_test,predictions):
    labels_true = labels_test.tolist()
    labels_true = [1 if x == 'dem' else 0 for x in labels_true]
    predictions = predictions.tolist()
    predictions = [1 if x == 'dem' else 0 for x in predictions]
    for i in range(len(labels_true)):
        labels_true[i]=int(labels_true[i])
    for i in range(len(predictions)):
        predictions[i]=int(predictions[i])
    #print(type(labels_test[0]))
    #print(predictions)
    fpr, tpr, thresholds = metrics.roc_curve(labels_true,predictions, pos_label=0)
    print("AUC :" , end = '')
    print(metrics.auc(fpr, tpr))
    print("\n")

if __name__ == '__main__':
    main()
