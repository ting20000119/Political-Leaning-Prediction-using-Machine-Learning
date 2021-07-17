from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import math
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
import pandas.util.testing as tm

def main():
    dffeatures = pd.read_pickle("./partinormalize.pkl")
    dffeatures = dffeatures.fillna(0)
    #dffeatures = dffeatures.head(200) #debug to lower time by lowering author (row) count
    features = dffeatures
    features.reset_index(drop=True, inplace=True) 
    features = features.sample(frac=1) 
    features.reset_index(drop=True, inplace=True) 
    labels = np.array(features['leaning'])
    features = features.drop('leaning', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    # ---------------------------------------開始降維 by SVD
    svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
    svd.fit(features)
    new_features = svd.transform(features)
    new_features = features
    #----------------------- 資料降維完畢
#    new_features_train, new_features_test, labels_train, labels_test = train_test_split(new_features,labels,test_size=0.2,random_state=101,stratify =labels)
    # ----------- Do SVM
    X = new_features
    labels = labels.tolist()
    for i in range(len(labels)):
        if labels[i] == "dem" :
            labels[i] = int(0)
        else :
            labels[i] = int(1)   
    y = np.array(labels)
    train_features, test_features, train_labels, test_labels = train_test_split(X,y,test_size=0.2)
    model = Sequential()
    model.add(layers.Dense(3, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
    results = model.fit(train_features, train_labels.ravel(), epochs=20, batch_size=100, validation_data=(test_features, test_labels))
    loss, accuracy = model.evaluate(test_features, test_labels)
    print('accuracy:', accuracy)
    predictions = model.predict(test_features)
    for i in range(len(predictions)):
        if predictions[i] < 0.5 :
            predictions[i] = int (0)
        else :
            predictions[i] = int (1)
    print(confusion_matrix(test_labels,predictions))
    print(classification_report(test_labels,predictions))
    


if __name__ == '__main__':
    main()
