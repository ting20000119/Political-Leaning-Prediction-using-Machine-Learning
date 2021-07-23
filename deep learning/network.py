from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
import pandas.util.testing as tm

def main():
    dffeatures = pd.read_pickle("./score.pkl")
    dffeatures = dffeatures.fillna(0)
    #print(dffeatures.info()) #to know the number of columns
    features = dffeatures
    features = features.sample(frac=1)
    labels = np.array(features['leaning'])
    features = features.drop('leaning', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    '''---------------------------------------開始降維 by SVD
    svd = TruncatedSVD(n_components=5000, n_iter=7, random_state=42)
    svd.fit(features)
    features = svd.transform(features)
    ----------------------- 資料降維完畢'''
    # ----------- Do SVM
    X = features
    labels = labels.tolist()
    for i in range(len(labels)):
        if labels[i] == "dem" :
            labels[i] = int(0)
        else :
            labels[i] = int(1)
    y = np.array(labels)
    train_features, test_features, train_labels, test_labels = train_test_split(X,y,test_size=0.2)
    model = Sequential()
    model.add(layers.Dense(5, input_dim=30385, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
    results = model.fit(train_features, train_labels.ravel(), epochs=20, batch_size=1000, validation_data=(test_features, test_labels))
    predictions = model.predict(test_features)
    loss, accuracy = model.evaluate(test_features, test_labels)
    print('\n')
    print('accuracy:', accuracy)
    print('loss',loss)
    print('\n')
    for i in range(len(predictions)):
        if predictions[i] < 0.5 :
            predictions[i] = int (0)
        else :
            predictions[i] = int (1)
    print(confusion_matrix(test_labels,predictions))
    print('\n')
    print(classification_report(test_labels,predictions))

if __name__ == '__main__':
    main()
