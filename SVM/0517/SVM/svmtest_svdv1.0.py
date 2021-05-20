import pandas as pd
import csv
import time
import datetime
import mypreprocessing
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
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

def main():
    features=mypreprocessing.getfeaturesuser() #can change to getfeaturesuser
    labels = np.array(features['leaning'])
    features= features.drop('leaning', axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)
    features_num=np.sum(features) # 計算features的個數
    #---------------------------------------
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    svd.fit(features)
    new_features=svd.fit_transform(features)
    #print(new_features)
    #----------------------- 資料降維完畢
    new_features_train, new_features_test, labels_train, labels_test = train_test_split(new_features,labels,test_size=0.2,random_state=101)
    #----------- Do SVM
    from sklearn.svm import SVC
    model = SVC()
    model.fit(new_features_train,labels_train.ravel())
    predictions = model.predict(new_features_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(labels_test,predictions)) #Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    print('\n')
    print(classification_report(labels_test,predictions))
    print('\n')
    #----------- Do SVM Grid Search
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
    grid = GridSearchCV(SVC(),param_grid,verbose=0)
    grid.fit(new_features_train,labels_train.ravel())
    grid_predictions = grid.predict(new_features_test)
    print(confusion_matrix(labels_test,grid_predictions))
    print('\n')
    print(classification_report(labels_test,grid_predictions))

if __name__ == '__main__':
    main()
