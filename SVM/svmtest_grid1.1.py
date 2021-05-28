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
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
def AUC(labels_test,predictions):
    labels_true = labels_test.tolist()
    predictions = predictions.tolist()
    for i in range(len(labels_true)):
        labels_true[i]=int(labels_true[i])
    for i in range(len(predictions)):
        predictions[i]=int(predictions[i])
    fpr, tpr, thresholds = metrics.roc_curve(labels_true,predictions, pos_label=1)
    return metrics.auc(fpr, tpr)
    #print("AUC :" , end = '')
    #print(metrics.auc(fpr, tpr))
    #print("\n")

def adjusted_classes(y_scores, t):
    #print(t)
    #print("-------")
    #for i in y_scores:
    #    print(i)
    return [1 if y > t else 0 for y in y_scores]

def main():
    dffeatures = pd.read_pickle("../activitydata.pkl")
    dffeatures = dffeatures.fillna(0)
    #features=mypreprocessing.getfeaturesmax() #can change to getfeaturesuser
    features=dffeatures
    labels = np.array(features['leaning'])
    features= features.drop('leaning', axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)
    features_num=np.sum(features) # 計算features的個數
    #---------------------------------------開始降維 by SVD
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    svd.fit(features)
    new_features=svd.fit_transform(features)
    #print(new_features)
    #----------------------- 資料降維完畢
    new_features_train, new_features_test, labels_train, labels_test = train_test_split(new_features,labels,test_size=0.2,random_state=101,stratify =labels)
    #----------- Do SVM
    from sklearn.svm import SVC
    #----------- Do SVM Grid Search
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm, datasets
    param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
    grid = GridSearchCV(SVC(probability=True),param_grid,verbose=0)
    grid.fit(new_features_train,labels_train.ravel())
    grid_predictions = grid.predict(new_features_test)
    print("default Grid Search")
    print(confusion_matrix(labels_test,grid_predictions)) #Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    print('\n')
    print(classification_report(labels_test,grid_predictions))
    print('\n')
    y_scores = grid.predict_proba(new_features_test)[:,1]
    for i in np.arange(0.4,0.5,0.0001):
        print(i)
        y_pred_adj = adjusted_classes(y_scores,i)
        pred_adj=np.array(y_pred_adj)
        pred_adj = pred_adj.tolist()
        for i in range(len(pred_adj)):
            for i in range(len(pred_adj)):
                if pred_adj[i]==1:
                    pred_adj[i]="rep"
                elif pred_adj[i]==0:
                    pred_adj[i]="dem"
        print(confusion_matrix(labels_test,pred_adj))
        print(classification_report(labels_test,pred_adj))

    


if __name__ == '__main__':
    main()
