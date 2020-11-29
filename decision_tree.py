# -*- coding: utf-8 -*-
"""
@author: Dayoon Lee
"""

# Import data
import pandas as pd
import os
path = "C:/Users/User/Desktop/COMP309/group"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
bicycle_data = pd.read_csv(fullpath,sep=',')

# Print data
print(bicycle_data.columns.values)
print(bicycle_data.shape)
print(bicycle_data.describe())
print(bicycle_data.dtypes) 
print(bicycle_data.head(5))

# Get predictors and target
colnames=bicycle_data.columns.values.tolist()
predictors=[colnames[i] for i in (11,22)]
target=colnames[21]
print(predictors)
print(target)

import numpy as np
bicycle_data['is_train'] = np.random.uniform(0, 1, len(bicycle_data)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
train, test = bicycle_data[bicycle_data['is_train']==True], bicycle_data[bicycle_data['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

#Create a decision tree and fit
from sklearn.tree import DecisionTreeClassifier
dt_bicycle = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_bicycle.fit(train[predictors], train[target])
preds=dt_bicycle.predict(test[predictors])
pd.crosstab(test['Status'],preds,rownames=['Actual'],colnames=['Predictions'])
from sklearn.tree import export_graphviz
with open('C:/Users/User/Desktop/bicycle_dtree.dot', 'w') as dotfile:
    export_graphviz(dt_bicycle, out_file = dotfile)
dotfile.close()

"""
@author: Xinglong Lu
"""
#Evaluate model using crossvalidation
from sklearn.model_selection import KFold
crossvalidation=KFold(n_splits=10,shuffle=True,random_state=1)
from sklearn.model_selection import cross_val_score
score=np.mean(cross_val_score(dt_bicycle_model,trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print("Cross validation:",score)

#Print the confusion matrix
from sklearn.metrics import confusion_matrix
labels = bicycle_data[target].unique()
print(labels)
print("Confusion matrix \n" , confusion_matrix(testY, preds, labels))

#Import scikit-learn metrics module for accuracy, recall, f1 and precision score calculation
from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(testY, preds))
print("Recall:",metrics.recall_score(testY, preds))
print("F1 score:",metrics.f1_score(testY, preds))
print("Precision score:",metrics.precision_score(testY, preds))

#Calculate ROC
from sklearn.metrics import roc_auc_score
probs = dt_bicycle_model.predict_proba(testX)
#scores = np.array([0.1, 0.4, 0.35, 0.8])
print("ROC:",roc_auc_score(testY,probs))
fpr, tpr, thresholds = metrics.roc_curve(testY, probs, pos_label=2)

