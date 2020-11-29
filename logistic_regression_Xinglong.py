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

# Logistic regression
# Convert string to num & Replace Nan with mean value or 0
bicycle_data['Division'].fillna(value=bicycle_data['Division'].mean(), inplace=True)
bicycle_data['Hood_ID'].fillna(value=bicycle_data['Hood_ID'].mean(), inplace=True)
bicycle_data['Status'] = bicycle_data['Status'].map({'UNKNOWN':0,'STOLEN':0,'RECOVERED':1})
bicycle_data['Status'] = bicycle_data['Status'].fillna(0)

# Extract data
features = bicycle_data[['Division', 'Hood_ID']]
status = bicycle_data['Status']

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, status)

# Modeling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_features, train_labels)

# Test accuracy
print(model.score(train_features, train_labels))
print(model.score(test_features, test_labels))

# Test coefficients
print(model.coef_)

# Output
# Division ↓ = Status ↑
# Hood_ID ↓ = Status ↑

"""
@author: Xinglong Lu
"""
#Evaluate model using crossvalidation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, features, status, scoring='accuracy', cv=10)
print ("Cross validation :",scores)
print ("Cross validation :",scores.mean())

import numpy as np
probs = model.predict_proba(features)
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
Y_true =test_labels.values
Y_prob = np.array(prob_df['predict'])
preds=model.predict(test_features)
#Print the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_true, Y_prob)
print (confusion_matrix)
#Import scikit-learn metrics module for accuracy, recall, f1 and precision score calculation
from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(Y_true, preds))
print("Recall:",metrics.recall_score(Y_true, preds))
print("F1 score:",metrics.f1_score(Y_true, preds))
print("Precision score:",metrics.precision_score(Y_true, preds))
#Calculate ROC
from sklearn.metrics import roc_auc_score
#scores = np.array([0.1, 0.4, 0.35, 0.8])
print("ROC:",roc_auc_score(Y_true,probs))
fpr, tpr, thresholds = metrics.roc_curve(Y_true, probs, pos_label=2)


###Yoonseop 

# x =  df_g9_hour.index
import pickle 
x= ['18', '17', '12', '9', '19']
pickle.dump(x,open('C:/Users/User/Documents/pickle.pkl', 'wb'))
print("Models columns dumped!")

pickle.load(open('C:/Users/User/Documents/pickle.pkl' ,'rb')) 


