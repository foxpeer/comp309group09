# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:03:08 2020

@author: Nayul Kim
"""

import pandas as pd
import numpy as np
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

## Data transformations
# Handling missing data 
# Convert string to num & Replace missimg value with mean or 0
bicycle_data['Division'].fillna(value=bicycle_data['Division'].mean(), inplace=True)
bicycle_data['Hood_ID'].fillna(value=bicycle_data['Hood_ID'].mean(), inplace=True)
bicycle_data['Status'] = bicycle_data['Status'].map({'UNKNOWN':0,'STOLEN':0,'RECOVERED':1})
bicycle_data['Status'] = bicycle_data['Status'].fillna(0)

# Categorical data management : Created dummy values 
data_dummies = pd.get_dummies(bicycle_data)

# Categorical data management : Remove unneccessary columns
data_dummies.drop(data_dummies.columns.difference(['Status','Division','Hood_ID']), 1, inplace=True)
print(data_dummies.head())

## Feature selection
features = data_dummies[['Division','Hood_ID']]
status = data_dummies['Status']

# Assigned X (features) and Y (target) / numpy.ndarray
X = features.values
Y = status.values
print((X.shape,Y.shape))
type(Y)
type(X)

## Train, test data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)