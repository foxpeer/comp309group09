# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:03:08 2020

@author: Nayul Kim
"""

import pandas as pd
import numpy as np
import os
#path = "C:/Users/User/Desktop/COMP309/group"
path = "D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/"
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
# Choose columns for using modeling
bicycle_data = bicycle_data[['Status', 'Division', 'Hood_ID']]
print(bicycle_data.head())

# Check missing values from columns
bicycle_data.isna().sum()

# Handling missing data 
# Convert string to num & Replace missimg value with mean or 0
print(bicycle_data['Division'].mean()) # mean of cost of bike column
bicycle_data['Division'].fillna(value=bicycle_data['Division'].mean(), inplace=True)
bicycle_data['Hood_ID'].fillna(value=bicycle_data['Hood_ID'].mean(), inplace=True)
bicycle_data['Status'] = bicycle_data['Status'].map({'UNKNOWN':0,'STOLEN':0,'RECOVERED':1})
bicycle_data['Status'] = bicycle_data['Status'].fillna(0)
bicycle_data['Status'].head(10)

# Categorical data management : Created dummy values 
data_dummies = pd.get_dummies(bicycle_data)
print(list(data_dummies.columns))
data_dummies.head()

## Feature selection
# Specifies the column from beginning to end except 
# for the target 'Status' in the data_dummies data
features = data_dummies.loc[:,'Division':'Hood_ID']
# Target = y
status = data_dummies['Status']

# Assigned X (features) and Y (target) / numpy.ndarray
X = features.values
Y = status.values
print((X.shape,Y.shape))
type(Y)
type(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))

# Test coefficients
print(model.coef_)