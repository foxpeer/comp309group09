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
