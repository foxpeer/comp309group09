# -*- coding: utf-8 -*-
"""
@author: Dayoon Lee
"""

# Import data
import pandas as pd
import os
# path = "C:/Users/User/Desktop/COMP309/group"
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

# Get predictors and target
colnames=bicycle_data.columns.values.tolist()
predictors=colnames[:4] # ['X', 'Y', 'FID', 'Index_']
target=colnames[4]  # event_unique_id

'''
bicycle_data['X'].nunique
bicycle_data['Y'].nunique
bicycle_data['FID'].value_counts()
print(bicycle_data['Index_'].unique)

import matplotlib.pyplot as plt
hist_FID= plt.hist(bicycle_data['Index_'],bins=18)
plt.xlabel('FID')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -FID vs Stolen')
'''


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
with open('path = "D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/bicycle_dtree.dot', 'w') as dotfile:
    export_graphviz(dt_bicycle, out_file = dotfile)
dotfile.close()
