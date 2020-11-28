# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:19:50 2020
@author: Group09
https://docs.google.com/presentation/d/e/2PACX-1vQA18l8l1xjQdvxVzzFHxfnMdgYl_mLlue-V2UlPsDO4MUPDqINxTYk71HFHteBwym5qOo9b-UB6fmL/pub?start=true&loop=false&delayms=3000
"""

'''
@author: Liping
'''
'''
2. Data modelling: 
2.1 Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
2.2 Feature selection – use pandas and sci-kit learn.
2.3 Train, Test data splitting – use NumPy, sci-kit learn.
2.4 Managing imbalanced classes if needed.  Check here for info: https://elitedatascience.com/imbalanced-classes
'''


import pandas as pd
import os
import numpy as np
#change to your local 
path = "D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
print(fullpath)
df = pd.read_csv(fullpath)
pd.set_option('display.max_columns',20)
df.columns.values
df.shape   # 21584*30 columns
df.describe() 
df.describe
df.dtypes
df.head(5)

#

#check missing values of each column
df.isna().sum()

#For Bike model check how many null before fill the missing 
print(df['Bike_Model'].isnull().sum().sum()) #8140
df['Bike_Model'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(df['Bike_Model'].isnull().sum())  #0

# For Bike_Colour, fill missing with "UNKNOWN"
# check how many null before fill the missing 
print(df['Bike_Colour'].isnull().sum()) #1729
df['Bike_Colour'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(df['Bike_Colour'].isnull().sum())  #0

# fill missing Cost_of_Bike with median
median = df['Cost_of_Bike'].median()
# fill missing value or 0 with median 
df['Cost_of_Bike'].fillna(median, inplace= True)  
df['Cost_of_Bike'].replace(0, median, inplace= True)


# change 'Status' Column from Object to int
df['Status'] = df['Status'].map({'UNKNOWN':1,'STOLEN':1,'RECOVERED':0})
df.isna().sum()
#Fill null for Column Status
print(df['Status'].isnull().sum().sum()) #
df['Status'].fillna( 1, inplace= True)
# check how many null after fill the missing 
print(df['Status'].isnull().sum()) 


#from Occurrence_Date to get dayofweek
df["datetime"] = pd.to_datetime(df["Occurrence_Date"])
df['dayofweek'] =  df["datetime"].dt.dayofweek

# from Occurrence_Time to get hour of day
df["datehour"] = pd.to_datetime(df["Occurrence_Time"])
df['dayofhour'] =  df["datehour"].dt.hour
df['dayofhour'].dtype


# Drop some columns unwanted
df = df.drop(columns = ['X', 'Y', 'FID','Index_', 'event_unique_id','Occurrence_Date',"Occurrence_Time",'Neighbourhood','City','Location_Type',  'datehour', 'datetime'])
df = df.drop(columns = ['Primary_Offence'])
df = df.drop(columns = ['Occurrence_Year', 'Occurrence_Day'])
df = df.drop(columns = ['Division'])
df = df.drop(columns = ['dayofweek'])
df.columns.values
df.shape   #   (21584, 13)
df.describe() 
df.describe
df.dtypes
df.head(5)

'''
array(['Occurrence_Month', 'Premise_Type', 'Bike_Make', 'Bike_Model',
       'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike', 'Status',
       'Hood_ID', 'Lat', 'Long', 'dayofhour'], dtype=object)
'''
