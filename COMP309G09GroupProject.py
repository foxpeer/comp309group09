# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:19:50 2020
@author: Group09
"""

'''
@author: Liping Wu
1. Data exploration: a complete review and analysis of the dataset 
1.1 Load the 'Bicycle_Thefts.csv' file into a dataframe and descibe data elements(columns),
 provide descriptions & types, ranges and values of elements as appropriate.
1.2 Statistical assessments including means, averages, correlations
1.3 Missing data evaluations – use pandas, NumPy and any other python packages
1.4 Graphs and visualizations – use pandas, matplotlib, seaborn, NumPy and any other python packages, you also can use power BI desktop.
'''

import pandas as pd
import os
path = "D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
print(fullpath)
data_bicycle_group09 = pd.read_csv(fullpath)
data_bicycle_group09.columns.values
data_bicycle_group09.shape
data_bicycle_group09.describe()
data_bicycle_group09.describe
data_bicycle_group09.dtypes
data_bicycle_group09.head(5)

data_bicycle_group09['Neighbourhood'].describe()
data_bicycle_group09['Premise_Type'].describe()
data_bicycle_group09['Location_Type'].describe()
data_bicycle_group09['Bike_Make'].describe()
data_bicycle_group09['Bike_Colour'].describe()
data_bicycle_group09['Cost_of_Bike'].describe()
data_bicycle_group09['Bike_Make'].describe()
data_bicycle_group09['Bike_Type'].describe()
data_bicycle_group09['Bike_Speed'].describe()







'''
@author: 
2. Data modelling: 
2.1 Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
2.2 Feature selection – use pandas and sci-kit learn.
2.3 Train, Test data splitting – use NumPy, sci-kit learn.
2.4 Managing imbalanced classes if needed.  Check here for info: https://elitedatascience.com/imbalanced-classes
'''



'''
@author: 
3. Predictive model building 
a.	Use logistic regression and decision trees as a minimum– use scikit learn
'''



'''
@author: 
4. Model scoring and evaluation
a.	Present results as scores, confusion matrices and ROC      - use sci-kit learn
b.	Select and recommend the best performing model 
'''




'''
@author: 
5. Deploying the model
c.	Using flask framework arrange to turn your selected machine-learning model into an API.
d.	Using pickle module arrange for Serialization & Deserialization of your model.
e.	Build a client to test your model API service. Use the test data, which was not previously used to train the module. You can use simple Jinja HTML templates with or without Java script, REACT or any other technology but at minimum use POSTMAN Client API.
'''



'''
@author: 
6. Prepare a report explaining your project and detailing all the assumptions, constraints you applied should have the following sections:
o	Executive summary (to be written once nearing the end of project work, should describe the problem/solution and key findings)
o	Overview of your solution(to be written once nearing the end of project work)
o	Data exploration and findings (dataset field descriptions, graphs, visualizations, tools and libraries used….etc.)
o	Feature selection (tools and techniques used, results of different combinations…etc.)
o	Data modeling (data cleaning strategy, results of data cleaning, data wrangling techniques, assumptions and constraints)
o	Model building (train/ test data, sampling, algorithms tested, results: confusion matrixes  ...etc.)
'''

