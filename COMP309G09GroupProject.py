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
import numpy as np
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


data_bicycle_group09 = pd.read_csv(fullpath)

# For Cost_of_Bike, fill missing with median
# check how many null before fill the missing 
print(data_bicycle_group09['Cost_of_Bike'].isnull().sum()) #1536
median = data_bicycle_group09['Cost_of_Bike'].median()
print(median)
# fill missing value with median
data_bicycle_group09['Cost_of_Bike'].fillna(median, inplace= True)
# check how many null after fill the missing 
print(data_bicycle_group09['Cost_of_Bike'].isnull().sum())  #0


# For Bike_Model, fill missing with "UNKNOWN"
# check how many null before fill the missing 
print(data_bicycle_group09['Bike_Model'].isnull().sum().sum()) #8140
data_bicycle_group09['Bike_Model'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(data_bicycle_group09['Bike_Model'].isnull().sum())  #0


# For Bike_Colour, fill missing with "UNKNOWN"
# check how many null before fill the missing 
print(data_bicycle_group09['Bike_Colour'].isnull().sum().sum()) #1729
data_bicycle_group09['Bike_Colour'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(data_bicycle_group09['Bike_Colour'].isnull().sum())  #0



from matplotlib import pyplot as plt
#create a scatterplot
fig_Premise_Cost = data_bicycle_group09.plot(kind='scatter',x='Premise_Type',y='Cost_of_Bike')
fig_Location_Cost = data_bicycle_group09.plot(kind='scatter',x='Location_Type',y='Cost_of_Bike')
fig_Division_Cost = data_bicycle_group09.plot(kind='scatter',x='Division',y='Cost_of_Bike')


# Save the scatter plot
figfilename = "ScatterPlot_Liping.pdf"
figfullpath = os.path.join(path, figfilename)
fig_Premise_Cost.figure.savefig(figfullpath)


 # Plot a histogram
import matplotlib.pyplot as plt
hist_year= plt.hist(data_bicycle_group09['Occurrence_Year'],bins=12)
plt.xlabel('Occurrence_Year')
plt.ylabel('and Stolen')
plt.title('Occurrence_Year and Stolen')

import matplotlib.pyplot as plt
hist_month= plt.hist(data_bicycle_group09['Occurrence_Month'],bins=12)
plt.xlabel('Occurrence_Month')
plt.ylabel('and Stolen')
plt.title('Occurrence_Month and Stolen')


 # Plot a histogram
import matplotlib.pyplot as plt
hist= plt.hist(data_bicycle_group09['Occurrence_Time'],bins=24)
plt.xlabel('Occurrence_Time')
plt.ylabel('and Stolen')
plt.title('Occurrence_Time and Stolen')

 # Plot a histogram
import matplotlib.pyplot as plt
hist_location= plt.hist(data_bicycle_group09['Location_Type'],bins=12)
plt.xlabel('Location_Type')
plt.ylabel('and Stolen')
plt.title('Location_Type and Stolen')

 # Plot a histogram
import matplotlib.pyplot as plt
hist_Premise= plt.hist(data_bicycle_group09['Premise_Type'],bins=12)
plt.xlabel('Premise_Type')
plt.ylabel('Stolen')
plt.title('Premise_Type and Stolen')

import matplotlib.pyplot as plt
hist_Division= plt.hist(data_bicycle_group09['Division'],bins=12)
plt.xlabel('Division')
plt.ylabel('Stolen')
plt.title('Division and Stolen')


# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle_group09['Occurrence_Day'])
plt.ylabel('Occurrence_Day')
plt.title('Box Plot of Occurrence_Day')

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle_group09['Cost_of_Bike'])
plt.ylabel('Cost_of_Bike')
plt.title('Box Plot of Cost_of_Bike')

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle_group09['X'])
plt.ylabel('X')
plt.title('Box Plot of X')

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle_group09['Y'])
plt.ylabel('Y')
plt.title('Box Plot of Y')













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

