#!/usr/bin/env python
# coding: utf-8

# The Sparks Foundation Internship Program 
# Domain (Technical):
# Data Science and Business Analytics 
# Intern Name: Adwait Deshpande
# Batch - July 2021
# 

# Task 1 : Prediction using Supervised ML
#  To Predict the percentage of an student based on the no. of study hours.
# 
# 

# Problem Statement :What will be predicted score if a student studies for 9.25 hrs/ day?

# Importing Python Libraries 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading Data from the given dataset

# In[4]:


data = "http://bit.ly/w-data"
df = pd.read_csv(data)
print("Data Imported Successfully")
df.head(10)


# Describing Data ( counting mean, std, min, max etc of the dataset )

# In[8]:


df.isnull().sum()


# Data Visualization

# In[9]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# 
# Preparing Data

# In[10]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# Spliting Data Using Scikit-Learn's built-in train_test_split()

# In[11]:


from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(" X_train = ",X_train.shape)
print(" X_test = ",X_test.shape)
print(" Y_train = ",Y_train.shape)
print(" Y_test = ",Y_test.shape)


# Training Algorithm

# In[12]:


from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(X_train, Y_train)


# In[13]:


print("Coefficient = ",reg.coef_)
print("Intercept = ",reg.intercept_)


# In[14]:


# Plotting the regression line
line = reg.coef_*X+reg.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# Making Prediction

# In[15]:


print(X_test) # Testing data - In Hours
Y_pred = reg.predict(X_test) # Predicting the scores


# In[16]:


print(Y_pred)


# Comparison Between Actual Data & Predicted Data

# In[17]:


c_df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
c_df


# Problem Statement : What will be predicted score if a student studies for 9.25 hrs/ day?

# In[18]:


hours = [9.25]
pred = reg.predict([hours])
print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# Model Evaluation

# In[19]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test, Y_pred))


# -x-x-x-x-x-
