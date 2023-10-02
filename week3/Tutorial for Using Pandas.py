#!/usr/bin/env python
# coding: utf-8

# This is a short introduction to pandas and numpy libraries, geared mainly for new users
# 
# The objective includes:
# - Creat a DataFrame
# - Viewing a DataFrame
# - Selection sub-data from a DataFrame 
# 

# In[1]:


#Import the library to use
import pandas as pd
import numpy as np


# A DataFrame: two-dimensional tabular data structure with labeled axes (rows and columns)

# In[2]:


df = pd.DataFrame({'A': 1.,
                        'B': pd.Timestamp('20201008'),
                        'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                        'D': np.array([3] * 4, dtype='int32'),
                        'E': ["test", "train", "test", "train"],
                        'F': 'foo'})
df


# A DataFrame has columns of different types

# In[3]:


df.dtypes


# Convert a list to a DataFrame 

# In[4]:


height_weight_list = [['David', 175, 71], ['Peter', 170, 58], ['Mark', 186, 92]]

df_index = pd.DataFrame(height_weight_list, columns=['Name', 'Height', 'Weight'])
print(df_index)


# Viewing data from the Wine Quality dataset

# In[5]:


#Read the red wine quality dataset
wine_red_dataset = pd.read_csv("winequality-red.csv", sep=';')


# In[6]:


wine_red_dataset


# Read the top rows of the dataframe

# In[7]:


wine_red_dataset.head(10)


# Get the headers of a Dataframe

# In[8]:


list(wine_red_dataset.columns) 


# Sorting by values

# In[9]:


wine_red_dataset.sort_values(by = "fixed acidity")


# Getting values of columns

# In[10]:


wine_red_dataset[['fixed acidity', 'volatile acidity', 'alcohol']]


# In[11]:


#drop a column
wine_red_dataset.drop(['pH', 'quality'], axis = 1)


# Concatenate two DataFrames

# In[12]:


#Read a white wine data
wine_white_dataset = pd.read_csv("winequality-white.csv", sep=';')


# In[13]:


#show the shape of a DataFrame
wine_white_dataset.shape


# In[14]:


wine_red_dataset.shape


# In[15]:


#Concatenate to build a wine dataset (no indexes repeated)
wine_dataset = pd.concat([wine_red_dataset, wine_white_dataset], ignore_index=True)


# In[16]:


#statistic summary of wine data:
wine_dataset.dtypes


# In[17]:


wine_dataset.shape


# Selection by row index

# In[18]:


#get first 5 rows
wine_dataset[0:5]


# Selection by labels

# In[19]:


#get rows 5 to 15 of two columns alcohol and quality
wine_dataset.loc[5:15, ['alcohol', 'quality']]


# Selection by position

# In[20]:


#Get rows 1 to 4 of the wine dataset
wine_dataset.iloc[1:5]


# In[21]:


#Getting rows 1 to 4 from columns 2 to 4
wine_dataset.iloc[1:5, 2:5]


# Check for missing values

# In[22]:


wine_dataset.isnull().sum()


#  Getting all values of a column

# In[23]:


wine_dataset['quality'].value_counts()


# Adding headers if neccessary 

# In[24]:


iris_dataset = pd.read_csv("iris.data", sep=',', header = None)


# In[25]:


iris_dataset


# In[26]:


#Adding a header 
new_iris_dataset = pd.DataFrame(iris_dataset.values, columns = ["sepal_length", "sepal_width", "petal_length", 
                                                "petal_width", "species"])


# In[27]:


new_iris_dataset


# In[30]:


new_iris_dataset.to_csv('new_iris.csv', sep=',', encoding='utf-8')


# Adding a header when reading a data

# In[28]:


iris_dataset_with_header = pd.read_csv("iris.data", sep=',', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])


# In[29]:


iris_dataset_with_header


# In[ ]:




