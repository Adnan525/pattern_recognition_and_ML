#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Feature selection is also known as Variable selection or Attribute selection. Essentially, it is the process of selecting the most important/relevant features of a dataset. 
# 
# The importance of feature selection can best be recognized when you are dealing with a dataset that contains a vast number of features. This type of dataset is often referred to as a high dimensional dataset. Now, with this high dimensionality, comes a lot of problems such as - this high dimensionality will significantly increase the training time of your machine learning model, it can make your model very complicated which in turn may lead to Overfitting.
# 
# Often in a high dimensional feature set, there remain several features which are redundant meaning these features are nothing but extensions of the other essential features. These redundant features do not effectively contribute to the model training as well. So, clearly, there is a need to extract the most important and the most relevant features for a dataset in order to get the most effective predictive modeling performance.
# 
# Let me summarize the importance of feature selection:
# 
# - It enables the machine learning algorithm to train faster.
# - It reduces the complexity of a model and makes it easier to interpret.
# - It improves the accuracy of a model if the right subset is chosen.
# - It reduces Overfitting.

# # Feature Selection Methods
# In this section, you will study the different types of general feature selection methods 
# 
# - Filter methods
# - Wrapper methods

# ## Filter methods
# 
# The filter method uses the principal criteria of ranking technique and uses the rank ordering method for feature selection. The reason for using the ranking method is simplicity, produce excellent and relevant features. The ranking method will filter out irrelevant features before classification process starts.
# 
# Filter methods are generally used as a data preprocessing step. The selection of features is independent of any machine learning algorithm. Features give rank on the basis of statistical scores which tend to determine the features' correlation with the outcome variable. Some examples of some filter methods include the Chi-squared test and correlation coefficient scores.

# ## An example on Python code
# 
# For this example, we will use the Pima Indians Diabetes dataset. The description of the dataset can be found here. https://www.kaggle.com/uciml/pima-indians-diabetes-database
# 
# The dataset corresponds to classification tasks on which you need to predict if a person has diabetes based on 8 features.

# ### Import and Loading dataset

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# load dataset
diabetes_dataset = pd.read_csv("diabetes.csv", sep = ",")


# In[3]:


diabetes_dataset.head()


# In[4]:


diabetes_dataset.shape


# In[5]:


# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# ### Feature extraction

# In[6]:


feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = diabetes_dataset[feature_columns] # Features
y = diabetes_dataset.Outcome # Target


# In[7]:


X.head()


# In[8]:


y.unique()


# ### Chi-Squared statistical test
# 
# Chi-squared stats of non-negative features for classification tasks. See more details on https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
# 
# We now implement a Chi-Squared statistical test for non-negative features to select 4 of the best features from the dataset. You have already seen Chi-Squared test belongs the class of filter methods. If anyone's curious about knowing the internals of Chi-Squared, see the video here https://www.youtube.com/watch?v=VskmMgXmkMQ.
# 
# The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features, in this case, it is Chi-Squared.

# In[9]:


best_features  = SelectKBest(score_func=chi2, k=3)
best_features.fit(X, y)


# ### Summarize scores

# In[10]:


print(best_features.scores_)


# ### Obtaining features

# In[11]:


features = best_features.transform(X)
# Summarize selected features
print(features[0:5,:])


# ### Print features and their scores

# In[12]:


df_scores = pd.DataFrame(best_features.scores_)
df_columns = pd.DataFrame(X.columns)

# concatenate dataframes
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Features','Schi-square Score']  
print(feature_scores.nlargest(8,'Schi-square Score'))  


# ### Interpretation
# 
# You can see the scores for each attribute and the 3 attributes chosen (those with the highest scores): 
# 
# - Insulin 
# - Glucose
# - Age 
# 
# This scores will help you further in determining the best features for training your model.

# ## Wrapper methods
# 
# A wrapper method needs one machine learning algorithm and uses its performance as evaluation criteria. This method searches for a feature which is best-suited for the machine learning algorithm and aims to improve the performance. To evaluate the features, the predictive accuracy used for classification tasks.

# **Recursive Feature Elimination**
# 
# The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.
# 
# It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
# 
# You can learn more about the RFE class in here https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html.
# 
# ## An example on Python code
# 
# In this example, we will continue to work with the diabetes dataset and a classification model to select features based on the RFE.

# ### Import libraries

# In[13]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[14]:


classifiers = {
#reason: these models has feature importances
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter = 1000)
}


# ### Comparison between various classifiers

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.8)


# In[16]:


best_model_score = 0


# In[17]:


for name, classifier in classifiers.items():    
    print("classifier is ", name)
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if (best_model_score < metrics.accuracy_score(y_test, y_pred)):
        best_model = classifier
        best_model_score = metrics.accuracy_score(y_test, y_pred)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#    print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
#    print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
    print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))
    print('--------------------------------')


# In[18]:


best_model


# ### Adding the RFE method
# 
# You will use RFE with the various classifiers to select the top 3 features. The choice of algorithm does not matter too much as long as it is skillful and consistent.

# In[19]:


best_model_score = 0


# In[20]:


for n_features in range(1, 9, 1):    
    print("Number of features is ", n_features)
    
    rfe = RFE(best_model, n_features_to_select=n_features)
    model = rfe.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if (best_model_score < metrics.f1_score(y_test, y_pred, average = 'weighted')):
        n_best_features = n_features
        rfe_best = rfe
        best_model_score = metrics.f1_score(y_test, y_pred, average = 'weighted')
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#    print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
#    print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
    print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))
    print('--------------------------------')


# In[21]:


n_best_features


# In[22]:


rfe_best


# ### Summarize scores

# In[23]:


print("Num Features: %s" % (rfe_best.n_features_))
print("Selected Features: %s" % (rfe_best.support_))
print("Feature Ranking: %s" % (rfe_best.ranking_))


# ### Print the features

# In[24]:


df_scores = pd.DataFrame(rfe_best.ranking_)
df_columns = pd.DataFrame(X.columns)

# concatenate dataframes
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Features','Ranking']  
print(feature_scores.nsmallest(8,'Ranking'))  


# These are marked True in the support array and marked with a choice “1” in the ranking array. This, in turn, indicates the strength of these features.

# In[ ]:




