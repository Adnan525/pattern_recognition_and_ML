#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Proliferation of cryptocurrencies (e.g., Bitcoin) that allow pseudo-anonymous transactions, has made it easier for ransomware developers to demand ransom by encrypting sensitive user data. The recently revealed strikes of ransomware attacks have already resulted in significant economic losses and societal harm across different sectors, ranging from local governments to health care.
# Most modern ransomware use Bitcoin for payments. However, although Bitcoin transactions are permanently recorded and publicly available, current approaches for detecting ransomware depend only on a couple of heuristics and/or tedious information gathering steps (e.g., running ransomware to collect ransomware related Bitcoin addresses).
# 
# In this tutorial, we would like to do Ransomware Detection on the Bitcoin Blockchain using the K-NN model. 

# # Dataset 
# 
# 
# The dataset used is BitcoinHeist Ransomware Dataset
# 
# https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset 
# 
# Features
# 
# - address: String. Bitcoin address.
# - year: Integer. Year.
# - day: Integer. Day of the year. 1 is the first day, 365 is the last day.
# - length: Integer.
# - weight: Float.
# - count: Integer.
# - looped: Integer.
# - neighbors: Integer.
# - income: Integer. Satoshi amount (1 bitcoin = 100 million satoshis).
# - label: Category String. Name of the ransomware family (e.g., Cryptxxx, cryptolocker etc) or white (i.e., not known to be ransomware).
# 
# Our graph features are designed to quantify specific transaction patterns. Loop is intended to count how many transaction i) split their coins; ii) move these coins in the network by using different paths and finally, and iii) merge them in a single address. Coins at this final address can then be sold and converted to fiat currency. Weight quantifies the merge behavior (i.e., the transaction has more input addresses than output addresses), where coins in multiple addresses are each passed through a succession of merging transactions and accumulated in a final address. Similar to weight, the count feature is designed to quantify the merging pattern. However, the count feature represents information on the number of transactions, whereas the weight feature represents information on the amount (what percent of these transactions’ output?) of transactions. Length is designed to quantify mixing rounds on Bitcoin, where transactions receive and distribute similar amounts of coins in multiple rounds with newly created addresses to hide the coin origin.
# 
# White Bitcoin addresses are capped at 1K per day (Bitcoin has 800K addresses daily).
# 
# Note that although we are certain about ransomware labels, we do not know if all white addresses are in fact not related to ransomware.
# 
# When compared to non-ransomware addresses, ransomware addresses exhibit more profound right skewness in distributions of feature values.

# # Read the data

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


bitcoin_heist = pd.read_csv("BitcoinHeistData.csv")


# # Data Exploration

# In[3]:


bitcoin_heist.head()


# In[4]:


bitcoin_heist.describe()


# In[5]:


bitcoin_heist.describe(include="O")


# In[6]:


bitcoin_heist.dtypes


# In[7]:


bitcoin_heist


# # K-NN model for ransomware detection

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics


# ## Convert categorical values to numeric values
# 
# Firstly, we convert a categorical column to numeric column: if a label is 'white', it is known to be ransomeware and is labelled 1. Otherwise, it is labelled 0

# In[9]:


bitcoin_heist["labels"] = [0 if x == 'white' else 1 for x in bitcoin_heist['label']]


# In[10]:


bitcoin_heist["labels"].value_counts()


# ## Extract features
# 
# We use only the first 200000 instances. This step aims to reduce time complexity for identifing the optimal K. If we extract all instance, this will spend a lot of time.  

# In[11]:


X = bitcoin_heist.loc[0:200000, ['year',"day", "length", "weight","count", "looped", "neighbors", "income"]]
y = bitcoin_heist.loc[0:200000,'labels']


# ## Split features and labels into a trainning and testing sets

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size = 0.8,
                                                    random_state=0)


# ## Build a K-NN model

# In[13]:


model = KNeighborsClassifier(3)


# ## Training the model and make a prediction

# In[14]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ## Evaluate the model

# In[15]:


cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)


# In[16]:


report_knn = classification_report(y_test, y_pred)
print(report_knn)
f1_knn = f1_score(y_test, y_pred,average='weighted')
print(f1_knn)


# ## **Parameter Tunning using GridSearchCV**

# In[17]:


from sklearn.model_selection import GridSearchCV


# In[18]:


model = KNeighborsClassifier()


# In[19]:


params = {'n_neighbors': range(1,10)}


# In[20]:


# 10-fold
#grs = GridSearchCV(model, param_grid=params, cv = 10)

# 5-fold default
grs = GridSearchCV(model, param_grid=params)
grs.fit(X_train, y_train)


# In[21]:


print("Best Hyper Parameters:",grs.best_params_)


# In[22]:


y_pred=grs.predict(X_test)


# In[23]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))


# **Now, we will try to get more data to train the model**

# In[24]:


X = bitcoin_heist[['year',"day", "length", "weight","count", "looped", "neighbors", "income"]]
y = bitcoin_heist['labels']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size = 0.8,
                                                    random_state=0)


# In[26]:


model = KNeighborsClassifier(3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[27]:


cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)


# In[28]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))

