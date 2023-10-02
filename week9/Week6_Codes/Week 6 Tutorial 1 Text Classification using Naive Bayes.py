#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# 
# Text Classification is one of the widely use from a task of natural language processing (NLP). It is also one of the important and typical task in pattern recognition. Text classification aims to automatically classify text documents into one or more defined categories which  can be a web page, library book, media articles, gallery etc. Some examples of text classification can be:
# 
# - Understanding audience sentiment from social media,
# - Detection of spam and non-spam emails,
# - Auto tagging of customer queries, and
# - Categorization of news articles into defined topics.
# 
# In this tutorial, we would like to demonstrate how we can categorize different new articles into defined topics using sklearn library. 
# 
# # Dataset
# 
# The dataset used is the 20 Newsgroup dataset (http://qwone.com/~jason/20Newsgroups/). The 20 Newsgroups data set is a collection of about 20000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. The dataset was originally collected by Ken Lang. The 20 newsgroups dataset has become a popular data for experiments in text applications of machine learning techniques including text classification and text clustering. 
# 
# The dataset used in this tutorial is extracted from the 20 Newsgroups dataset sorted by date in which duplicates and some headers removed. The dataset contains 18846 documents. The data is categorized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. comp.sys.ibm.pc.hardware / comp.sys.mac.hardware), while others are highly unrelated (e.g misc.forsale / soc.religion.christian). Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter:
# 
# - comp.graphics
# - comp.os.ms-windows.misc
# - comp.sys.ibm.pc.hardware
# - comp.sys.mac.hardware
# - comp.windows.x	
# - rec.autos
# - rec.motorcycles
# - rec.sport.baseball
# - rec.sport.hockey	
# - sci.crypt
# - sci.electronics
# - sci.med
# - sci.space
# - misc.forsale	
# - talk.politics.misc
# - talk.politics.guns
# - talk.politics.mideast	
# - talk.religion.misc
# - alt.atheism
# - soc.religion.christian
# 
# 
# 

# # Classification of news articles into different topics using Naive Bayes models

# ## Retrieving data

# In[1]:


from sklearn.datasets import fetch_20newsgroups
#categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']


# In[2]:


#newsgroups = fetch_20newsgroups(subset='all', categories=categories)
newsgroups = fetch_20newsgroups(subset='all')


# ## Exploring the dataset

# In[3]:


newsgroups.data[1]


# In[4]:


list(newsgroups.target_names)


# In[5]:


newsgroups.filenames.shape


# In[6]:


newsgroups.target.shape


# ## Feature extraction
# 
# This step will calculate a bag of words and their frequencies. We use the CountVectorizer from sklearn to do this.

# In[7]:


# import the CountVectorizer 
from sklearn.feature_extraction.text import CountVectorizer


# ### Let's see an example first

# In[8]:


document = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]


# In[9]:


cv = CountVectorizer()


# In[10]:


cv = CountVectorizer()
cv.fit_transform(document)

cv.vocabulary_


# In[11]:


cv.fit_transform(document).toarray()


# **Output a bag of words and their frequencey**

# In[12]:


word_list = cv.get_feature_names()
count_list = cv.fit_transform(document).toarray().sum(axis=0)

print(dict(zip(word_list,count_list)))


# ### Feature Extraction for the 20NewsGroup dataset

# In[13]:


features = cv.fit_transform(newsgroups.data)


# In[14]:


features.shape


# In[15]:


features.toarray()


# ## Split these feature vectors into a trainning and testing sets

# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    newsgroups.target, 
                                                    train_size = 0.8,
                                                    random_state=0)


# In[17]:


X_train.shape


# In[18]:


X_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# ## Build a Multinomial Naive Bayes model 

# In[21]:


from sklearn.naive_bayes import MultinomialNB


# In[22]:


model = MultinomialNB(alpha=1)


# ## Training the model and make a prediction

# In[23]:


model.fit(X_train,y_train)


# In[24]:


y_pred = model.predict(X_test)


# In[25]:


print(y_test)
print(y_pred)


# ## Evaluate the model

# In[26]:


from sklearn import metrics
from sklearn.metrics import classification_report


# In[27]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))


# **Classification report**

# In[28]:


report = classification_report(y_test, y_pred, target_names = newsgroups.target_names)


# In[29]:


print(report)


# **Confusion matrix**

# In[30]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# ## Finding the best parameter alpha for the model using GridSearchCV 
# 
# GridSearchCV is an exhaustive search with cross validation over specified parameter values to determine the optimal value for a given model. The parameters of the model used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# In[31]:


#Import the Grid Search

from sklearn.model_selection import GridSearchCV


# In[32]:


model = MultinomialNB()


# **Parameters setting for alpha = 10, 1, 0.1, 0.001, 0.0001**

# In[33]:


params = {'alpha': [10, 1, 1e-1, 1e-2, 1e-3]}


# **Running the Grid Search on parameters and fit the trainning data**

# In[34]:


# 10-fold
#grs = GridSearchCV(model, param_grid=params, cv = 10)

# 5-fold default
grs = GridSearchCV(model, param_grid=params)


# In[35]:


grs.fit(X_train, y_train)


# **Output the optimal values**

# In[36]:


print("Best Hyper Parameters:",grs.best_params_)


# **Make a prediction and calculate metrics**

# In[37]:


y_pred=grs.predict(X_test)


# In[38]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))


# In[ ]:




