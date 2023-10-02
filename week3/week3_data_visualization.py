# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:56:40 2020

@author: s434074
"""

import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt

#read iris data
iris_dataset = pd.read_csv("iris.data", sep=',', names=["sepal_length", "sepal_width", "petal_length", 
                                                "petal_width", "species"])

iplot = iris_dataset['species']\
    .value_counts()\
    .plot(kind='pie',autopct='%.2f',figsize=(8, 8))

iplot.set_ylabel('')

iris_dataset.boxplot(by="species", figsize=(12, 6))


sns.set(style="darkgrid")
sc=iris_dataset[iris_dataset.species=='Iris-setosa'].plot(kind='scatter',x='sepal_length',y='sepal_width',
                                                          color='red',label='Setosa')
iris_dataset[iris_dataset.species=='Iris-versicolor'].plot(kind='scatter',x='sepal_length',y='sepal_width',
                                                           color='green',label='Versicolor',ax=sc)
iris_dataset[iris_dataset.species=='Iris-virginica'].plot(kind='scatter',x='sepal_length',y='sepal_width',
                                                          color='orange', label='virginica', ax=sc)
sc.set_xlabel('Sepal Length in cm')
sc.set_ylabel('Sepal Width in cm')
sc.set_title('Sepal Length Vs Sepal Width')