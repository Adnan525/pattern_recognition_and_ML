# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:46:24 2020

@author: s434074
"""

import pandas as pd

#read iris data
iris_dataset = pd.read_csv("iris.data", sep=',', names=["sepal_length", "sepal_width", "petal_length", 
                                                "petal_width", "species"])




