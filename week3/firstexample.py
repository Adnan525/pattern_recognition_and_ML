# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:43:41 2020

@author: s434074
"""

total_mark = 0
marks = [30, 25, 40]
for point in marks:
	total_mark += point
print(total_mark)

total_mark = 0
marks = [30, 25, 40]
for i in range(len(marks)):
	total_mark += marks[i]
print(total_mark)


n = input("Enter a positive integer: ")
sum_squares = 0
for i in range(1, int(n)):
    sum_squares += i**2
print(sum_squares)

a = int(input("Enter a positive integer: "))
b = int(input("Enter a positive integer: "))
c = int(input("Enter a positive integer: "))
max = 0
if (max < a):
    max = a
if (max < b):
    max = b
if (max < c):
    max = c
print(max)    
