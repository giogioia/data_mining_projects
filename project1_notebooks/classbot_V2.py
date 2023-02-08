#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:59:07 2021

@author: giovanni.scognamiglio
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import time
from multiprocessing import Manager, Pool, Process, cpu_count
import concurrent.futures 
import itertools
import json

def preprocessing():
    global df
    #importd df
    df = pd.read_csv('Datasource_2.csv')
    #enable full view of pandas dataframes
    pd.set_option('display.max_columns', None)
    
    df.drop(columns = ['YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TotalWorkingYears'], inplace=True)
    df.drop(columns = ['Gender','JobInvolvement','PerformanceRating'], inplace=True)
    df.drop(columns = ['EducationField','JobRole','MaritalStatus'], inplace=True)
    #encoding
    d ={'Non-Travel':1, 'Travel_Rarely':2, 'Travel_Frequently':3}
    df.loc[:,'BusinessTravel'] = df.loc[:,'BusinessTravel'].map(d)

def small_beast(dataset, shared_dic, i, y):
        X = dataset[list(i)].values
        scores = cross_val_score(clf, X, y, cv=3)
        shared_dic[i] = (scores.mean()).round(5)
        print(i,':',(scores.mean()).round(5))
        
def beast_v2(dataframe, target_var, n):
    global dizz
    shared_dic = {}
    #linear      
    if mode == 'linear':
        y = df[target_var]
        dataset = dataframe.drop(columns=target_var, errors= 'ignore')
        for i in list(itertools.combinations(list(dataset),n)):
            small_beast(dataset, shared_dic, i, y)
        dizz = shared_dic
    #future multipro
    else:
        y = df[target_var]
        dataset = dataframe.drop(columns=target_var, errors= 'ignore')
        shared_dic = Manager().dict()
        with concurrent.futures.ProcessPoolExecutor(2) as executor:
            for i in list(itertools.combinations(list(dataset),n)):
                executor.submit(small_beast, dataset, shared_dic, i, y)
        dizz = dict(shared_dic)

def time_est():
    global conf
    x = 0
    for i in (min_col, max_col+1): 
        x += len(list(itertools.combinations(list(df),i)))
    print(f'Algorithm will try {x} combinations',
          f'Estimated time:\t{int((0.01 * x)/60)} minutes')
    conf = input('Continue? (y/n)\n').lower()

if __name__ == '__main__':
    preprocessing()
    while True:
        mode = input('Select desired process (linear/else):\n').lower()
        min_col = int(input('Minimum number of columns:\t'))
        max_col = int(input('Maximum number of columns:\t'))
        depth = input('Max_depth:\t')
        try: depth = int(depth)
        except ValueError: depth =  None
        time_est()
        if conf in ['yes','y','si']:
            break
    #instanciate
    start =  time.perf_counter()
    final_dict = {}
    clf = DecisionTreeClassifier(criterion='gini', max_depth= depth, min_samples_split=2, min_samples_leaf=1)
    for i in (min_col, max_col+1):
        beast_v2(dataframe = df, target_var = 'Attrition', n = i)
        final_dict.update(dict(sorted(dizz.items(), key=lambda item: item[1],reverse=True)[:5]))
    finish =  time.perf_counter()
    q = dict(sorted(final_dict.items(), key=lambda item: item[1],reverse=True)[:10])
    print('Top 5 results:')
    for i,o in zip(q,list(q.values())): print(i,':',o)
    print(round((finish-start),2),'seconds')
    with open('classbot_results.py','w') as file:
        for i,o in zip(q,list(q.values())):
            file.write(f'{i} : {o}')

    
    


