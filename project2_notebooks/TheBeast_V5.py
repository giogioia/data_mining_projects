#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:59:07 2021

@author: giovanni.scognamiglio
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import time
from multiprocessing import Manager, Pool, Process, cpu_count
import concurrent.futures 
import itertools
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

class BEAST:    
    def topolino(dataset, shared_dic, i, y):
            X = dataset[list(i)].values
            #print(i,':',(scores.mean()).round(5))
            param_list_0 = {'max_depth': [None, 5, 10, 15],
              'min_samples_split': [round((len(X)*0.01)), round(1.5*(len(X)*0.01)),round(2*(len(X)*0.01))],
              'min_samples_leaf': [round((len(X)*0.01)), round(1.5*(len(X)*0.01)),round(2*(len(X)*0.01))]}
            grid_search = GridSearchCV(clf, param_grid=param_list_0, n_jobs=1, pre_dispatch='3*n_jobs')
            grid_search.fit(X, y)
            results = grid_search.cv_results_
            candidates = np.flatnonzero(results['rank_test_score'] ==1)
            results['mean_test_score'][candidates[0]].round(3)
            shared_dic[i] = results['mean_test_score'][candidates[0]].round(3)
            
    def launcher1(dataframe, target_var, n):
        global dizz, bar
        shared_dic = {}
        #linear      
        if multi == False:
            dataset = dataframe.drop(columns=target_var, errors= 'ignore')
            for i in tqdm(list(itertools.combinations(list(dataset),n))):
                BEAST.topolino(dataset, shared_dic, i, y)
            dizz = shared_dic
        #future multipro
        else:
            dataset = dataframe.drop(columns=target_var, errors= 'ignore')
            shared_dic = Manager().dict()
            with concurrent.futures.ProcessPoolExecutor(2) as executor:
                for i in tqdm(list(itertools.combinations(list(dataset),n))):
                    executor.submit(BEAST.topolino, dataset, shared_dic, i, y)
            dizz = dict(shared_dic)
            
    def paperino(pippo,param_list_2, future_dic):
        X = df.loc[:,list(pippo)].values
        grid_search = GridSearchCV(clf, param_grid=param_list_2, n_jobs=1, pre_dispatch='2*n_jobs')
        grid_search.fit(X, y)
        results = grid_search.cv_results_
        candidates = np.flatnonzero(results['rank_test_score'] ==1)
        results['mean_test_score'][candidates[0]].round(4)
        future_dic[pippo] = [results['mean_test_score'][candidates[0]].round(4), str(grid_search.best_estimator_)]

    def cherry_cake(q):
        global pluto
        future_dic = Manager().dict()
        param_list_2 = {'max_depth': [None] + list(np.arange(2, 10)),
              'min_samples_split': [2]+list(np.arange(3, 20, 2)),
              'min_samples_leaf': [1]+list(np.arange(3, 20, 2))}
        with concurrent.futures.ProcessPoolExecutor(2) as executor:
            for pippo in tqdm(list(q.keys())):
                executor.submit(BEAST.paperino,pippo,param_list_2, future_dic)
        best_dic = dict(future_dic)
        pluto = dict(sorted(best_dic.items(), key=lambda item: item[1],reverse=True)[:1])
        return pluto
        
    def initiate(classifier, data, t_var, column_range = 'all', multi_process = True):
        #convert param to gloval vars
        global df, clf, min_col, max_col, grid, multi, y
        clf = classifier
        df = data
        y = df[t_var]
        col_range = str(column_range)
        if col_range == 'all':
            min_col = 2
            max_col = len(list(df))-1
        else:
            min_col = col_range[0]
            max_col =  col_range[1]
        multi = multi_process
        #main
        final_dict = {}
        for i in range(min_col, max_col+1):
            BEAST.launcher1(dataframe = df, target_var = t_var, n = i)
            final_dict.update(dict(sorted(dizz.items(), key=lambda item: item[1],reverse=True)[:5]))
        q = dict(sorted(final_dict.items(), key=lambda item: item[1],reverse=True)[:2])
        #end
        paperina =  BEAST.cherry_cake(q)
        return paperina