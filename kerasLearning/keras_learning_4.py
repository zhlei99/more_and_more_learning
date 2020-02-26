#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:27:47 2018

@author: zhaolei
"""
import numpy as np

#hold out validation------未能执行
def hold_out_validation():

    num_validation_samples = 10000
    
    np.random.shuffle(data)
    
    validation_data = data[:num_validation_samples]
    data = data[num_validation_samples:]
    
    training_data = data[:]
    
    model = get_model() #方法没有实现
    model.train(training_data)
    validation_score = model.evalute(validation_data)
    
    #tune model,retrain it ,evaluate it, tune it again..
    
    model = get_model()
    #train final model from entire data
    model.train(np.concatenate([training_data,validation_data]))
    test_score = model.evaluate(test_data)

#K-fold cross-validation
def k_fold_cross_validation():
    k = 4
    num_validation_samples = len(data) //k
    
    np.random.shuffle(data)
    
    validation_scores = []
    for fold in range(k):
        validation_data = data[num_validation_samples * fold :
            num_validation_samples * (fold + 1)]
        training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold +1): ]
            
        model = get_model()
        model.train(training_data)
        validation_score = model.evaluate(validation_data)
        validation_scores.append(validation_score)
    validation-score = np.average(validation_data)
    
    model = get_model()
    model.train(data)
    test_score = model.evaluate(test_data)
    

    
