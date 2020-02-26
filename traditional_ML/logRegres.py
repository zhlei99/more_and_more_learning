#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:30:49 2018

@author: zhaolei
"""
import logRegres
import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    with open('./data/testSet.Txt') as fr:

        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid (inX):
    return 1.0/(1 + np.exp(-inX))

def gradAscent (dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)              #convert to Numpy matrix data type
    labelMat = np.mat(classLabels).transPose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights
    
    
    

if __name__ == '__main__':
    dataMat, labelMat = logRegres.loadDataSet()
    
    