#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:44:55 2018
decision trees
ID3
@author: zhlei99
"""
from numpy import *
import numpy as np
import operator
import importlib
from os import listdir
import trees
import math

def calcShannonEnt(dataSet):
    """
    function to calculate the Shannon entropy of a dataSet
    """
    numEntries = len(dataSet)
    labelCounts = {}
    #create dictionary of all possible classes
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0        
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2)
    #    print (shannonEnt)
    #print (labelCounts)    
    return shannonEnt

def createDataSet():
    """
    数据集特点：1、必须list,2、最后一列是分类标签
    labels是包含了所有特征名称的数组
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    
    labels = ['no surfacing','flippers']
    return dataSet , labels

def splitDataSet (dataSet, axis, value):
    """
    
    dataSet: the dataset we'll split
    axis: the feature we will split on (index)
    value: the value of the feature to return (not only)
    dataSet splitting on a given feature
    
    """
    retDataSet = []             #create separate list 
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #remove fetaure and select row with feature=value
            reducedFeatVec.extend(featVec[axis+1:]) 
            retDataSet.append(reducedFeatVec)
    return retDataSet           #new ndarray, 
            
def chooseBestFeatureToSplit(dataSet):
    """
    choosing the best feature to split on
    
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #creat unique list of feature class labels
        featList = [example[i] for example in dataSet] #the all values of feature
        uniqueVals = set (featList) 
        newEntropy = 0.0
        for value in uniqueVals:
            #calculate entropy for each split
            subDataSet = splitDataSet(dataSet, i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob* calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #find the best information gain
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    create vote when features are consumed and the class labels are not all the same
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),\
                              key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
        
def createTree (dataSet, labels):
    """
    Tree-building code
    dataSet: dataSet
    labels:the algorithm could function without this ,but it would be difficult
    to make any sense of the data
    labels是对数据属性的名字的数组，对dataSet的解释，dataSet中都是数值，难以理解
    
    """ 
    classList = [example[-1] for example in dataSet]
    #stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #stop when no more features, return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]    #for example labels[0] = 'no surfacing'
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set (featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
              (dataSet, bestFeat, value), subLabels)
    
    return myTree
    
def classify(inputTree, featLabels, testVec):
    """
    classification function for an existing decision tree
    """
   
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)       #Translate label string to index

    for key in secondDict.keys():          
        if testVec[featIndex] == key :      #get feature value of testVec
            if type(secondDict[key]).__name__ =='dict':

                classLabel = classify(secondDict[key],featLabels , testVec)
     
            else:
                classLabel = secondDict[key]
    return classLabel
    
def storeTree (inputTree,filename):
    """
    methods for persisting the decision tree with pickle
    """
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)




def classContactLens():
    """
    using decision trees to predict contact lens type
    """
    lenses = []
    with open(r'./data/lenses.txt','rb') as fr:
        for inst in fr.readlines():
            inst = inst.strip()
            lenses.append(str(inst,encoding = 'utf-8').split('\t')) #byte to str
#        lenses = [inst.strip().split('\t')  for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = trees.createTree(lenses, lensesLabels)
        print ("lensesTree is :%s"%(lensesTree))
        
        
    
    
 
if __name__ == '__main__':
#    myDat,labels = createDataSet()
#    myTree = trees.createTree(myDat,labels)
 #   myDat,labels = createDataSet()
#    classLabel = trees.classify(myTree,labels,[1,1])
#    print (classLabel)
    classContactLens()
    
        