# -*- coding: utf-8 -*-
"""
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: zhlei99
"""

from numpy import *
import numpy as np
import operator
import importlib
import kNN
from os import listdir

#importlib.reload(sys)
from matplotlib import *
import matplotlib.pyplot as plt


def createDataSet():
   
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
    labels = ['A','A','B','B']
    return group, labels

 
def classify0(inX, dataSet, labels, k):
    """
    kNN  algorithm
    """
    #get dataSet number of row Vector
    dataSetSize = dataSet.shape[0]
    #construct an array by repeating inX the number of times given by (dataSetSize,1)
    #the times of line is dataSetSize, the times of columns is 1 
    #calculate distance
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #return the indices that would sort this array
    sorteDistIndicies = distances.argsort()
    #select the smallest k points
    classCount={}
    for i in range(k):
        voteIlabel = labels[sorteDistIndicies[i]]
        #get :D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None.
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #sorted
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True) #After f = itemgetter(2), the call f(r) returns r[2]
    #return predictive lable
    print 
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    date conversion
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #creat NumPy matrix
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    #label change int
    classLabelDict = {'largeDoses':3, 'smallDoses':2,'didntLike':1}
    
    #Text data parsing
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        
#       change label into int type
        classLabelVector.append(classLabelDict.get(listFromLine[-1]))
        index += 1
    fr.close()
    
    return returnMat, classLabelVector


def showDataMap(inX, inY, datingLabels):
    """
    show data map scatter 二维图
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    ax.scatter(inX, inY)
    ax.scatter(inX, inY, s=15.0 * np.array(datingLabels), c=15.0* np.array(datingLabels))
    plt.show()
    

def autoNorm(dataSet):
    """
    #normalized data 
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet /np.tile(ranges , (m , 1)) 
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    test calssfier error rate 
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    m =normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:], \
                                     datingLabels[numTestVecs:m],3)
        print ("the classfierResult came back with %d ,the real answer is : %d"\
            %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is : %f" %(errorCount/float(numTestVecs)))
        
def classifyPerson():
    """
    imput someone information and predicts how much she will like this person
    """  
    resultList = ['not at all','in small doses','in large doses'] 
    percentTats = float (input(\
                                   "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent fliter miles earned per year?"))    
    iceCream = float(input("liters of ice cream consumed per year?")) 
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    classifierResult = kNN.classify0([ffMiles, percentTats, iceCream ],normMat, \
                                     datingLabels,3)
    print ("you will probably like this person : %s" %(resultList[classifierResult - 1]))
"""
handwriting recognition with out kNN classifier,we'll be working only with the digits 0-9
"""        
def img2vector(filename):
    returnVect = np.zeros((1,1024))  
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    fr.close()
    
    return returnVect

def handwritingClassTest():
    hwLabels = []               #get contents of directory
    trainingFileList = listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]           #process class num from filename
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('./digits/trainingDigits/%s' %fileNameStr)
#        print (type(trainingMat[i,:]))
    testFileList = listdir ('./digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest= img2vector('./digits/testDigits/%s' %fileNameStr)
        classiferResult = classify0(vectorUnderTest, trainingMat , hwLabels, 3)         #k= 5, rate=0.017970 k=3 rate =0.010571
        print ("the classifier came back with:%d ,the real answer is : %d " \
               %(classiferResult, classNumStr))
        if (classiferResult != classNumStr):
            errorCount +=1.0
        
        print ("\n the total number of is : %d " % (errorCount))
        print ("\nthe total error rate is : %f" % (errorCount/float(mTest)))
        
        
        
        
        
        
        

if __name__ == '__main__':
    
#    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    
#    showDataMap(datingDataMat[:,1],datingDataMat[:,2], datingLabels)   
#    normDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)
#    kNN.datingClassTest()
#    kNN.classifyPerson()
#    testVector = kNN.img2vector('./digits/testDigits/0_13.txt')
    kNN.handwritingClassTest()
    
        
    
    
    