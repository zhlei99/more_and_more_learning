#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:04:36 2018

@author: zhaolei
"""
import numpy as np
import bayes
import chardet
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])              #greate an empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)      #greate the union of two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList , inputSet):
    """
    input one line 
    """
    returnVec = [0]*len(vocabList)      #greate a vector of all 0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print ("the word : %s is not in my Vocabulary") % (word)
    return returnVec

def trainNB0 (trainMatrix,trainCategory):
    """
    trainMatrix: ndarray
    two-class problem,
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)   
    p0Num = np.ones (numWords)  #to lessen the impact of that when we multiply the together we get 0
    p1Num = np.ones(numWords)   #we 'll initialize all of out occurrence counts to 1
    p1Denom = 2.0       #to lessen the impact of that when we multiply the together we get 0
    p0Denom = 2.0       #we'll initialize the denominators to 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] 
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)           #element-wise multiplication
    p0Vect = np.log(p0Num/p0Denom )         #change to log()
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec , p1Vec , pClass1):
    """
    vec2Classify: a vector to classify
    """
    p1 = sum(vec2Classify *p1Vec) + np.log(pClass1)     #element-wise multiplication
    p0 = sum(vec2Classify *p0Vec) + np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts, listClasses = bayes.loadDataSet()
    myVocabList = bayes.createVocabList(listOPosts)
    print (myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        
        trainMat.append( bayes.setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V ,pAb = bayes.trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    print 
    thisDoc = np.array(bayes.setOfWords2Vec(myVocabList, testEntry))

    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) )
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList, inputSet):
    """
    naive Bayes bag-of-words model
    """
    returnVec = [0]*len(vocabList)      #greate a vector of all 0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print ("the word : %s is not in my Vocabulary") % (word)
    return returnVec

def textParse(bigString):
    """
    file parsing
    """
    import re
#    regEx = re.compile('\\W*') 
#    listOfTokens = regEx.split(bigString)
    if isinstance(bigString,str) :
        listOfTokens = re.split(r'\W*', bigString) 
    else:
        print (type(bigString))
#    print ("listOfTokens is :%s" %listOfTokens)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2 ]

def spamTest():
    docList = []
    classList = []
    fullText =[]
    for i in range (1,26):              #load and parse text files
        # open by byte
        textContent = open('./data/email/spam/%d.txt' % i, 'rb').read()
        #如果读出的文件有特殊编码，则检测编码格式，用特定的编码解码二进制内容，转变成字符
        textContent = textContent.decode(chardet.detect(textContent)['encoding'])
        
        if isinstance(textContent,str) :
            wordList = textParse(textContent)
        else:
            print ("errro : ",'./data/email/spam/%d.txt' % i)
            print (type(textContent))
            print (textContent)
#        wordList = textParse(open('./data/email/spam/%d.txt' % i, 'rb').read())
        #如果读出的文件有特殊编码，则检测编码格式，用特定的编码解码二进制内容，转变成字符
       
            
        #print ('./data/email/spam/%d.txt' % i )
#        print (type(wordList))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        textContent = open('./data/email/ham/%d.txt' % i, 'rb').read()
        textContent = textContent.decode(chardet.detect(textContent)['encoding'])
        if isinstance(textContent,str) :
            wordList = textContent
            wordList = textParse(textContent)
        else:
            print ("errro : ",'./data/email/ham/%d.txt' % i)
            print (type(textContent))
            print (textContent)
        
#        wordList = textParse(open('./data/email/ham/%d.txt' % i ,'rb').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):        
        randIndex = int(np.random.uniform(0,len(trainingSet)) )  #randomly create the training set
        testSet.append( trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex] ))
        trainClasses.append(classList[docIndex])
    p0V, p1V ,pAb = bayes.trainNB0(trainMat, trainClasses)
#    print (p0V, p1V ,pAb)
#    print (vocabList)
    errorCount = 0
    for docIndex in testSet:
        textVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(textVector, p0V , p1V , pAb) != classList[docIndex]:
            errorCount += 1
    print ("the error rate is :" ,(float(errorCount)/len(testSet)),'\n',"the number of error is :",errorCount)
    


if __name__ =='__main__':
    spamTest()
#    mySent = 'this book is the best book on pyhton ro M.L. I .'
#    listOfTokens = textParse(mySent)
    
        