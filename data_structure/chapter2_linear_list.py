#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:56:44 2019

@author: zhaolei
第二章 线性表 

"""
'''
算法2.1
'''

def union(list_a, list_b ):
    '''
    list_a = []
    list_b = []
    执行操作A = A U B
    '''
    for item in list_b:
        tag = False
        for item_a in list_a:
            if item_a == item:
                tag = True
                break
        if not tag :
            list_a.append(item)
    return list_a

def test_union():
    '''
    test case
    '''
    a = [1,3,4]
    b = [1,4,7,8,9]
    list_a = union(a, b)
    print (list_a)
    
'''
算法2.2
已知线性表LA和LB中的元素按照非递减有序排列，现要求将LA和LB归并为一个新的线性表LC，
LC中的数据元素仍按值非递减有序排列。
'''
def test_MergeList():

    LA = (3,5,8,11)
    LB = (2,6,8,9,11,15,20)    
    #LC = (2,3,5,6,8,8,9,11,11,15,20)
    Lc = MergeList(LA, LB)
    
    print (Lc)
#注意剩余表的处理
def MergeList(La, Lb):
    La_len = len(La)
    Lb_len = len(Lb)
    Lc = []
    i = j = k = 0
    while (i < La_len) and (j < Lb_len):
        if (La[i] <= Lb[j]):
            Lc.append(La[i])
            i += 1
        else:
            Lc.append(Lb[j])
            j += 1
        k += 1
    #插入余下的La中的数据
    while(i < La_len):
        Lc.append(La[i])
        i += 1
    #插入余下的Lb中的数据
    while(j < Lb_len):
        Lc.append(Lb[j])
        j += 1
        
    return Lc
    
'''
线性表的顺序表示和实现
'''   
class LinearListSquence(object):
    def __init__(self, listsize = 20):
        #默认容纳20个元素
        self.listsize = listsize
        #数据结构初始化
        self.length = 0
        self.date = [None] * self.listsize         #list类型
    
    #在第i个元素之前插入一个元素
    def ListInsert_Sq(self, i, e):
        #i的位置需要合法
        if (i < 1 ) or (i > self.length ) :
            print ("Error : i is illegal ")
        if (self.length >= self.listsize):
            print ("Error : overflow")
        for j in range(self.length-1, i-2, -1):
            print (j)
            self.date[j+1] = self.date[j]
        self.date[i-1] = e
        self.length += 1
    
    #删除第i个元素    
    def ListDelete_Sq(self, i, e):
        if (i <i ) or(i > self.length):
            print ("Error : i is illegal ")        
        e = self.date[i]
        for j in range(i, self.length - 1):
            self.date[j-1] = self.date[j]
        self.length -= 1
        return e
        
    #创建一个默认对象
    def CreateList(self, L):
        for i in range(len(L)):
            self.date[i] = L[i]
            self.length += 1

'''


'''            
    
            
        
    
        
        

    
        
    
    
        
            



if __name__=='__main__':
    lk = LinearListSquence()
    lk.CreateList([1,2,3,4,5,6])
#    e = None
#    e = lk.ListDelete_Sq(2,e)
    lk.ListInsert_Sq(3,10)
    lk.date
    
    

    