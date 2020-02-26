#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:23:19 2019

@author: zhaolei
"""
'''
排序
input : n个数的序列<a1, a2, ..., an>
output: 有序的序列
'''
class SortMethod():
    
    def __init__(self, alist):
        self.list = alist   
    '''
    插入排序
    时间复杂度:O(n^2)
    空间复杂度:O(1)
    '''
    def insertion_sort(self, a : list):
        for i in range(1,len(a)):   #i元素循环变量
            key = a[i]
            j = i - 1 #j是局部变量从后向前插入,得到当前元素
            while j >=0 and key < a[j-1]:
                a[j+1] = a[j]
                j -= 1
            a[j+1] = key
        return a
                
    '''
    归并排序
    时间复杂度：O(nlgn)
    空间复杂度：O(n)
    递归三步骤：1、分解； 2、解决； 3、合并：（对于归并排序来说最重要的步骤）
    A[p,...,r]
    '''
    def merge_sort(self, a , p, r):    
        if p < r:
            mid = int((p + r)/2) #找到中位数
            self.merge_sort( a , p, mid) #左部分
            self.merge_sort( a,mid+1, r) #右部分
            self.merge( a, p, mid, r) #合并操作
        return a
    #两堆数组，进行合并排序，分解成最小粒度
    def merge(self, a, p, mid, r):
        #没放入哨兵牌,分解后变成两个数组L = [p,...mid],R = [mid+1,...r]
        L = a[p:mid+1] #注意python的数组切片。a = [1,3,4],a[0:2]=[1, 3]
        R = a[mid+1:r+1]
        #两个数组合并成一个有序数组,注意一方为空后，直接加入
        print(L,R)
        i = j = 0
        k = p
        while i< len(L) and j < len(R):
            if L[i] <= R[j]:
                a[k] = L[i]
                k = k + 1
                i = i + 1
            else:
                a[k] = R[j]
                k = k + 1
                j = j + 1
        #把剩余链直接赋值给a
        if i == len(L) and j < len(R):
            a[k:r+1] = R[j:]
        if j == len(R) and i < len(L):
            a[k:r+1] = L[i:]
    
    '''
    快速排序 
    分治法：1、分解、2解决、3、合并
    最坏的时间复杂度O(n^2)
    期望时间复杂度O(nlgn)
    '''
    def quicksort(self, a, p, r):
        if p < r:
            mid = self.partition(a, p, r) #这步操作已经排好一个元素的位置了
            self.quicksort(a, p, mid-1) #低位部分寻找
            self.quicksort(a, mid+1, r) #高位部分寻找
        
    
    #思想：找到一个元素，让左边元素都小于等于此元素，右边的元素都大于给定元素       
    def partition(self, a, p, r):
        temp = a[r] #临时存储需要选择的元素
        #定义两个变量，low从左到右扫描，找到比temp大的第一个值，high从尾向前扫描，找到比temp小的元素
        low = p 
        high = r
        while low < high:
            while low < high and a[low] <= temp : #在低位开始寻找
                low = low + 1 #元素小于等于temp，向后寻找
            a[high] = a[low] #把找到比temp 大的值，给high位置
            while low < high and a[high] >= temp: #在高位开始寻找
                high = high - 1  
            a[low] = a[high]                
        a[low] = temp
        return low
    
    '''
    ==================================================
    线性时间排序
    ==================================================
    计数排序:把A中元素的值，当作C数组的索引
    时间复杂度O(n+k)
    空间复杂度O(n+k)
    '''
    def countSort(self, a):
        c = []
        b = [None] * len(a)
        maximum, minimum = max(a),min(a)
        k = maximum -minimum + 1 #需要申请的空间数
        #c纪录的是元素的位置信息,c是一个计数收集器。
        c = [0]*k
        #把原数组a中的元素放到合适的位置上,并纪录出现的次数
        for i in a:
            c[i - minimum] += 1
        #calculate the position of ervey element
        for i in range(1,len(c)):
            c[i] += c[i-1]
        #从尾巴开始输出,稳定排序，若变成range(len(a))则变成非稳定排序
        for i in range(len(a) - 1, -1, -1 ):
            countIndex = a[i] - minimum
            b[c[countIndex] - 1] = a[i]
            c[countIndex] -= 1
        return b
    
            
if __name__ == '__main__':
    a = [11,33,13,2,4,7,5]
    example = SortMethod(a)
    b = example.countSort(example.list)
    print(b)
    
    
            
            
        
        
        

