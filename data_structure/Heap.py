#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:35:31 2019

@author: zhaolei
"""

'''
堆排序
索引是从零开始 
维护堆的时间复杂度O(h)
建堆的时间复杂度O(nlgn)  
堆排序时间复杂度O(nlgn)
'''
class Heap():
    
    def __init__(self,a):
        self.list = a
        self.heap_size = len(a)
    
    #堆的结点索引，父节点，左右节点
    def PARENT(self, i):
        return (int((i-1)/2))
    
    def LEFT(self, i):        
        return 2*i + 1
    
    def RIGHT(self,i):
        return 2*i+2
    
    def exchange(self, a, i, j): #把a数组里面的第i个元素与j个元素交换位置
        temp = a[i]
        a[i] = a[j]
        a[j] = temp
        return a
    
    #大顶堆维护,假设堆已经维护好，检查第i个元素
    def MaxHeapify(self, a, i):
        l = self.LEFT(i)
        r = self.RIGHT(i)
        if l < self.heap_size and a[l] > a[i]:
            largest = l
        else:
            largest = i
        if r < self.heap_size and a[r] > a[largest]:
            largest = r
        if largest !=  i: #需要调整
            self.exchange(a, i, largest)
            self.MaxHeapify(a, largest) #顺序调换后，可能还是违反堆的性质。
    #建堆
    def BuildMaxHeap(self, a):
        n = len(a)
        #子树a[int(n/2),...n]都是树的叶结点,所以要倒序拜访所有跟结点。
        for i in range(int((n-1)/2), -1, -1):
            print(a[i])
            self.MaxHeapify(a, i)
        return a
    
    #堆排序,倒序，堆顶元素与最后一个元素互换。大顶堆最大元素是a[0]
    def heapsort(self, a):
        self.BuildMaxHeap(a)
        print(a)
        for i in range(len(a)-1, 0, -1):
            print(a[0])
            self.exchange(a, 0, i) #交换最大元素与当前的i元素
            self.heap_size -= 1 #排序好的元素,不用再排序了
            self.MaxHeapify(a, 0) #维护大顶堆
        #恢复原样堆大小
        self.heap_size = len(a)
            
            
            
if __name__ == '__main__':
    
    a = [7,10,3,9,16,1]
    p = Heap(a)
    p.heapsort(a)
    
        
            
            
            