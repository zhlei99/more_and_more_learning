#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:17:13 2019

@author: zhaolei
"""

'''
查找
'''
class SearchMethod():
    
    def __init__(self, a):
        self.list = a
    
    '''
    二分查找
    a = [1,3,5,7,9,10]
    时间复杂度O(lgn)
    空间复杂度O(1)
    二分查找：查找操作快，插入慢，由此引入BST（二叉查找树）
    '''
    def binarySearch(self, a, key, low, high):
        if not a:
            return "error, a is Null"
        mid = int((low + high)/2)
        print(a[mid])
        if low > high : return -1
        if a[mid] == key:
            return mid
        elif key < a[mid]:
            return self.binarySearch(a, key, low, mid-1)
        else :
            return self.binarySearch(a, key, mid + 1, high)
    
    
       
        
if __name__ == '__main__':
    a = [1,3,5,7,9,10]
    p = SearchMethod(a)
    mid = p.binarySearch(p.list, 3, 0, len(p.list) - 1)
    print(mid)
    
            
        