#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:31:04 2019

@author: zhaolei
"""
'''
分治策略
1、分解；2、解决；3、合并
'''
class DivideAndConquer():
    
    def __init__(self, a):
        self.list = a
    
    '''
    求解最大子数组问题
    A = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    画成树
    
    实质：若要找到和最大，假设数组为[1,2,3,4],则要检查[1],[2],[3],[4],[1,2],[3,4],[2,3],[1,2,3,4]
    递归到最后，则检查了单个子数组，返回最大值。
    merge中，检查，由中心向外扩展的所有组合，返回最大的组合。
    '''
    def find_maximum_subarray(self, a:list, low, high):
        if low == high:
            return (low, high, a[low])
        else:
            mid = int((low + high)/2)
            #向左边求和
            (left_low, left_high, left_sum) = self.find_maximum_subarray(a, low, mid)
            #向右边求和
            (right_low, right_high, right_sum) = self.find_maximum_subarray(a, mid+1, high)
            #相当于merge操作
            (cross_low, cross_high, cross_sum) = self.find_maximum_cross_subarray(a, low, mid, high)
            
            if left_sum >=right_sum and left_sum >= cross_sum:
                print("return left:" , left_sum)
                return(left_low, left_high, left_sum)
            elif right_sum > left_sum and right_sum >= cross_sum:
                print("return right:" , right_sum)
                return(right_low, right_high, right_sum)
            else:
                print("return cross:" , cross_sum)
                return(cross_low, cross_high, cross_sum)
            
    def find_maximum_cross_subarray(self, a, low, mid, high):
        #left_list = [low,...,mid] right_list = [mid+1,...,high]

        left_sum = float('-inf') #用来存储当前最大和,初始化负无穷
        sum_num = 0
        #由中点向低处检查数组和
        for i in range(mid,low-1,-1):
            sum_num = sum_num + a[i]
            if sum_num > left_sum:
                left_sum = sum_num
                max_left = i

        right_sum = float('-inf') #存储当前最大和，初始化无穷
        sum_num = 0
        #由中心点开始向高出检查
        for j in range(mid+1, high+1):
            sum_num = sum_num + a[j]
            if sum_num > right_sum:
                right_sum = sum_num
                max_right = j
        print ("a:",a[low:high+1])
        print ("max_left:", max_left)
        print ("max_right", max_right)
        return (max_left, max_right, left_sum + right_sum)
        


if __name__ == '__main__':
    p = DivideAndConquer([13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7])
    low, high, sum = p.find_maximum_subarray(p.list, 0, len(p.list)-1)
