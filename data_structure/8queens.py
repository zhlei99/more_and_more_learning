#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:06:34 2018
Algorithm
@author: zhaolei
"""
import sys


'''
八皇后问题
解体方法
1、设置conflict检测冲突的函数。（下一个位置与之前的皇后是否有冲突）
2、将要放的皇后与之前的没有冲突，就检测是否是最后一个要放皇后，是的话，就生成最后一个皇后的位置。
3、如果不是最后一个皇后，且没有任何冲突，就保存状态继续递归调用。
4、执行list(queens())

'''

def queens(num=8, state=()):
    for pos in range(num):
        if not conflict(state, pos):
            if len(state) == num-1:
                yield (pos,)
            else:
                for result in queens(num, state +(pos,)):
                    yield (pos,)+result
                
def conflict(state, nextX):
    #state  = 位置元组，nextX = 下一个位置
    #纵向查找
    nextY = len(state)
    for i in range(nextY):
        
        if abs( state[i]-nextX ) in (0, nextY-i):#与所有之前的皇后在同一列或斜对角线
            return True
    return False

if __name__ == '__main__':test()


