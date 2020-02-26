#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:14:10 2019

@author: zhaolei
队列
队列的形式分为两种存储方式:数组和链表
用链表进行存储

队列的数组形式，直接用数组就可以模拟实现。若要包装成数据结构，
在加上长度
"""
class Node(object):
    def __init__(self,elem):
        self.data = elem
        self.next = None
        
class Queue_learing(object):
    def __init__(self):
        self.head = None #头部结点，用来出队列，指向队头元素
        self.tail = None #尾部结点，用来入队列,指向队尾元素
    
    def is_empty(self):
        return self.head is None
    
    def enquenue(self, elem):
        '''
        1、如果队列为空，进行入队
        2、如果队不空，入队
        3、尾结点入队
        '''
        p = Node(elem)
        if not self.head: #链表为空，则把此结点赋值给头与尾结点
            self.head = p
            self.tail = p
        else:
            self.tail.next = p #尾结点的后继结点与p相链接
            self.tail = p #尾结点后移
            
    def dequeue(self):
        '''
        出栈要判空
        头结点出队
        '''
        if not self.head :
            raise IndexError('queue is empty')
        else:
            temp = self.head.data
            self.head = self.head.next  #头指针后移
            return temp
        
    def peek(self):
        '''
        查看队列头元素
        '''
        if not self.head: #队列为空
            print('队列为空')
            return
        else:
            return self.head.data #返回对头部元素
            
        
if __name__ == '__main__':
    queue = Queue_learing()
    queue.enquenue(1)
    queue.enquenue(2)
    queue.enquenue(3)
    
    temp = queue.dequeue()
    print (temp)
    
    print('============')
    cur = queue.head
    while cur:
        print(cur.data)
        cur = cur.next

