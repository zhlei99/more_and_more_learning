#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:49:49 2019

@author: zhaolei
单链表操作
"""
#创建Node类
class Node:
    def __init__(self, data):
        self.data = data 
        self.next = None
#创建Linked_List类
class Single_Linked_List:
    def __init__(self, head = None):
        self.head = head #链表头指针
    
    def is_empty(self):
        return not self.head        
    
    def append(self, new_element): #增加元素要判断链表是否为空
        current = self.head  #得到当前链表的头指针
        if self.head:  #头指针存在,则循环遍历
            while current.next: #循环找到最后一个结点
                current = current.next               
            current.next = new_element #进行新结点链接
        else:  #头部结点不存在时
            self.head = new_element
            
    def get_length(self):
        '''
        返回链表长度
        '''
        temp = self.head
        length = 0
        while temp != None:
            length += 1
            temp = temp.next
        return length
                        
    def insert(self, position, new_element:Node):
        '''
        单链表插入：
        0、输入是0-(length -1),new_element 是Node
        1、索引是否在范围内
        2、插入的是否为头结点
        3、插入一般结点
        '''               
        if position < 0 or position > self.get_length() :
            raise IndexError('插入超出范围')
            
        temp = self.head
        if position == 0:
            new_element.next = temp
            self.head = new_element #更改链表头结点属性
            return 
        
        #遍历找到位置结点，在结点前面插入结点
        i = 0
        while i < position - 1: #找到前一个结点的位置
            i += 1
            temp = temp.next
        new_element.next = temp.next  #循环退出，temp指向前一个结点元素，要不就丢失前驱结点
        temp.next = new_element  #链接新结点的前驱
        
    def remove(self, item):
        '''
        0、item 是结点值
        1、删除结点考虑链表为空
        2、删除的元素没有找到
        3、删除的元素找到，删除后链表为空
        4、删除的元素找到，删除后链表不为空
        '''
        current = self.head #当前头结点
        pre = None
        
        if not self.head:
            raise IndexError('链表为空，没找到结点')
        while current :
            if current.data == item: #查找需要删除数据
                if current == self.head:  #需要删除的数据是头结点
                    self.head = current.next
                    return
                else:
                    pre.next = current.next #释放中间结点
                    return
            else:
                pre = current  #没找到结点，就保存前驱
                current = current.next #当前结点下移
        raise IndexError('没有找到所要删除的结点') 
        
    def reverse(self):
        '''
        链表反转
        '''
        pre = None
        current = self.head
        while current:
            next_node = current.next  #假设已经有三个结点都已经设置好
            current.next = pre  #反转链表
            pre = current  #下移
            current = next_node #下移
        self.head = pre #循环结束后，pre是最后一个结点。
        
    def initlist(self, data_list:list):
        '''
        列表转换为链表
        data_list: 数组
        '''
        if data_list:
            #创建头结点
            self.head = Node(data_list[0])
            temp = self.head
            for i in data_list[1:]:
                node = Node(i)
                temp.next = node
                temp = temp.next
            

                        
if __name__ == '__main__':
    sing_list = Single_Linked_List()
    node = Node(1)
    sing_list.insert(0,node)
    node = Node(2)
    
    sing_list.insert(0,node)    #插入 
    
    sing_list.initlist([1,2,3,4,5]) #初始化
    
    sing_list.remove(1)
    #sing_list.remove('')
    sing_list.remove(5)
    sing_list.remove(4)
    sing_list.remove(3)
    sing_list.remove(2)
    sing_list.remove(5)
    current = sing_list.head #打印
    while current:
        print (current.data)
        current = current.next
    
    
                                

            
        
        
            
            
            
        
            
            
        


