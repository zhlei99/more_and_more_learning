#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:49:49 2019

@author: zhaolei
红黑树是一种平衡二叉树
可以保证在最坏的情况下动态集合的操作的时间复杂度为O(lgn)
《算法导论》174页红黑树

红黑树的性质：
1、每个结点或是红色或者黑色。
2、跟结点是黑色的。
3、每个叶结点是黑色的（叶结点即是None结点）
4、如果一个结点是红色的，则它的两个子结点都是黑色的。（反之不成立）
5、对每个结点，从该结点到其所有后代叶结点的简单路径上，均包含相同数目的黑色结点。

为了简便，定义个T.nil（None值）的结点，这个结点是哨兵结点.属性color是黑色，其他属性不限制。所有叶结点与根
结点的父结点都指向这个结点，这样就简化了程序的边界。

一棵有n个内部结点的红黑树的高度至多为2lg(n+1)

重点知识：
（1）左旋、右旋。p176，旋转是以x-y之间的链接作为轴进行。
"""
class Node(object):
    def __init__(self, key):
        self.color = None
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
    
class RedBlackTree(object):
    def __init__(self):
        self.root = None
        self.nil = None #定义一个树的哨兵结点
    
    #把左旋要背下来    
    def left_rotate(self, x):
        #左旋，意味着原来x是y的父结点，x的右孩子是y，旋转后，变成y是x父结点,x是y的左孩子
        #x变成右孩子的左孩子。左旋转。
        y = x.right #辅助变量
        x.right = y.left
        
        if y.left != None: #y的左孩子存在则赋值属性parent
            y.left.parent = x
        y.parent = x.parent 
        
        if x == self.root : #x是根结点
            self.root = y
            
        elif x == x.parent.left :  #若x不是根结点，看是父结点的左孩子还是右孩子。原分之结构不变。
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x   #先搞定y与父结点的关系再搞定y和x的关系
        x.parent = y
        
    def right_rotate(self, x):
        #右旋：x变成左孩子的右孩子
        y = x.left
        x.left = y.right 
        if y.right != None:
            y.right.parent = x
        
        y.parent = x.parent
        if x == self.root:
            self.root = y
        
        elif x == x.parent.left :
            x.parent.left = y
        else:
            x.parent.right = y
            
        y.left = x
        x.parent = y
        
    #插入,插入某一个结点可能会违反rb树规则，所以最后进行调整
    def rb_insert(self, z):
        y = None #辅助位，经过while，能保存父结点。
        x = self.root
        while x != None:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.parent = y.parent #循环结束后y找到就是叶结点（插入位置）的父亲结点.
        if y == self.root: #空树判断
            self.root = z #其他的属性之后操作
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.color = RED
        self.rb_insert_fixup(z) #修正函数
    
    def re_insert_fixup(self, z):  #修正
        
            
            
        
            
        
        
        
            
        
        
        


        


