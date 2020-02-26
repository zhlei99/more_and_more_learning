#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:05:44 2019

@author: zhaolei
实现二叉搜索树
1、若左子树不为空，则左子树的所有结点值均小于或等于它的根结点的值。
2、若右子树不为空，则右子树的所有结点的值均大于或等于它的跟结点的值。
3、左右子树也分别为二叉搜索树
《算法导论》167页
"""

class Node(object):
    def __init__(self, item):
        self.value = item
        self.left = None
        self.right = None
        self.parent = None
        
    def __str__(self):
        #print 打印类
        return str(self.value)
    
class BSTree(object):
    def __init__(self):
        self.root = Node('root') #根结点定义为root，作为哨兵使用，永不删除
    
    #中序遍历一个以x为结点的树
    def inorder_tree_walk(self, x):
        if x:
            self.inorder_tree_walk(x.left)
            print(x)
            self.inorder_tree_walk(x.right)
        
    #查询二叉搜索树
    def tree_search(self, x, k):
        if (not x) or x.value == k:
            return x
        if x.value < k:
            return self.tree_search(x.left, k)
        else:
            return self.tree_search(x.right, k)
            
    #迭代树的搜索算法效率一般要高于递归,x 是根结点
    def iterative_tree_search(self, x, k):
        while x!=None and x!= k:
            if x < k:
                x = x.left
            else:
                x = x.right
        return x   #x为空或者x= k
    
    #查找最大最小关键字:最左就是最小，最右就是最大。
    def tree_minimum(self, x):
        while x.left:
            x = x.left
        return x
        
    #插入,T表示树，z表示要插入的元素,node类型
    def tree_insert(self, z):
        '''
        1、先找到要插入的位置。
        2、插入位置为根结点、左孩子、右孩子
        '''
        y = None  #设置辅助位,防止父结点丢失
        x = self.root #x 为移动指针
        while x:
            y = x
            if x.value < z.value:
                x = x.left
            else:
                x = x.right
        z.parent = y  #把父结点的属性赋值上
        if y == self.root:
            self.root = z
        elif z.value < y.value:
            y.left = z
        else :
            y.right = z
    
    
    def transplant(self, u, v):
        '''
        定义一个替换结点函数（辅助删除函数的,用v结点替换u结点。只操作其父结点，而孩子结点在tree_delete中完成
        输入为:u,v Node类型
        思想：
        1、u是根结点，没有父结点
        2、u是父结点的左孩子
        3、u是父结点的右孩子
        '''
        if u == self.root:
            self.root = v
        elif u.parent.left == u:
            u.parent.left = v
        else:
            u.parent.right = v
        if v:  #有可能v是None结点，即删除u后没有结点可替换
            v.parent = u.parent
        
        
    #删除
    def tree_delete(self, T, z):
        '''
        思想：
        删除要考虑四种情况：
        1、删除的结点z，没有任何孩子结点。直接删除z,修改其父结点属性。
        2、删除的结点z,只有左孩子结点，或只有右孩子结点，则直接把左或右孩子替换自己。直接替换
        3、删除的结点z，同时有左孩子与右孩子结点y、右孩子y没有左孩子。
        （把z的左孩子赋值给y的左孩子，把z的右孩子赋值给y的右孩子.（y取代z的位置）
        4、删除的结点z，同时有左孩子与有孩子结点、右孩子有左孩子。 （把步骤4的情况，先变成3,再处理）       
            （1）找到z结点的后继，即在z的右子树中的最小结点y，从z右子树找最左结点（左孩子为空）。
            （2）把y有右孩子（注意y没有左孩子，右孩子可能为空，可能不为空），则用y的右
            孩子取代y的位置。（做完这一个步骤4，变为3的情况）
            (3)把z的左孩子赋值给y的左孩子，把z的右孩子赋值给y的右孩子.（y取代z的位置）
        '''
        if z.left == None:
            self.transplant(z, z.right)
        elif z.right == None:
            self.transplant(z, z.left) #设置其夫结点，其他结构都不变
        else:
            y = self.tree_minimum(z.right) #找到z右孩子的左小结点，即z的后继结点。
            
            if y != z.right: #即第4中情况，先变成第三种情况
                self.transplant(y, y.right)  #用y的右孩子替换y的位置
                y.right = z.right    #设置其右孩子
                y.right.parent = y  #设置其父结点
            
            self.transplant(z, y)  #改变y的父结点
            y.left = z.left
            y.left.parent = y  #设置左孩子的父亲结点。
                
            
                
                
                
        
        
            
                
        
        
        
            
        
        
        