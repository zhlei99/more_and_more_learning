#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:32:37 2019

@author: zhaolei
"""
#栈操作

class Stack(object):
    def __init__(self, limit = 10):
        self.stack = []  #存放元素
        self.limit = limit  #栈内容极限
    
    def push(self, data):
        #入栈判满，判断是否溢出
        if len(self.stack) >= self.limit:
            raise IndexError('超出栈容量极限')
        self.stack.append(data)
    
    def pop(self):
        #出栈,判空
        if self.stack:
            return self.stack.pop()
        else:
            raise IndexError('stack is empty')
            
    #查看栈最上面元素,判空
    def peek(self):
        if self.stack:
            return self.stack[-1]
    
    #判断栈为空,空返回真，不空返回假
    def is_empty(self):
        return not bool (self.stack)
    
    #返回栈的大小
    def size (self):
        return len(self.stack)
    
'''
栈的应用
'''
#括号匹配
'''
思考过程：
1、输入： 一个有序字符串s, 假设s的字符都在括号里
2、输出：此字符串是否匹配
3、边界条件：1、空字符串 2、匹配的字符串 3、不匹配字符串 4、最后栈不空返回false
4、问题划分：栈的问题
5、测试用例 s = '', '(', '}','([)]','(())'

'''
def balanced_parentheses(parentheses):
    stack_temp = Stack(len(parentheses))
    dict_temp = {')':'(', ']':'[','}':'{'}
    if not parentheses:  #空返回真
        return True
    for i in range(len(parentheses)): 
        if parentheses[i] in dict_temp.values():    #左括号入栈，右括号出栈匹配
            stack_temp.push(parentheses[i])
        else:
            if stack_temp.is_empty(): #输入的第一个元素为右括号,且栈为空，返回false
                return False
            top_a = stack_temp.pop()                
            if top_a != dict_temp[parentheses[i]]:
                return False
    return stack_temp.is_empty()
        
if __name__ == '__main__':
    test_exa = ['','(','}','([)]])','(([[]])','([{}])']
    
    for ex in test_exa:
        print(ex + ':' + str(balanced_parentheses(ex)))
                    
                
        

