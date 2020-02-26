#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:14:07 2018

@author: zhaolei
"""

import tensorflow as tf

#output tensor value    
def output_value(value):
    with tf.Session() as sess:
        #An Op that initializes global variables in the graph.
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print (value.eval())

a = tf.constant([1.0,2.0],name = "a")
b = tf.constant([2.0, 3.0],name = "b")
result = a + b

#print (a.graph is tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    #在计算图g1中定义变量“v”，并设置初始值为0
    v = tf.get_variable(
            "v",initializer = tf.zeros(shape = [1]))
print (g1 is tf.get_default_graph())  
    
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
            "v", initializer = tf.ones(shape = [1]))
print (g2 is tf.get_default_graph()) 
        
#在计算图g1中读取变量v的取值
with tf.Session(graph = g1) as  sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print (sess.run(tf.get_variable("v")))
        
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse = True):
        print (sess.run(tf.get_variable("v")))

#标准写法
##a.dtype.is_compatible_with(b.dtype)    比较两个张量的类型是否相等
#输出张量中的数值。先创建会话，创建操作，然后用会话运行操作，打印       
weights = tf.Variable(tf.random_normal([2,3],stddev = 2))
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print (weights.eval())

    
biases = tf.Variable(tf.zeros([2,3]))
w2 = tf.Variable(weights.initialized_value())            

#feedforward
#这里通过seed参数设定了随即种子，这样可以保证每次运行得到的结果是异样的。
w1 = tf.Variable(tf.random_normal([2,3],stddev =1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev =1, seed = 1))
#x = tf.constant([0.7,0.9],shape = [1,2])
#定义placeholder作为输入数据的地方，维度的定义用于降低出错概率。一旦应用placeholder,则需要显示
#的定义feed_dict变量。
x = tf.placeholder(tf.float32, shape = [1,2],name = "input")    
#Multiplies matrix a by matrix b, producing a * b.
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
with tf.Session() as sess:
    #初始化
    init_op=tf.global_variables_initializer()
    #or  sess.run(w1.initializer)
    sess.run(init_op)
#    print (sess.run(y))
    print (sess.run(y,feed_dict = {x:[[0.7,0.9]]})) 
#    print (sess.run(y,feed_dict = {x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]})) #feed_dict,是3个样本输入
    
#w1.assign(w2 )  
    
#定义placeholder作为输入数据的地方，维度的定义用于降低出错概率。一旦应用placeholder,则需要显示
    #的定义feed_dict变量。

#定义损失函数来刻画预测值与真实值的差距，1e-10 = 1*10^10
cross_entropy =  tf.reduce_mean(y * tf.log(tf.clip_by_value(y, 1e-10,1.0)))
#定义反向传播算法来优化神经网络中的参数
learning_rate = 0.001
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)




    

    

