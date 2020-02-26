#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:00:47 2018

@author: zhaolei
"""

import tensorflow as tf
from numpy.random import RandomState #导入生成模拟数据集
import numpy as np

batch_size = 8  #数据batch 大小

#定义神经网络参数
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1.0, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1.0, seed = 1))
#在shape的一个维度上使用None，可以方便使用不同大小的batch大小。训练时吧数据分成比较小的batch。
x = tf.placeholder(tf.float32, shape = (None,2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None,1), name = 'y-input')

#定义神经网络前向传播,没有激活函数,y是输出元素，y_表示输入变量标签
z = tf.matmul(x, w1)
a = tf.matmul(z, w2)


#定义损失函数，反像传播算法,y值进行截取，不能小于0，不能大于1，
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(a, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

#定义样本值,把Y变成维度为2维数组
Y = [[(int(x1 + x2 < 1))] for (x1, x2) in X]
#Y = np.array(Y)
 
#创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)
    print ("w1:",sess.run(w1))
    print ("w2:",sess.run(w2))
    
    
    #定义训练的次数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size 个样本训练
        start = (i *   batch_size ) % dataset_size
        end = min(start + batch_size, dataset_size)
       # print (i)
        feed_dict = {x:X[start : end], y_:Y[start:end]}
        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict = feed_dict )
        
        if i % 1000 == 0:
            #每隔一段时间计算在所有的数据上的交叉熵并输出。必须保证填入维度与定义维度一样。否则不通过
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print ("After %d training step(s),cross entropy on all data is %g "%(i,total_cross_entropy))
    print ("w1:",sess.run(w1))
    print ("w2:",sess.run(w2))
    
    





