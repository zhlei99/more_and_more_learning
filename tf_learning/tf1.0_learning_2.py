#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 09:40:27 2018
learning tensorflow
@author: zhaolei
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np


v = tf.get_variable("fv", shape = [1,2])


#cs = tf.get_variable_scope().reuse_variables()


with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)) as foo_scope:
    v1 = tf.get_variable("v", [1])
    assert v.eval() == 0.4
    print (v1)
with tf.variable_scope(foo_scope):
    v2 = tf.get_variable("v", [1])
    print(v2)
    

tf.name_scope()

tf.nn.separable_conv2d()
tf.nn.avg_pool

tf.nn.sigmoid_cross_entropy_with_logits()
tf.nn.softmax_cross_entropy_with_logits()


#——————————————————————#  
#定义权重函数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#定义一个模型：
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    #第一全连接层
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    #第二个全连接层
    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    #输出预测值
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

#加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
                    mnist.test.labels
#
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

#初始化权重参数
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

#生成网络模型,输入层/隐层dropout初始值，创建得到预测值变量
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X,w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

#定义损失函数，
cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
#训练操作
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#预测操作
predict_op = tf.argmax(py_x, 1)

#存储操作
ckpt_dir = "./chpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

#定义一个计数器，为训练轮输计数
global_step = tf.Variable(0, name='global_step', trainable = False)

#在生命完所有的变量后，调用tf.train.Saver
saver = tf.train.Saver()
#位于tf.train.Saver之后的变量不会被存储
non_storable_variable = tf.Variable(777)

#训练模型并存储
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    start = global_step.eval()
    print ("start from: ",start)
    
    for i in range(start,100):
        #以128作为batch_size
        for start, end in zip(range(0,len(trX),128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict = {X:trX[start:end], Y:trY[start:end]},\
                                            p_keep_input = 0.8 , p_keep_hidden=0.5)
            global_step.assign(i).eval()    #更新计数器
            saver.save(sess, ckpt_dir + "./model.ckpt", global_step= global_step)
            print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, 
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
    
#加载模型
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        print (ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)#加载所有参数
        #从这里开始就可以直接使用模型进行预测，或者接着训练


#——————————————————————#          
#队列操作
q = tf.FIFOQueue(3,"float")
init = q.enqueue_many(([0.1,0.2,0.3],))
#定义出队，+1，入队操作
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    sess.run(init)
    quelen = sess.run(q.size())
    for i in range(2):
        sess.run(q_inc)
    quelen = sess.run(q.size())
    for i in range(quelen):
        print (sess.run(q.dequeue()))
        
        
#——————————————————————#        
#创建含有队列的图，此代码一直循环而不终止。
q = tf.FIFOQueue(1000, "float")        
counter = tf.Variable(0.0)  #计数器
increment_op = tf.assign_add(counter, tf.constant(1.0))#操作：给计数器加1
enqueue_op = q.enqueue(counter) #操作：把操作入队列

#创建一个队列管理器，用两个操作像q中添加元素。
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

#启动一个会话，从队列管理器qr中创建线程,
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #创建协调器（coordinator)来管理线程,协调线程的关系，用来做同步
    coord = tf.train.Coordinator()
    #启动入队线程，协调器是线程的参数
    enqueue_threads = qr.create_threads(sess, coord = coord, start = True)
    coord.clear_stop()#通知其他线程关闭
    #主线程
    for i in range(10):
        try:
            
            print (sess.run(q.dequeue()))
        except tf.errors.OutOfRangeError:
            break
    
    coord.join(enqueue_threads) #join 操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回。


#——————————————————————#      
#预加载数据
x1 = tf.constant([2,3,4])
x2 = tf.constant([4,0,1])
y = tf.add(x1, x2)

#填充数据
a1 = tf.placeholder(tf.int16)
a2 = tf.placeholder(tf.int16)
b = tf.add(a1, a2)

li1 = [2,3,4]
li2 = [2,12,4]

with tf.Session() as sess:
    print (sess.run(b, feed_dict = {a1: li1, a2: li2} ))
    
#——————————————————————#  
#从文件读取数据：
#（1）把样本数据写入 TFRecord二进制文件（2）从队列中读取
def main(unused_argv):
    #获取数据
    data_sets = 




    
            


                    
