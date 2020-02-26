#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:39:06 2018

learning keras
keras-deep-learning with python

@author: zhaolei
"""

from keras.datasets import mnist

from keras import models
from keras import layers
from keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np


#1、数据处理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#读入的图像是3Dtenshor,(60000,28,28)变成一个2D tensor （60000，28*28）
train_images = train_images.reshape((60000, 28*28))
#让图像在0-1之间，必须除以255
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#2、模型建立
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28, ))) #注意input_shape一定是可迭代对象，注意后面的逗号
network.add(layers.Dense(10, activation = 'softmax'))

#exact purpose of the loss function and optimizer will be made clear throughout the next two chapters.
network.compile(optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])


#3、训练数据
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

#4、评估数据
test_loss, test_acc = network.evaluate(test_images, test_labels)



 #图像   
def example_plot(train_images):
    digit = train_images[4]
    plt.imshow(digit, cmap = plt.cm.binary)
    
    #得到一个14*14的切片
    digts = train_images[:, :14, :14]
    #得到一个 从中心开始的14*14的图。
    train_images[:, 7:-7, 7:-7]
    
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    x = x.copy()
    for i in range (x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

#if __name__ == '__main__':
#    test_loss,test_acc,train_images,test_images = example_keras()
    
    
    
