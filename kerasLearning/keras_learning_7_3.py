#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:27:59 2018

@author: zhaolei

"""

from keras import Input, layers
from keras import models
import numpy as np
from keras import applications
from keras.datasets import imdb
import keras
from keras.preprocessing import sequence

'''
example of Batch Normalization 
'''
model = models.Sequential()
model.add(layers.Conv2D(32,3,activation='relu', input_shape= (250,250,3)))
model.add(layers.BatchNormalization())


'''
depthwise separabel convolution

'''
height = 64
width = 64
channels = 3
num_classes = 10

model = models.Sequential()
model.add(layers.SeparableConv2D(32,3, activation='relu',
                                 input_shape=(height, width, channels,)))
model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.SeparableConv2D(128,3,activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.SeparableConv2D(128, 3,activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes,activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')