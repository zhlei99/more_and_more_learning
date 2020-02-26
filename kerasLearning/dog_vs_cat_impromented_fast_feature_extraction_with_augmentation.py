#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:15:05 2018

@author: zhlei
"""

'''
通过已有的模型进行学习，特征提取的两个方法.第一种方法，用以训练好的模型进行图像的特征提取
然后建立新的全连接层进行分类。代价较小，不用增大技术。第二种方法，通过在训练好的模型的conv_base（这部分训练时冻结）,和全连接层
一起重新做训练，这时需要把小数据集进行增大处理，但此方法代价较大。
以下代码模拟第二种方法，此方法需要通过图像的增大处理减少过拟合。
需要GPU。
结论：
可运行,还未执行完全

'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model

'''
第一步：图像处理
'''
base_dir = '/Users/zhaolei/Desktop/dataset/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#Training the convnet using data-augmentation generators
#创建类对象，并初始化,视图化展示图像的增大。
train_datagen = ImageDataGenerator(        
        rescale=1./255,
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)   #测试集不要变形

#Takes the path to a directory & generates batches of augmented data.
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

'''
第二步 定义模型
'''
conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))


conv_base.summary()

model = models.Sequential()
#加入读取的卷积部分
model.add(conv_base)
#层收缩，做全联接
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), 
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

'''
第二种的fine-tuning有2亿个参数，重新计算的成本太高，因此需要冻结conv_base的部分，冻结
操作必须在编译compile之前，否则需要重新编译，要不然则冻结操作会被忽略。

'''
print ("This is the number of trainabel weights"
       "before freezing the conv base:", len(model.trainable_weights))
#freeze a layers
conv_base.trainable = False

print ("This is the number of trainabel weights"
       "after freezing the conv base:", len(model.trainable_weights))
'''
第二点半步 编译
'''
#define compilation step and loss function
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 2e-5),
              metrics = ['acc'])

'''
第三步 训练模型
'''
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

'''
第四步 评估模型
'''

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_feature_extract2.h5')
