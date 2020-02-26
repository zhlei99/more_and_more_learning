#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 08:56:46 2018

@author: zhaolei

此程序还未执行。
"""
'''
fine-tuning a network are as follow:
1、Add your custom netowrk on top of an already-trained base network
2、freeze the base network
3、Train the part you added
4、Unfreeze some layers （2-3layers） in the base network
5、jointly train both these layers and the part you added
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

#Takes the path to a directory & ge nerates batches of augmented data.
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
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#freezing all layers up to a specific one
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainabel = False

'''
第2.5步，编译模型
'''        
#fine-tuning the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

'''
第三步 训练模型
'''

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_data=validation_generator,
        validation_steps = 50)

'''
第四部 评估模型，画出图
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

#去掉图像噪音，通过指数加权平均
#smoothing the plots
def smooth_curve(points, factor = 0.8):
    smoothed_point = []
    for point in points:
        if smoothed_point:
            previous = smoothed_point[-1]
            smoothed_point.append(previous * factor + point * (1 - factor))
        else:
            smoothed_point.append(point)
    return smoothed_point

plt.plot(epochs,
         smooth_curve(acc),'bo',label = 'Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label  = 'Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, 
         smooth_curve(loss), 'bo', label = 'Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label = 'Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#用测试数据评估模型
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_small_fine_tuning.h5')

    
