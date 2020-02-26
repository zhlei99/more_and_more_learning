#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:03:46 2018
fest feature extraction wihtout data augmentation
通过已有的模型进行学习，特征提取的例子，第一种方法，用以训练好的模型进行图像的特征提取
然后建立新的全连接层进行分类。代价较小。此时方法是不适合做图像的增大处理的，由于分段处理。
instantiating the VGG16 convolutional base
结论:acc 0.90，第五代后出现过拟合
@author: zhaolei
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model



base_dir = '/Users/zhaolei/Desktop/dataset/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale = 1./255)  
batch_size = 20

#提取VGG16模型
###########
#第一种策略：将VGG的convent的部分,作为特征提取器，提取原始数据的特征
#然后，将特征作为绸密输入到模型中
#使用ImagenDataGenerator类型中conv_base模型中的predict方法
#
###########
conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))


conv_base.summary()

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count,4,4,512))
    labels = np.zeros(shape = (sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150,150),
            batch_size=batch_size,
            class_mode='binary')
    i=0
    for inputs_batch , labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        #wind up for loop
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

#flatten then to (sample, 8192)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim = 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), 
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels, epochs=30, batch_size = 20 ,
                    validation_data=(validation_features, validation_labels))

#plotting the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_fast_feature_cxtract.h5')

#model = load_model('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_fast_feature_cxtract.h5')