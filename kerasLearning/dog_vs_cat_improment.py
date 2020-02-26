#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:11:37 2019
dog Vs cat improvement
相比最早的技术，添加dropout技术，图像变换进行数据集变大技术
效果提升：提升到82%，提升了15%
进一步提升方案：regularization techniques , tuning the network's parameters(such
as the numbers of filters per convolution layer, or the number of layers in the network
)likely up to 86% or 87%
@author: zhaolei
"""

import os
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image

"""
第一步：对数据进行处理：加载训练集与验证集与测试集
"""
base_dir = '/Users/auser/Desktop/dataset/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')

'''
图像变换增大数据集技术
'''
#创建类对象，并初始化,增大图像。
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
看生成器的内容用例
'''
#for data_batch, labels_batch in train_generator:
#    print('data batch shape:',data_batch.shape)
#    print ('labels batch shape:',labels_batch.shape)
#    break

'''
displaying some randomly augmented training images
'''
#fnames = [os.path.join(train_cats_dir, fname) for
#          fname in os.listdir(train_cats_dir)]

#img_path = fnames[3]

#img = image.load_img(img_path, target_size=(150,150))
#x= image.img_to_array(img)  #converts it to a numpy array with shape(150,150,3)

#x = x.reshape((1,) + x.shape) #reshape it to (1,150,150,3)

#看生成器的生成过程
#i = 0
#for batch in datagen.flow(x, batch_size=1):
#    plt.figure(i)
#    imgplot = plt.imshow(image.array_to_img(batch[0]))
#    i += 1
#    if i % 4 == 0 :
#        break
#plt.show()

"""
第二步：定义模型
增加dropout 技术
"""
#instantlating a small convet for dogs vs.cats classification
#define model layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation= 'relu', 
                        input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
#收缩、进行全链接
model.add(layers.Flatten())
#加入dropout防止全联接层的过拟合
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#检查模型
model.summary()

#define compilation step and loss function
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc'])

'''
第三步：训练模型
epochs 调整后应该设置成100，或者更长
'''
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)
       
#saving the model
#model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_improved_1.h5')


'''
第四步：评估模型、展示信息
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


    


