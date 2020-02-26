#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:35:03 2019
dogs Vs cat data 分类问题
acc = 0.7   val_acc = 0.7  overfitting
@author: zhaolei
"""
import os, shutil
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

"""
第一步：对数据进行处理：训练集与验证集与测试集
"""

#图片split,is_split_flag,表示是否需要把文件进行分割，一般只进行一次分割
def split_original_images(is_split_flag = False):
    original_dataset_dir = '/Users/zhaolei/Desktop/dataset/kaggle/train'
    
    base_dir = '/Users/zhaolei/Desktop/dataset/cats_and_dogs_small'
    
       
    #当前路径
    #current_dir = os.getcwd()
    if not os.path.exists(base_dir):        
        os.mkdir(base_dir)
    #把原数据分成对猫和狗的不同的训练集与验证集与测试集
    #directories for the training, validation, and test splits
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)
    
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)
    
    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)
    
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)
    
    #如果已经做过就不在做了
    if is_split_flag :
        #copies the first 1000 cat images to train_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)
        #copies the next 500 cat images to validation_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)
        #copies the next 500 cat images to test_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)
    
        #copies the first 1000 dogs images to train_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)
        #copies the next 500 dog images to validation_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)
        #copies the next 500 dog images to test_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)
    
    return (original_dataset_dir,base_dir,train_dir,validation_dir,
            test_dir,train_cats_dir,train_dogs_dir,validation_cats_dir,
            validation_dogs_dir,test_cats_dir,test_dogs_dir)    

(original_dataset_dir,base_dir,train_dir,validation_dir,
 test_dir,train_cats_dir,train_dogs_dir,validation_cats_dir,
 validation_dogs_dir,test_cats_dir,test_dogs_dir) = split_original_images(False)        

#以上数据单元测试用例：
def test_ImagePreprocessing(train_cats_dir,train_dogs_dir,validation_cats_dir,
                            validation_dogs_dir,test_cats_dir,test_dogs_dir):    
    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test cat images:', len(os.listdir(test_cats_dir)))
    print('total test dog images:', len(os.listdir(test_dogs_dir)))
    
#################################################################
#优化操作    
#对图像进行预处理,图片裁剪，用生成器对对象Batch进行训练，高效，内存利用率高。   
#Using ImageDataGenerator to read images from directories
#train_datagen = ImageDataGenerator(rescale=1./255)  #rescales all images by 1/255
#################################################################
train_datagen = ImageDataGenerator(rescale=1./255)

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

#看生成器的内容用例
#for data_batch, labels_batch in train_generator:
#    print('data batch shape:',data_batch.shape)
#    print ('labels batch shape:',labels_batch.shape)
#    break
##################################################################
#
#
##################################################################
"""
第二步：定义模型
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

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#检查模型
model.summary()

#define compilation step and loss function
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc'])

#检查模型结构
model.summary()

'''
第三步：训练模型
'''
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

#saving the model
model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_small_1.h5')
    
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
