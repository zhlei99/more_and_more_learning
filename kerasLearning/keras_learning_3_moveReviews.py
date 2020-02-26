#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:53:33 2018

@author: zhaolei
"""
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words = 10000)

word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
#get 方法若找不到，则默认?
decode_review = ' '.join(
        [reverse_word_index.get(i-3, '?') for i in train_data[0]])

#one-hot encode
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#定义模型结构
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#定义损失函数
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics = ['acc'])

#定义交叉验证，拟合数据
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#plotting the training and validation loss
history_dic = history.history
loss_values = history_dic['loss']
val_loss_values = history_dic['val_loss']
acc= history.history['acc']


epochs = range(1,len(acc)+1)
#epochs = range(1,21)


plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#等价表达
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss='binary_crossentropy',
#              metrics = ['accuracy'])
#等价表达
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss='binary_crossentropy',
#              metrics = metrics.binary_accuracy)





