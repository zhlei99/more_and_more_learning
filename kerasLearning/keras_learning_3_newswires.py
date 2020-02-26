#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:12:51 2018

@author: zhaolei
"""

from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers as op
import matplotlib.pyplot as plt
from keras import losses 

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=10000)

#decode newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items() ])
decoded_newswire = ''.join([reverse_word_index.get(i - 3, '?') for i in 
                            train_data[10]])

#one-hot encode，trainset and testset
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#one-hot encode label
#def to_one_hot(lables, dimension = 46):
#    results = np.zeros((len(lables), dimension))
#    for i, lables in enumerate(lables):
#        results[i, sequence] = 1.
#    return results
#
#one_hot_train_labels = to_one_hot(train_data)
#one_hot_test_labels = to_one_hot(test_data)

#built-in way to do one-hot
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#model definition
model = models.Sequential()
#第一层
model.add(layers.Dense(64, activation='relu', input_shape = (10000,)))
#第二层
model.add(layers.Dense(64, activation='relu'))
#输出层
model.add(layers.Dense(46, activation='softmax'))

#op.Adam()
#compile model,定义损失函数
model.compile(optimizer= 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


#setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))
#在测试集上，验证模型效果
result = model.evaluate(x_test, one_hot_test_labels)

#generating predictions for new data
predictions = model.predict(x_test)


#画图 plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epoches = range(1, len(loss)+1)
plt.plot(epoches, loss, 'bo', label='Training loss')
plt.plot(epoches, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#plotting the training and validation accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

epoches = range(1, len(loss)+1)
plt.plot(epoches, acc, 'bo', label='Training acc')
plt.plot(epoches, val_acc, 'b', label = 'Validation val_acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


