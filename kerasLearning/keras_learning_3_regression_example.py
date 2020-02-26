#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:25:20 2018

@author: zhaolei
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

#1、数据处理\数据类型、输入维度调整.reshape()
(train_data, train_targets), (test_data, test_targets) =boston_housing.load_data()

#normalizing the data,
#均值，列方向（属性方向求平均），注意先均值在方差，不可调换顺序，否则差距很大。
mean = train_data.mean(axis = 0)
train_data -=mean
#标准差，列方向
std = train_data.std( axis = 0 )
#数据归一化
train_data /= std
test_data -=mean
test_data /= std

#2、建立模型
#model definition,input_shape每个样本输入特征。
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,
                           activation='relu',
                           input_shape = (train_data.shape[1],)))
#应用正则化的层。    
#    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), 
#                           activation= 'relu', input_shape = (10000,)))
    #使用dropout，轻易改变系统性能。
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64,
                           activation='relu',))
    #使用dropout
    #model.add(layers.Dropout(0.5))
    #没有activation，只有一个值，是一个线性层，典型回归设置
    model.add(layers.Dense(1))
    
    #lost function，适合回归任务的损失函数
    model.compile(optimizer='rmsprop',
                  loss = 'mse',
                  metrics=['mae'])
    return model

#3、评估模型、并调整参数，防止过拟合，等其他参数，评定模型足够优秀。
#交叉验证，评估模型。
k = 4
num_val_samples = len(train_data)//k  #长度划分
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    #切片处理
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i+1) * num_val_samples]
    
    #通过交叉验证，划分训练集与验证集。一个划分为验证集其他的划分为训练集,联结其他的划分
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
            train_data[(i+1)*num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i*num_val_samples],
            train_targets[(i+1)*num_val_samples:]],
            axis = 0)
    
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=1)
    mae_history = history.history['val_mean_absolute_error']
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 1)
    all_mae_histories.append(mae_history)

#列方向对k个数据进行求均值，最后生成一个行方向数据，数据显示为均值，然后输出
average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#plot valldation scores
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#plotting valldation scores, excluding the fist 10 data point
#指数加权，评估模型的图像。在80代达到最优，超过80代，开始overfitting
def smooth_cureve(points, factor =0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            preious = smoothed_points[-1]
            smoothed_points.append(preious * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_cureve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#4、最终调整模型，进行模型的最后训练。
#training the final model,不进行交叉验证。并用测试集测试。
model = build_model()
model.fit(train_data, train_targets,
          epochs = 80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)



    
#
