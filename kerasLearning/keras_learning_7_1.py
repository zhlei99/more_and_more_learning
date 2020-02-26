#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:02:19 2018

@author: zhaolei

本章涉及函数式编程,一个线性的sequential模型不能解决的问题。例如：多输入，多输出，同时输入训练，独立训练，合并，
。本章的内容较多，要在看一遍，7.1小节
"""
from keras import Input, layers
from keras.models import Sequential, Model
import numpy as np
from keras import applications
from keras.datasets import imdb
import keras
from keras.preprocessing import sequence


'''
第一部分 函数式编程与sequential
'''

'''
sequential model
'''
                    
'''
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
seq_model.summary()
'''

'''
function API 与上面有类似结果
'''
#a tensor
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
y = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation = 'softmax')(y)

model = Model(input_tensor, output_tensor)

model.summary()

'''
compile is the same above all
'''
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128)

score = model.evaluate(x_train, y_train)

'''
第二部分 multi-input models
text input and question input then concatenate these vectors ,finally add
a sotfmax classifier on top of the concatenated representations
page238
'''

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
#训练独立的text模型
text_input = Input(shape = (None,), dtype = 'int32', name = 'text')
#embeds the inputs into a sequence of vectors of size 64
embedded_test = layers.Embedding(64, text_vocabulary_size)(text_input)
#encodes the vectors in a single vector via an LSTM
encoded_text = layers.LSTM(32)(embedded_test)

#训练独立的question模型
question_input = Input(shape=(None,),
                       dtype='int32',
                       name='question')

embedded_question = layers.Embedding(
        32,question_vocabulary_size)(question_input)
encode_question = layers.LSTM(16)(embedded_question)
#结合两个独立模型，形成汇总模型
concatenated = layers.concatenate([encoded_text, encode_question], axis = -1)

answer = layers.Dense(answer_vocabulary_size , activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

#feeding data to a multi-input model
num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size,size= (num_samples,max_length))

question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

model.fit([text, question], answer, epochs=10, batch_size=128)

#fitting using a dictionary of inputs(only if inputs are named)
model.fit({'text':text, 'question':question }, answer, epochs = 10, batch_size = 128)


'''
multi-output models
'''
#functional api implementation of a three-output model
vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,
                                 activation='softmax',
                                 name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

#compilation options of a multi-output model:mutiple losses
model.compile(optimizer='rmsprop',
              loss=['mse','categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1.0, 10.])
#equivalent(possible only if you give names to the output layers)
model.compile(optimizer='rmsprop',
              loss={'age':'mse',
                    'income':'categorical_crossentropy',
                    'gender':'binary_crossentropy'},
              loss_weights={'age':0.25,
                            'income':1.,
                            'gender':10.})

#feeding data to a multi-output model
model.fit(posts, [age_targets, income_targets,gender_targets],
          epochs=10, batch_size=64)

model.fit(posts, {'age': age_targets,
                  'income':income_targets,
                  'gender':gender_targets},
            epochs=10, batch_size=64)


'''
第三部分：Directed acyclic graphs of layers
page244
'''
#inception modules
x = ...
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)

branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3,activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

#residual connections,the same size both 
x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.add([y, x])  #adds the original x back to the output features

#different size between x and y
x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

#uses a 1*1 convolution to linerly downsample the original x tensor to the 
#same shape as y
residual = layers.Conv2D(128, 1, strides=2,padding='same')(x)

y = layers.add([y, residual])



'''
how to implement such a model using layer sharing (layer reuse) in the 
Keras functional API
'''
lstm = layers.LSTM(32)

#building the left branch of the model:inputs are variable-length sequences 
#of vectors of size 128
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

#building the right branch of the model :when you call a existing layer instance
#you reuse its weights,共享实例参数
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

#builds the classifier on top
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

#instantiating and training the model:when you train such a model,the weights
#of the LSTM layer are updated based on both inputs
model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)



'''
models as layers

Siamese vision model (shared convolutional base)
'''
y = model(x)
#or
y1, y2 = model([x1, x2])
#the base image-processing model is the Xception network (convolutional base only)
xception_base = applications.Xception(weights = None, include_top = False)
#the inputs are 250*250RGB images
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
#call the same vision model twice
left_features = xception_base(left_input)
right_input = xception_base(right_input)
#the merged features contain information from the right visual feed and the 
#left visual feed
merged_features = layers.concatenate([left_features, right_input], axis=-1)


'''
监控最好的模型
keras.callbacks module includes a number of built-in callbacks

keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger

'''
#the modelcheckpoint and earlystopping callbacks

#callbacks are passed to the model via the callbacks argument in fit,
#which takes a list of callbacks,you can pass any number of callbacks
callbackes_list = [
        #Interrupts training when improvement stops
        keras.callbacks.EarlyStopping(
                monitor='acc',
                patience=1,         #interrupts traning when accuracy has stopped 
                ),                  #improving for more than one epoch (that is , two epochs)
        keras.callbacks.ModelCheckpoint(            #save the current weights after every epoch
                filepath='keras_modelCheckpoint_my_model.h5',
                monitor='val_loss',         #thess two arguments mean you won't overwrite the model file
                save_best_only=True,        #unless val_loss has improved ,which allows you to keep the
                )]                          #best model seen during training

#you mointor accuracy, so it should be part of the model's metrics
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

#note that because the callback will monitor validation loss
#and validation accuracy,you need to pass validation_data to the call to fit
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbackes_list,
          validation_data=(x_val, y_val))

'''
ReduceLROnPlateau callback
'''
#you can use this callback to reduce the learning rate when the validation
#loss has stopped improving
callbacks_list = [
        keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,         #divides the learning rate by 10 when triggered
                patience=10,
                )]
        
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))


'''
writing your own callback
'''
#here is a simple example of a custom callback that saves to disk the cativations
#of every layer of the model at the end of every epoch, computed on the 
#first sample of the validation set

class ActivationLogger(keras.callbacks.Callback):
    
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        #obtains the first input sample of the validation data
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_'+ str(epoch)+'.npz','w')
        np.savez(f, activations)
        f.close()

'''
TensorBoard Tensor面板,以下代码通过，但输入histogram_freq=1,要求输入
embeddings_data，没有搞清楚需要填充什么数据
let's demonstrate these features on a simple example. You 'll train a 1D convent
ont the IMDB sentiment-analysis task.
'''
#text-calssfication model to use with TensorBoard
max_features = 2000
max_len = 500

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len) #把每个数组的长度变成500

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
              loss= 'binary_crossentropy',
              metrics = ['acc'])

#creating a directory for TensorBoard log files
#training the model with a TensorBoard callback
callbacks = [
        keras.callbacks.TensorBoard(
                log_dir='/Users/zhaolei/Desktop/dataset/my_log_dir',
#                histogram_freq=1,   #records activation histograms every 1 epoch
#                embeddings_freq = 1,    #recoreds embedding data every 1 epoch

        )
   ]
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)

#launch tensorBoard from the command line
#tensorboard --logdir = my_log_dir










