#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:01:28 2018

@author: zhaolei

Visualizing Intermediate activations

Use the improved model of keras_learning_5_2_catVsDogs_drop_augment.py
代码无问题，原书代码存在错误。

"""
from keras.models import load_model
from keras.preprocessing import image     #preprocesses the image into a 4D tensor
import numpy as np
import matplotlib.pyplot as plt
from keras import models

model = load_model('/Users/auser/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_small_2.h5')
model.summary()

#Preprocessing a single image
img_path = '/Users/auser/Desktop/dataset/cats_and_dogs_small/test/cats/cat.1700.jpg'

img = image.load_img(img_path, target_size=(150, 150))          
img_tensor = image.img_to_array(img)        #img_tensor.shape = (150, 150, 3)
img_tensor = np.expand_dims(img_tensor, axis = 0)  # img_tensor.shape = (1, 150, 150, 3)
img_tensor /=255.
#display the picture
plt.imshow(img_tensor[0])
plt.show()

#instantiating a model from an input tensor and a list of output tensors
layer_outputs = [layer.output for layer in model.layers[:8]]    #extract the outputs of the top eight layers
#creates a model that will rerurn these outputs ,given the model input
#一个输入，多个输出,one output per layer activation
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

#running the model in predict mode
activations = activation_model.predict(img_tensor)

#visualizing the fourth channel with layer first
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0,:,:,4], cmap = 'viridis')
#visualizing the seven channel with layer first
plt.matshow(first_layer_activation[0,:,:,7], cmap = 'viridis')

#visualizing every channel in every intermediate activation每个一层有32个通道，有8个层
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16
#循环每一个层
for layer_name, layer_activation in zip(layer_names, activations):
    #每一层的通道数=fliter数量
    n_features = layer_activation.shape[-1]     
    #the feature map has shape (l, size, size, n_feautres)
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    #循环每一个filter。tiles each filter into a big horizontal grid
    for col in range(n_cols):
        #循环每一个
        for row in range(images_per_row):
            #print ("col is : "+ str(col))
                #print ("row is : "+ str(row))
                
            channel_image = layer_activation[0,:,:,col * images_per_row + row ]
                                                        
            #视觉可见
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    #display the grid
    scale = 1./ size
    plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    