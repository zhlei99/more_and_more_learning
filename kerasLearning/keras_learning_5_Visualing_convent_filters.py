#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:38:35 2018

对VGG16的每层的filter 进行可视化

@author: zhaolei
"""
from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import copy

model = VGG16(weights = 'imagenet', include_top = False)

    
def deprocess_image(x):
    #normalize the tensor : centers on 0, ensures that std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    #clips to [0,1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    #Convert to an RGB array
    x *=255
    x = np.clip(x, 0 , 255).astype('uint8')
    return x

#function to generate filter visualizations
def generate_pattern(layer_name, filter_index, size):
       
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    
    #computes the gradient of the loss with regard to the input
    grads = K.gradients(loss, model.input)[0]
    #gradient-normalization trick
    #add 1e-5 before dividing to avoid accidentally dividing by 0
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5 )
    #returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    #loss_value, grads_value = iterate([np.zeros((1,150,150,3))])
    
    #loss maximization via stochastic gradient descent
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    step = 1.
    
    for i in range(40):
        #print(i)

        loss_value, grads_value = iterate([input_img_data])
        #print(loss_value, grads_value)
        input_img_data += grads_value * step
        
        img = copy.deepcopy(input_img_data[0])
        #打印每一次变化
        #plt.figure()
        #plt.imshow(deprocess_image(img))
        
    img = input_img_data[0]
    
    return deprocess_image(img)
    
#generate_pattern('block3_conv1', 0,150)

plt.imshow(generate_pattern('block3_conv1', 0,150))

#generating a grid of all filter response patterns in a layer
layer_name_list = ['block1_conv1','block1_conv2',
                   'block2_conv1','block2_conv2',
                   'block3_conv1','block3_conv2',
                   'block4_conv1','block4_conv2']
for layer_name in layer_name_list:
    
    #layer_name = 'block1_conv1'
    size = 64
    #size = 150
    margin = 5
    
    results = np.zeros((8*size +7 *margin, 8*size +7*margin,3))
    
    for i in range(8):
        for j in range(8):
            #generates the pattern for filter i+(j*8) in layer_name
            filter_img = generate_pattern(layer_name, i + (j * 8), size = size)
            #put the result in the square (i, j) of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end,:] = filter_img
        print ("i = ",i)
            
        
    #注意图片数据的类型必须是astype('uint8'),否则是会形成空白图片
    plt.figure(figsize=(20,20))
    plt.imshow(results.astype('uint8'))
        
        
    
    



    
