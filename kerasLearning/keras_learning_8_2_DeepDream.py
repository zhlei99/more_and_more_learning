#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:39:50 2018

@author: zhaolei
"""
from keras.applications import inception_v3
from keras import backend as K
import numpy as np
import scipy
from keras.preprocessing import image


#you won't training the model, so this command disables all training-specific operations
K.set_learning_phase(0)
#builds the Inception V3 network, without its convolutional base.The model will
#be loaded with pretrained ImageNet weights
model = inception_v3.InceptionV3(weights = 'imagenet',
                                 include_top = False)
'''
Dictionary mapping layer names to a coefficient quantifying
how much the layer's activation contributes to the loss 
you'll seek to maximize.Note that the layer names are
hardcoded in the built-in Inception V3 application.You can
list all layer names using model.summary().
'''
#setting up the deepDream configuration
layer_contributions = {
        'mixed2':0.2,
        'mixed3':3.,
        'mixed4':2.,
        'mixed5':1.5,}

#Defining the loss to be maximized
#creates a dictionary that maps layer names to layer instance
layer_dic = dict([(layer.name , layer) for layer in model.layers])
#you'll define the loss by adding layer contributinos to this scalar variable
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dic[layer_name].output
    #Multiplies the values in a tensor, alongside the specified axis.
    scaling = K.prod(K.cast(K.shape(activation),'float32'))
    #add the l2 norm of the features of a layer to the loss .you avoid
    #border artifact by only involving nonborder pixels in the loss
    loss += coeff * K.sum(K.square(activation[:,2:-2,2:-2,:])) / scaling

#gradient-ascent process
#this tensor holds the generated image : the dream
dream = model.input
#computes the gradients of the dream with regard to the loss
grads = K.gradients(loss, dream)[0]
#normalizes the gradients(important trick)    
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
#sets up a keras function to retrieve the value of the loss and gradients,given an input image
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
#this function runs gradient ascent for a number of iterations
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print ('...Loss value at', i, ':',loss_value)
        x += step * grad_values
    return x

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order =1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
    

#Util function to open, resize , and format pictures into tensors that Inception v3 can precess    
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img
#util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1,2,0))
    else:
        #undoes preprocessing that was performed by inception_v3.preprocess_input
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#running gradient ascent over different successive scales
#gradient ascent step size
step = 0.01
#number of scales at which to run gradient ascent
num_octave = 3
#size radio between scales
octave_scale = 1.4
#number of ascent steps to run at each scale
iterations = 20

#if the loss grows larger than 10, you'll interrupt the gradient-ascent process to avoid ugly artifacts
max_loss = 10.
#fill this with the path to the image you want to use
base_image_path = '/Users/zhaolei/Desktop/dataset/bread.png'

#loads the base image into a Numpy array
img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
#prepares a list of shape tuples defining the different scales at which to run gradient ascent
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
        for dim in original_shape])
    successive_shapes.append(shape)
    
#reverses the list of shapes so they're in increasing order
successive_shapes = successive_shapes[::-1]
#resizes the numpy array of the image to the smallest scale
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print ('Processing image shape', shape)
    #scales up the dream image
    img = resize_img(img, shape)
    #runs gradient ascent,altering the dream
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    #scales up the smaller varsion of the original image:it will be pixellated
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    #computes the high-quality version of the original image at this size
    same_size_original = resize_img(original_img, shape)
    #the fifference between the two is the detail that was lost when scaling up
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
    
save_img(img, fname='final_dream.png')


    

  