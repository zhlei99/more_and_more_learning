#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:52:49 2018

Neural style transfer in keras
Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, “A Neural Algorithm of Artistic Style,” arXiv (2015),
https://arxiv.org/abs/1508.06576.

@author: zhaolei
"""
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

#defining initial variables
target_image_path = '/Users/zhaolei/Desktop/dataset/yy_content.jpg'
style_reference_image_path = '/Users/zhaolei/Desktop/dataset/transfer_style_reference2.jpg'

#按照图片比例大小进行缩放，变换到固定尺寸
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

#auxiliary function :preprocessing and postprocessing the images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    #caffe: will convert the images from RGB to BGR,
    #then will zero-center each color channel with
    #respect to the ImageNet dataset,
    #without scaling.
    img = vgg19.preprocess_input(img)
    return img

#zero-centreing by removing the mean pixel value from imageNet. This reverses a 
#transformation done by vgg19.preprocess_input
def deprocess_image(x):
    #zero-centering by removing the mean pixel value from ImageNet
    #this is also part of the reversal of vgg19.preprocess_input
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    #converts image from 'BGR' to "RGB".
    #this is also part of the reversal of vgg19.preprocess_input
    #逆序输出
    x = x[:, :, ::-1]
    #最大最小限制，取整，转换类型,一定得用uint8类型，否则不能展示
    x = np.clip(x, 0, 255).astype('uint8')
    return x

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
#placeholder that will contain the generated image
combination_image = K.placeholder((1, img_height, img_width, 3))

#combines the three image in a single batch
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)
    
#builds the VGG19 network with the batch of three images as input.
#the model will be loaded with pretrained ImageNet weights.
model = vgg19.VGG19(input_tensor = input_tensor,
                    weights = 'imagenet',
                    include_top = False)
print ('Model loaded.')

#content loss，use only one upper layer-the block5_conv2 layer
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

#style loss , use a list of layers than spans both low-level and high-level layers
def gram_matrix(x):
    #将（400，299，3）变成（3，400，299），而后，变成（3，400*299）变成长向量操作
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    #自身点积操作,自身点积操作就是第i个特征图与第j个特征图的乘积矩阵
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channeles = 3
    size = img_height * img_width
    return K.sum(K.square(S - C ))/(4. * (channeles ** 2) * (size ** 2))

#Total variation loss
def total_variation_loss(x):
    a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] -
            x[:, 1:, :img_width - 1, :])
    b = K.square(
            x[:, :img_height - 1, :img_width -1,:] -
            x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#Defining the final loss that you'll miminmse
#Dictionary that maps layer names to activation tensors
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
#layer used for content loss
content_layer = 'block5_conv2'
#layers used for style loss
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
#weights in the weighted average of the loss components
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

#you'll define the loss by adding all components to this scalar variable
#adds the content loss
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
#第'block5_conv2'层的特征图
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

#adds a style loss component for each target layer
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    s1 = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * s1

#adds the total variation loss
loss += total_variation_weight * total_variation_loss(combination_image)

#setting up the gradient-descent process
#gets the gradients of the generated image with regard to the loss
grads = K.gradients(loss, combination_image)[0]
#function to fetch the values of the  current loss and the current gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

#this class fetch_loss_and_grads in a way that lets you retrieve the losses and
#gradients via two separate method calls, which is required by the SciPy optimizer you'll use
class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
evaluator = Evaluator()
        
#style-transfer loop
result_prefix = 'my_result'
iterations = 20

#this is the initial state:the target image
x = preprocess_image(target_image_path)
#flatten the image because scipy.optimize.fmin_1_bfgs_b can only process flat vectors
x = x.flatten()
for i in range(iterations):
    print('Start of iteration',i)
    start_time = time.time()
    #runs L-BFGS optimization over the pixels of the generated image to minimize
    #the neural style loss.Note that you have to pass the fuction that computes
    #the loss and the function that cumputes the gradients as two separate arguments
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun = 20)
    print('Current loss value:', min_val)
    #save the current generated image,reshape在原列表中进行变换
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    

    

