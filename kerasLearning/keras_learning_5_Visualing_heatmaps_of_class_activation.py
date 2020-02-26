#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:43:37 2018

@author: zhaolei

CAM(class activation map)
读懂这些代码需要读懂文章
paper：Grad-CAM:
Visual Explanations from Deep Networks via Gradient-based Localization

This is
helpful for debugging the decision process of a convnet, particularly in the case of a
classification mistake. It also allows you to locate specific objects in an image.
"""
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

img_path = '/Users/auser/Desktop/dataset/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
#transform imge to string
x = image.img_to_array(img)
#add a dimension to transform the array into a batch of size(1,224,224,3)
x = np.expand_dims(x, axis = 0) 
#preprocesses the batch (this does channel-wise color normalization)
x = preprocess_input(x)


model = VGG16(weights = 'imagenet')
#preds = model.predict(x)
#np.argmax(preds[0])
#print (decode_predictions(preds, top=3)[0])

#setting up the grad-cam algorithm
african_elephant_output = model.output[:, 386]
#output feature map of the block5_conv3 layer, the last convolutional layer in VGG
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis = (0, 1, 2))
#实例化一个keras函数,将4为tensor变成一个3维tensor
iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])
#multiplies each channel in the feature-map array by "how important this channel is"
#with reagard to the "elephant" class
for i in range(512):
    #计算每个通道的平均梯度与输出的积
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#the channel-wise mean of the resulting feature map is the heatmap of the calss activation
heatmap = np.mean(conv_layer_output_value, axis = -1)

#for visualization purposes , you will also mormalize the heatmap between
#0 and 1.
#Heatmap post-processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

#Superimposing the heatmap with the original picture by OpenCV
#uses cv2 to load the original image
img = cv2.imread(img_path)
#resizes the heatmap to be zhe same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1] , img.shape[0]))
#converts the hatmap to RGB
heatmap = np.uint8(255 * heatmap)
#Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#o.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img
#saves the image to disk
cv2.imwrite('/Users/auser/Desktop/dataset/elephant_cam.jpg', superimposed_img)

decode_predictions()

