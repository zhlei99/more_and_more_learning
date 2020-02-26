#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:06:48 2018

@author: zhaolei
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:01:13 2018

@author: zhaolei
对keras_learning_5_2_catVsDog.py的模型进行调整
使用dropout，图像变换进行数据集变大。预防过拟合
效果：训练30代可以通过图形看出，acc曲线和val_acc曲线基本吻合的上升，说明30代太少，要
延长训练时间到100代。理论上100代的训练精确度在82%，相比之前的模型提升10%

"""
import os
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image


"""
第一步：对数据进行处理：加载训练集与验证集与测试集
"""
base_dir = '/Users/zhaolei/Desktop/dataset/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats') 

#对图像进行预处理,图片裁剪，用生成器对对象Batch进行训练，高效，内存利用率高。   
#Using ImageDataGenerator to read images from directories
#train_datagen = ImageDataGenerator(rescale=1./255)  #rescales all images by 1/255

#Training the convnet using data-augmentation generators
#创建类对象，并初始化,视图化展示图像的增大。
train_datagen = ImageDataGenerator(        
        rescale=1./255,
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = 'nearest')

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
#加入dropout防止全联接层的过拟合
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#检查模型
model.summary()

#define compilation step and loss function
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc'])

def test_model(model):
    return model.summary()
    

'''
第三步：训练模型
epochs 调整后应该设置成100，或者更长
'''
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
       

#saving the model
#model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_small_1.h5')


'''
第四步：评估模型
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

'''
第五步 调整数据集或者调整模型,解决overfitting，第一步中的图像变换操作已经解决了部分的过拟合，以下
代码主要是图像变换的视觉展示
'''
#setting up a data augmentation configuation via ImageDataGenerator prevent overfitting
#创建类对象，并初始化,视图化展示图像的增大。
datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#displaying some randomly augmented training images
fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]

img_path = fnames[3]

img = image.load_img(img_path, target_size=(150,150))
x= image.img_to_array(img)  #converts it to a numpy array with shape(150,150,3)

x = x.reshape((1,) + x.shape) #reshape it to (1,150,150,3)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0 :
        break
plt.show()

'''
通过之前的图片增大方法，dropout 方法，最后调整好模型，再次训练，而后保存模型
'''
model.save('/Users/zhaolei/Library/Mobile Documents/com~apple~CloudDocs/zhleicode/MLStudy/cats_and_dogs_small_2.h5')
#读取模型
#model = load_model()

'''
其中在第五步中：通过已有的模型进行学习，特征提取的两个方法.第一种方法，用以训练好的模型进行图像的特征提取
然后建立新的全连接层进行分类。代价较小。第二种方法，通过在训练好的模型的conv_base,和全连接层
一起重新做训练，这时需要把小数据集进行增大处理，但此方法代价较大。此后是部分代码，详细代码在
keras_learning_5_image_feature_extracting.py 与
keras_learning_5_image_feature_extracting2.py 

'''    
