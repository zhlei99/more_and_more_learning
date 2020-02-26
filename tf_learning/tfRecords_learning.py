#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division                   #如果你想在Python 2.7的代码中直接使用Python 3.x的除法，可以通过__future__模块的division实现
                                                #10 // 3 =', 10 // 3
from __future__ import print_function          #让py2.7可以使用py3的打印特性

import argparse
import os.path
import sys
import time

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGES = None

#定义三种格式
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.trian.FloatList(value = [value]))

def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples
    
    if images.shape[0] != num_examples:
        raise ValueError("images size %d does not match label size %d" % 
                         (images.shape[0],num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    
    filename = os.path.join(FLAGES.directory , name + '.tfrecords')
    print ("Writing", filename)
    with tf.python_io.TFRecordWriter(filename) as writer:  #定义writer
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(
                    features =tf.train.Features(
                            
                            feature = {
                                    'height': _int64_feature(rows),
                                    'width': _int64_feature(cols),
                                    'depth': _int64_feature(depth),
                                    'label': _int64_feature(int(labels[index])),
                                    'image_raw': _bytes_feature(image_raw)
                                    }))#定义数据格式
            
            writer.write(example.SerializeToString())
def main (unused_argv):
#get the data
    data_sets = mnist.read_data_sets(FLAGES.directory ,
                                     dtype= tf.uint8 , 
                                     reshape= False ,
                                     validation_size= FLAGES.validation_size)
    #convert to examples and write the ruslut to TFRecords.
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--directory',
            type=str,
            default = '/tmp/data',
            help = 'Directory to download data files and write the converted result'
           
            )
    parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
      )
    FLAGES, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
    
    
    
#得到minis数据
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)