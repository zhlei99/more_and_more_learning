#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:40:04 2018

@author: zhaolei

Text generation with LSTM
"""
import keras
import numpy as np
from keras import layers
import random
import sys

#Download and parsing the initial text file
path = keras.utils.get_file(
        'nietzsche.txt',
        origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:',len(text))

#vectorizing sequences of character
maxlen = 60     #you'll extract sequences of 60 characters
step = 3        #you'll sample a new sequence every three characters
sentences = []   #holds the txtracted sequence
next_chars = [] #holds the targets(the follow-up characters)

for i in range(0,len(text) - maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i + maxlen])

print ('Number of sequences:', len(sentences))

chars = sorted(list(set(text))) #list of unique characters in the corpus
print ('Uniqued characters:, ', len(chars))
#dictionary that maps unique charcaters to their index in the list chars
char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization...')
#one-hot encodes the characters into binary arrays,x.shape = (200278,60,57),60个词，每一个词一行，有57个字符。
x = np.zeros((len(sentences), maxlen, len(chars)),dtype = np.bool)
y = np.zeros((len(sentences), len(chars)),dtype = np.bool)
for i , sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
#Building the network
#single-layer LSTM model for next-character prediction
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

#model compilation configuration
optimizer = keras.optimizers.RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#function to sample the next character given the model's predictions
#text evolues as the model begins to converge
def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) #make probaility distribution = 1
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

#Text-generation loop
for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    #selects a text seed at random
    start_index = random.randint(1, len(text)-maxlen -1)
    generated_text = text[start_index: start_index + maxlen]
    print('____generating with seed:"'+generated_text+'')
    
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------temperature:', temperature)
        sys.stdout.write(generated_text)
        
        #generates 400 characters starting from the seed text
        for i in range(400):
            #one-hot encodes the characters generated so fa
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled [0, t, char_indices[char]] = 1.
            
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            
            generated_text += next_char
            generated_text = generated_text[1:]
            
            sys.stdout.write(next_char)
















