# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:40:03 2018

@author: Rithwik
"""

import numpy as np
import pandas as pd
 


#data = np.genfromtxt('yelp_labelled.txt',delimiter="	")
 
from keras import optimizers
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


def gen_arr(text,words_dict):
    
    arr = np.zeros((1,len(words_dict)))
    
    for i in text.lower().split():
        arr[0][words_dict[i]] = 1
    return arr        



d={}

lst=['amazon_cells_labelled','yelp_labelled','imdb_labelled']
cnt=0
for files in range(3):    
    
    file = open(lst[files]+'.txt','r')    
    data = file.readlines()    
    for i in data:
        cnt+=1
        [text,label] =  (i.lower().split('\t'))
        for i in text.split():
            d[i]=1
   


stats_array = np.zeros((cnt,len(d.keys())+1))





         


lst=['amazon_cells_labelled','yelp_labelled','imdb_labelled']
keys_dict = {}

words = list(d.keys())

tem_cnt = 0
for i in words:
    keys_dict[i] =  tem_cnt
    tem_cnt+=1


cnt=0
for files in range(3):    
    file = open(lst[files]+'.txt','r')    
    data = file.readlines()    
    for i in data:
        [text,label] =  (i.lower().split('\t'))
        stats_array[cnt][-1] = label
        for txt in text.split():
            #print ("word is ", txt," index is ",keys_dict[txt])
            stats_array[cnt][keys_dict[txt]] = 1
            
        
        cnt+=1
        



len_test = int(.2 * len(stats_array))

len_val = int(.1 * len(stats_array))


np.random.shuffle(stats_array)

X_train = stats_array[:-len_test - len_val,:-1]
y_train = stats_array[:-len_test - len_val,-1]


X_test = stats_array[-len_test - len_val :- len_val,:-1]
y_test = stats_array[-len_test - len_val :- len_val,-1]


X_val = stats_array[-len_val:,:-1]
y_val = stats_array[- len_val:,-1]


model = Sequential()

model.add(Dense(4096,activation='relu',input_dim = len(words)))
model.add(Dropout(.5))
model.add(Dense(2014,activation='relu',))


model.add(Dropout(.25))
model.add(Dense(512,activation='relu',))
model.add(Dropout(.24))
model.add(Dense(16,activation='relu',))

model.add(Dense(1,activation='sigmoid',))


# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



model.fit(X_train,y_train,batch_size=200,epochs = 15,validation_data = (X_val,y_val),verbose=1)



score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])







    
        
