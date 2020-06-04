import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import sys
import random
from collections import deque
from keras.utils import to_categorical, Sequence
from keras.callbacks import Callback
import wandb
from bigbrain import toolbar_init, toolbar_tick

sequence_folder = "saved_sequences/"
typemap = ['basic', 'basic_velocity', 'delta', 'duration',  'delta_events']
data_split = ['train', 'validation', 'test']

def make_little(data, sequence_length):
    s_in, s_out = [], []
    prev_notes = deque(maxlen=sequence_length)
    for i in data:
        if len(prev_notes) == sequence_length:
            s_in.append(np.array(prev_notes))
            s_out.append(i%12)
        prev_notes.append(i)
    return to_categorical(s_in,num_classes=88,dtype=np.bool), s_out

def make_delta_sequences(data,sequence_length):
    # Get a midi sequence in, return sequences of desired length with their target
    # s_in => sequences of categorically encoded data, s_out => target note
    s_in, s_out = [], []
    prev_notes = deque(maxlen=sequence_length)
    for i in data:
        if len(prev_notes) == sequence_length and i[1]:
            s_in.append(np.array(prev_notes,dtype=np.bool))
            s_out.append(i[0]%12)
        prev_notes.append(np.concatenate(
            [
                to_categorical(i[0],num_classes=88,dtype=np.bool), 
                to_categorical(i[1],num_classes= 8,dtype=np.bool), 
                to_categorical(i[2],num_classes=12,dtype=np.bool)
            ]
            , axis=0))

    s_in = np.array(s_in,dtype=np.bool)
    s_out = np.reshape(np.array(s_out,dtype=np.int8), (len(s_in), 1))

    # shuffle
    hack = np.arange(s_in.shape[0])
    np.random.shuffle(hack)
    s_in = s_in[hack]
    s_out = s_out[hack]
    return s_in, s_out

def save_data(s_in, s_out, index=0, suffix='train'):
    #for i in range(batch_size,len(s_in),batch_size):
    #    np.save(f'{sequence_folder}seq-input-{suffix}-{start_index}.npy' , s_in[last_index:i])
    #    np.save(f'{sequence_folder}seq-output-{suffix}-{start_index}.npy', s_out[last_index:i])
    #    last_index = i
    #    start_index+=1

    np.save(f'{sequence_folder}seq-input-{suffix}-{index}.npy' , s_in)
    np.save(f'{sequence_folder}seq-output-{suffix}-{index}.npy', s_out)

def data_to_sequences(data, sequence_length, suffix, batch_size, start_index=0):
    s_in = np.empty((0,sequence_length,88+8+12),dtype=np.bool)
    s_out = np.empty((0,1),dtype=np.int8)
    index = 0

    # toolbar stuff
    cnt = len(data)
    c2=toolbar_init(60,cnt)

    for i in range(cnt):

        toolbar_tick(i%c2) # toolbar again

        t_in, t_out = make_delta_sequences(data[i],sequence_length)
        s_in  = np.concatenate([ s_in,  t_in],axis=0)
        s_out = np.concatenate([s_out, t_out],axis=0)

        while len(s_in) >= batch_size:
            save_data(s_in[:batch_size], s_out[:batch_size], index, suffix)
            s_in = s_in[batch_size:]
            s_out = s_out[batch_size:]
            index+=1

    return index, s_in, s_out # the leftout stuff

def brain(data_type, sequence_length, batch_size=1024):
    print("Loading dataset..")
    data_parsed = [pickle.load(open(f'saved_data/rick-{typemap[data_type]}-{x}','rb')) for x in data_split]
    print(len(data_parsed[0]),len(data_parsed[1]), len(data_parsed[2]))
    print("generating train..")
    train_cnt, train_x, train_y = data_to_sequences(data_parsed[0][:100], sequence_length, 'train', batch_size)
    print(f'{train_cnt} saved, {len(train_x)} sequences leftover')
    print("generating test..")
    test_cnt, test_x, test_y = data_to_sequences(data_parsed[1][:100], sequence_length, 'test', batch_size)
    print(f'{test_cnt} saved, {len(test_x)} sequences leftover')
    #print("generating validation..")
    #validation_cnt, validation_x, validation_y = data_to_sequences(data_parsed[2], sequence_length, 'validation', batch_size)
   # print(f'{validation_cnt} saved, {len(validation_x)} sequences leftover')
    return train_x.shape[2], train_cnt, test_cnt



class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, suffix, data_type, sequence_length, cnt, batch_size=1024):
        'Initialization'
        self.batch_size = batch_size
        self.suffix = suffix
        self.cnt = cnt
        self.sequence_length = sequence_length

        
        print("Loading dataset..")
        self.data = pickle.load(open(f'saved_data/rick-{typemap[data_type]}-{suffix}','rb'))
        print(len(self.data))

        self.s_in = np.empty((0,self.sequence_length,88+8+12),dtype=np.bool)
        self.s_out = np.empty((0,1),dtype=np.int8)
        self.data_index = 0


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.cnt

    def __generate(self):
        t_in, t_out = make_delta_sequences(self.data[self.data_index],self.sequence_length)
        self.s_in  = np.concatenate([ self.s_in,  t_in],axis=0)
        self.s_out = np.concatenate([self.s_out, t_out],axis=0)
        self.data_index+=1

    def __serve_from_generated(self):
        t_in, t_out = self.s_in[:self.batch_size], self.s_out[:self.batch_size]
        self.s_in  =  self.s_in[self.batch_size:]
        self.s_out = self.s_out[self.batch_size:]
        return t_in, t_out

    def __getitem__(self, index):
        'Generate one batch of data'
        while len(self.s_in) < self.batch_size:
            self.__generate()
        return self.__serve_from_generated()

        #return  np.load(f'saved_sequences/seq-input-{self.suffix}-{index}.npy'), np.load(f'saved_sequences/seq-output-{self.suffix}-{index}.npy')