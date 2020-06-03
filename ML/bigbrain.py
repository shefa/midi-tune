import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import sys
import random
from collections import deque
from keras.utils import to_categorical
from keras.callbacks import Callback
import wandb

def toolbar_init(toolbar_width, cnt):
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    return int((cnt+toolbar_width-1)/toolbar_width)

def toolbar_tick(smth):
    if not smth:
        sys.stdout.write("-")
        sys.stdout.flush()

def batch_generator(Train_df, batch_size,steps):
    idx=1
    while True: 
        yield load_data(Train_df,idx-1,batch_size)## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1

class CustomCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        wandb.log({'accuracy': logs['accuracy'], 'loss': logs['loss']})
    
def make_sequences_basic_one_hot(data,sequence_length):
    s_in = []
    s_out = []
    sequences = []
    prev_notes = deque(maxlen=sequence_length)
    cnt, sz = 0, len(data)
    for notes in data:
        prev_notes.clear()
        print(f'{cnt}/{sz}')
        cnt+=1
        for i in notes:
            if len(prev_notes) == sequence_length:
                sequences.append([np.array(prev_notes),i%12])
            prev_notes.append(i)

    random.shuffle(sequences)

    for i,o in sequences:
        s_in.append(i)
        s_out.append(o)
    
    s_in = np.array(s_in, dtype=np.int8)
    s_out = np.array(s_out, dtype=np.int8)
    s_out = np.reshape(s_out, (len(s_out), 1))

    return to_categorical(s_in, num_classes=88, dtype=np.bool), s_out

def make_sequences_basic_one_half(data,sequence_length):
    s_in = []
    s_out = []
    sequences = []
    prev_notes = deque(maxlen=sequence_length)
    cnt, sz = 0, len(data)
    for notes in data:
        prev_notes.clear()
        print(f'{cnt}/{sz}')
        cnt+=1
        for i in notes:
            if len(prev_notes) == sequence_length:
                sequences.append([np.array(prev_notes),i%12])
            prev_notes.append(i)

    random.shuffle(sequences)

    for i,o in sequences:
        s_in.append(i)
        s_out.append(o)
    
    #s_in = (np.array(s_in)/87.)
    s_in = np.array(s_in, dtype=np.int8)
    s_out = np.array(s_out, dtype=np.int8)
    n_patterns = len(s_in)
    s_in = np.reshape(s_in, (n_patterns, sequence_length, 1))
    s_out = np.reshape(s_out, (n_patterns, 1))

    return s_in, s_out#to_categorical(s_in, num_classes=88, dtype=np.bool), s_out

def make_sequences_basic(data,sequence_length):
    return make_sequences_basic_one_hot(data,sequence_length)
    s_in = []
    s_out = []
    sequences = []
    prev_notes = deque(maxlen=sequence_length)
    cnt, sz = 0, len(data)
    for notes in data:
        print(f'{cnt}/{sz}')
        cnt+=1
        for i in notes:
            if len(prev_notes) == sequence_length:
                sequences.append([np.array(prev_notes),i%12])
            prev_notes.append(i)

    random.shuffle(sequences)

    for i,o in sequences:
        s_in.append(i)
        s_out.append(o)

    #s_in = ((np.array(s_in)/87.)*2.)-1
    s_in = np.array(s_in, dtype=np.int8)
    s_out = np.array(s_out, dtype=np.int8)
    n_patterns = len(s_in)
    #s_in = np.reshape(s_in, (n_patterns, sequence_length, 1))
    s_out = np.reshape(s_out, (n_patterns, 1))

    return s_in, s_out #to_categorical(s_out,num_classes=12, dtype=np.bool)

def make_sequences_duration(data,sequence_length):
    s_in = []
    s_out = []
    sequences = []
    prev_notes = deque(maxlen=sequence_length)
    cnt, sz = 0, len(data)
    
    maxduration = np.max([i[2] for sub in data for i in sub])
    for notes in data:
        print(f'{cnt}/{sz}')
        cnt+=1
        for i in notes:
            if len(prev_notes) == sequence_length:
                sequences.append([np.array(prev_notes),i[0]%12])
            prev_notes.append([
                ((i[0]/87.0)*2)-1, 
                ((i[1]/127.0)*2)-1,
                ((i[2]/maxduration)*2.)-1])

    random.shuffle(sequences)

    for i,o in sequences:
        s_in.append(i)
        s_out.append(o)

    s_in = np.array(s_in)
    s_out = np.array(s_out, dtype=np.uint8)
    #n_patterns = len(s_in)
    #s_in = np.reshape(s_in, (n_patterns, sequence_length, 1))
    #s_out = np.reshape(s_out, (n_patterns, 1))

    return s_in, to_categorical(s_out,num_classes=12, dtype=np.bool)

def make_sequences_delta(data,sequence_length):
    s_in = []
    s_out = []
    sequences = []
    prev_notes = deque(maxlen=sequence_length)
    cnt, sz = 0, len(data)
    for notes in data:
        print(f'{cnt}/{sz}')
        cnt+=1
        for i in notes:
            if len(prev_notes) == sequence_length:
                sequences.append([np.array(prev_notes),i[0]%12])
            prev_notes.append(i)
            #    [((i[0]/87.0)*2)-1, 
            #    ((i[1]/127.0)*2)-1,
            #    ((i[2]/255.)*2.)-1])

    random.shuffle(sequences)

    for i,o in sequences:
        s_in.append(np.concatenate([
            to_categorical(i[:,0], num_classes=88, dtype=np.bool),
            to_categorical(i[:,1], num_classes=128, dtype=np.bool),
            to_categorical(i[:,2], num_classes=256, dtype=np.bool)
        ], axis=1))
        #s_in.append(i)
        s_out.append(o)

    s_in = np.array(s_in, dtype=np.bool)
    s_out = np.array(s_out, dtype=np.uint8)
    #n_patterns = len(s_in)
    #s_in = np.reshape(s_in, (n_patterns, sequence_length, 1))
    #s_out = np.reshape(s_out, (n_patterns, 1))

    return s_in, to_categorical(s_out,num_classes=12, dtype=np.bool)

def loss_choice(type):
    return 'sparse_categorical_crossentropy'
    if type==0 or type==2 or type==3:
        return 'categorical_crossentropy'
    else:
        return 'sparse_categorical_crossentropy'

def make_sequences_choice(data,sequence_length, type):
    if type==0:
        return make_sequences_basic(data,sequence_length)
    if type==1:
        return make_sequences_basic(data,sequence_length)
    if type==2:
        return make_sequences_delta(data,sequence_length)
    if type==3:
        return make_sequences_duration(data,sequence_length)

def vibe_check(d):
    print(max(d), min(d))
    print(np.mean(d), np.median(d))
    print(np.quantile(d,.25), np.quantile(d,.75), np.quantile(d,.95))
    m_value=max(d)#1000
    plt.hist(d,range(m_value), density=True)
    plt.gca().set(title='Frequency histogram for Delta time values', ylabel='Frequency')
    plt.xlim(0,m_value)
    plt.legend()
    plt.show()

def ticks_per_beat_test():
    switches=[132, 914, 1183]
    four_eighty = [ [x[2] for y in thing[:switches[0]] for x in y], [x[2] for y in thing[switches[1]:switches[2]] for x in y] ]
    three_eighty = [ [x[2] for y in thing[switches[0]:switches[1]] for x in y], [x[2] for y in thing[switches[2]:] for x in y] ]

    four=four_eighty[0]+four_eighty[1]
    three=three_eighty[0]+three_eighty[1]

    print(np.mean(four), np.mean(three))
    print(np.median(four), np.median(three))

    m_value=1000
    m_value=max(four+three)
    plt.hist(three,range(m_value), alpha=0.5, label='384 ticks', density=True)
    plt.hist(four,range(m_value),  alpha=0.5, label='480 ticks', density=True)
    plt.gca().set(title='Frequency histogram for Delta time with different ticks_per_beat', ylabel='Frequency')
    plt.xlim(0,500)
    plt.legend()
    plt.show()