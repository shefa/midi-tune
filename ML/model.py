import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import random
from collections import deque
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, Activation, BatchNormalization

import wandb
from wandb.keras import WandbCallback

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 256,
  layer_1_size = 256,
  layer_2_size = 512,
  layer_3_size = 512,
  learn_rate = 0.01,
  decay = 1e-6,
  epochs = 8,
  sequence_length = 100,
  input_data_type = 2, # delta
)
typemap = ['basic', 'duration', 'delta', 'delta_events']
data_split = ['train', 'validation', 'test']

# Initialize wandb
wandb.init(config=hyperparameter_defaults)
config = wandb.config


def make_sequences(data):
	s_in = []
	s_out = []
	sequences = []
	prev_notes = deque(maxlen=config.sequence_length)

	for notes in data:
		for i in notes:
			if len(prev_notes) == config.sequence_length:
				sequences.append([np.array(prev_notes),i[0]])
			prev_notes.append(i)

	random.shuffle(sequences)

	for i,o in sequences:
		s_in.append(i)
		s_out.append(o*87)

	return np.array(s_in), to_categorical(s_out,num_classes=88)


# load shit
data_parsed = [pickle.load(open(f'rick-{typemap[config.input_data_type]}-{x}','rb')) for x in data_split]
print("Dataset loaded.")

# sequence generation
print("generating train..")
train_x, train_y = make_sequences(data_parsed[0])
print("generating test..")
test_x, test_y = make_sequences(data_parsed[2])
print("generating validation..")
validation_x, validation_y = make_sequences(data_parsed[1])

print("Sequences created")

# create model
model = Sequential()
model.add(LSTM(config.layer_1_size,input_shape=(train_x.shape[1], train_x.shape[2]),return_sequences=True))
model.add(Dropout(config.dropout))
model.add(LSTM(config.layer_2_size, return_sequences=True))
model.add(Dropout(config.dropout))
model.add(LSTM(config.layer_3_size))
model.add(Dropout(config.dropout))
model.add(Dense(config.hidden_layer_size, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(88, activation='softmax'))

opt = keras.optimizers.Adam(lr=config.learn_rate, decay=config.decay)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_x, train_y,  validation_data=(test_x, test_y), epochs=config.epochs,callbacks=[WandbCallback()])

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
atm = str(time.strftime("%H-%M"))
model.save(f"models/{typemap[config.input_data_type]}-{atm}")