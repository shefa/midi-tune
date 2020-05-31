import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import random
from collections import deque
from keras.utils import to_categorical

import wandb
from wandb.keras import WandbCallback

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 128,
  layer_1_size = 16,
  layer_2_size = 32,
  learn_rate = 0.01,
  decay = 1e-6,
  momentum = 0.9,
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
train_x, train_y = make_sequences(data_parsed[0])
test_x, test_y = make_sequences(data_parsed[2])
validation_x, validation_y = make_sequences(data_parsed[1])

print("Sequences created")

# create model
