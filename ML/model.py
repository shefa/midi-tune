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
import tensorflowjs #The scripts tensorflowjs_converter and tensorflowjs_wizard are installed in '/home/shefa/.local/bin'

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
generated = False

# Initialize wandb
wandb.init(config=hyperparameter_defaults)
config = wandb.config


def make_sequences(data):
	s_in = []
	s_out = []
	sequences = []
	prev_notes = deque(maxlen=config.sequence_length)
	cnt, sz = 0, len(data)
	for notes in data:
		print(f'{cnt}/{sz}')
		cnt+=1
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
if not generated:
	print("generating train..")
	train_x, train_y = make_sequences(data_parsed[0])
	pickle.dump(train_x,open(f"trainx-{typemap[config.input_data_type]}",'wb'))
	pickle.dump(train_y,open(f"trainy-{typemap[config.input_data_type]}",'wb'))
	print("generating test..")
	test_x, test_y = make_sequences(data_parsed[2])
	pickle.dump(test_x,open(f"testx-{typemap[config.input_data_type]}",'wb'))
	pickle.dump(test_y,open(f"testy-{typemap[config.input_data_type]}",'wb'))
	print("generating validation..")
	validation_x, validation_y = make_sequences(data_parsed[1])
	pickle.dump(validation_x,open(f"validationx-{typemap[config.input_data_type]}",'wb'))
	pickle.dump(validation_y,open(f"validationy-{typemap[config.input_data_type]}",'wb'))
else:
	train_x=pickle.load(open(f"trainx-{typemap[config.input_data_type]}",'rb'))
	train_y=pickle.load(open(f"trainy-{typemap[config.input_data_type]}",'rb'))
	test_x=pickle.load(open(f"testx-{typemap[config.input_data_type]}",'rb'))
	test_y=pickle.load(open(f"testy-{typemap[config.input_data_type]}",'rb'))
	validation_x=pickle.load(open(f"validationx-{typemap[config.input_data_type]}",'rb'))
	validation_y=pickle.load(open(f"validationy-{typemap[config.input_data_type]}",'rb'))

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
atm = str(time.strftime("%H-%M"))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
try:
	model.fit(train_x, train_y,  validation_data=(test_x, test_y), epochs=config.epochs,callbacks=[WandbCallback()])
except KeyboardInterrupt:
	score = model.evaluate(validation_x, validation_y, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
finally:
	# Save model
	model.save(f"models/{typemap[config.input_data_type]}-{atm}")
	tfjs.converters.save_keras_model(model, f"models/js/{typemap[config.input_data_type]}-{atm}")