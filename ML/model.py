import numpy as np
import pandas as pd
import pickle
import time
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, LSTM, Activation, BatchNormalization
import tensorflowjs #The scripts tensorflowjs_converter and tensorflowjs_wizard are installed in '/home/shefa/.local/bin'

import wandb
from wandb.keras import WandbCallback

from bigbrain import make_sequences_basic as make_sequences

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 64,
  layer_1_size = 128,
  layer_2_size = 128,
  layer_3_size = 128,
  learn_rate = 0.01,
  decay = 1e-6,
  epochs = 8,
  sequence_length = 100,
  input_data_type = 0,
)

data_folder = "saved_data/"
sequence_folder = "saved_sequences/"
typemap = ['basic', 'basic_velocity', 'delta', 'duration',  'delta_events']
data_split = ['train', 'validation', 'test']

# Initialize wandb
wandb.init(config=hyperparameter_defaults)
config = wandb.config
data_type=typemap[config.input_data_type]

#load data
def load_data():
	train_x = 		pickle.load(open(f"{sequence_folder}trainx-{data_type}",		'rb'))
	train_y = 		pickle.load(open(f"{sequence_folder}trainy-{data_type}",		'rb'))
	test_x =  		pickle.load(open(f"{sequence_folder}testx-{data_type}",			'rb'))
	test_y =  		pickle.load(open(f"{sequence_folder}testy-{data_type}",			'rb'))
	validation_x =  pickle.load(open(f"{sequence_folder}validationx-{data_type}",	'rb'))
	validation_y =  pickle.load(open(f"{sequence_folder}validationy-{data_type}",	'rb'))
	return train_x, train_y, test_x, test_y, validation_x, validation_y

def create_data():
	print("Loading dataset..")
	data_parsed = [pickle.load(open(f'{data_folder}rick-{data_type}-{x}','rb')) for x in data_split]
	print("generating train..")
	train_x, train_y = make_sequences(data_parsed[0], config.sequence_length)
	print("generating test..")
	test_x, test_y = make_sequences(data_parsed[2], config.sequence_length)
	print("generating validation..")
	validation_x, validation_y = make_sequences(data_parsed[1], config.sequence_length)
	return train_x, train_y, test_x, test_y, validation_x, validation_y

def save_data():
	print("saving train..")
	pickle.dump(train_x,open(f"{sequence_folder}trainx-{data_type}",'wb'))
	pickle.dump(train_y,open(f"{sequence_folder}trainy-{data_type}",'wb'))
	print("saving test..")
	pickle.dump(test_x,open(f"{sequence_folder}testx-{data_type}",'wb'))
	pickle.dump(test_y,open(f"{sequence_folder}testy-{data_type}",'wb'))
	print("saving validation..")
	pickle.dump(validation_x,open(f"{sequence_folder}validationx-{data_type}",'wb'))
	pickle.dump(validation_y,open(f"{sequence_folder}validationy-{data_type}",'wb'))

train_x, train_y, test_x, test_y, validation_x, validation_y = create_data()
print("Sequences loaded")

# create model
model = Sequential()
model.add(LSTM(config.layer_1_size,input_shape=(train_x.shape[1], train_x.shape[2]),return_sequences=True, recurrent_dropout=config.dropout))
#model.add(LSTM(config.layer_2_size, return_sequences=True, recurrent_dropout=config.dropout))
model.add(LSTM(config.layer_3_size))
model.add(Dropout(config.dropout))
model.add(Dense(config.hidden_layer_size, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(88, activation='softmax'))

opt = Adam(lr=config.learn_rate, decay=config.decay)
atm = str(time.strftime("%H-%M"))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
try:
	model.fit(train_x, train_y,  validation_data=(test_x, test_y), epochs=config.epochs,callbacks=[WandbCallback()])
except KeyboardInterrupt:
	score = model.evaluate(validation_x, validation_y, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
finally:
	# Save model
	model.save(f"models/{data_type}-{atm}")
	tfjs.converters.save_keras_model(model, f"models/js/{data_type}-{atm}")