import numpy as np
import pandas as pd
import pickle
import time
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, LSTM, Activation, BatchNormalization
import tensorflowjs as tfjs #The scripts tensorflowjs_converter and tensorflowjs_wizard are installed in '/home/shefa/.local/bin'

import wandb
from wandb.keras import WandbCallback

#from bigbrain import make_sequences_choice as make_sequences # 

# bigbrain was replaced by smallbrain
from smallbrain import brain
from bigbrain import CustomCallback, loss_choice

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 128,
  layer_1_size = 128,
  layer_2_size = 128,
  layer_3_size = 128,
  learn_rate = 0.01,
  decay = 1e-6,
  epochs = 8,
  batch_size = 1024,
  sequence_length = 200,
  input_data_type = 0,
  input_data_size = 50, # idk
)

data_folder = "saved_data/"
typemap = ['basic', 'basic_velocity', 'delta', 'duration',  'delta_events']
data_split = ['train', 'validation', 'test']

# Initialize wandb
wandb.init(config=hyperparameter_defaults)
config = wandb.config
data_type=typemap[config.input_data_type]

featuers = brain(config.input_data_type, config.sequence_length, config.batch_size)

# create model
model = Sequential()
model.add(LSTM(config.layer_1_size,input_shape=(config.sequence_length, features),return_sequences=True, recurrent_dropout=config.dropout))
#model.add(LSTM(config.layer_2_size, return_sequences=True, recurrent_dropout=config.dropout))
model.add(LSTM(config.layer_3_size, recurrent_dropout=config.dropout))
#model.add(Dropout(config.dropout))
model.add(Dense(config.hidden_layer_size, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(12, activation='softmax'))

opt = Adam(lr=config.learn_rate, decay=config.decay)
atm = str(time.strftime("%H-%M"))
model.compile(loss=loss_choice(config.input_data_type), optimizer=opt, metrics=['accuracy'])
try:
    model.fit(train_x, train_y,  validation_data=(test_x, test_y), epochs=config.epochs, batch_size=config.batch_size, callbacks=[WandbCallback(), CustomCallback()])
except KeyboardInterrupt:
    pass
finally:
    # Save model
    print("-------------- Saving model ---------- ")
    model.save(f"models/{data_type}-{atm}")
    print("-------------- Saving js    ---------- ")
    tfjs.converters.save_keras_model(model, f"models/js/{data_type}-{atm}")