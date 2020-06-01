import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import sys
import random
from collections import deque

typemap = ['basic', 'duration', 'delta', 'delta_events']
data_split = ['train', 'validation', 'test']
input_data_type=0
if len(sys.argv)>1 and sys.argv[1] in set(typemap):
	input_data_type=typemap.index(sys.argv[1])
data_type=typemap[input_data_type]

sequence_folder = "saved_sequences/"
data_folder = "saved_data/"

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
data_parsed = [pickle.load(open(f'{data_folder}rick-{data_type}-{x}','rb')) for x in data_split]
print("Dataset loaded.")

# sequence generation
print("generating train..")
train_x, train_y = make_sequences(data_parsed[0])
pickle.dump(train_x,open(f"{saved_sequences}trainx-{data_type}",'wb'))
pickle.dump(train_y,open(f"{saved_sequences}trainy-{data_type}",'wb'))
print("generating test..")
test_x, test_y = make_sequences(data_parsed[2])
pickle.dump(test_x,open(f"{saved_sequences}testx-{data_type}",'wb'))
pickle.dump(test_y,open(f"{saved_sequences}testy-{data_type}",'wb'))
print("generating validation..")
validation_x, validation_y = make_sequences(data_parsed[1])
pickle.dump(validation_x,open(f"{saved_sequences}validationx-{data_type}",'wb'))
pickle.dump(validation_y,open(f"{saved_sequences}validationy-{data_type}",'wb'))