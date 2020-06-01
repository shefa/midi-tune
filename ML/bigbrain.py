import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import sys
import random
from collections import deque
from keras.utils import to_categorical

def toolbar_init(toolbar_width, cnt):
	sys.stdout.write("[%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1))
	return int((cnt+toolbar_width-1)/toolbar_width)

def toolbar_tick(smth):
	if not smth:
		sys.stdout.write("-")
		sys.stdout.flush()

def make_sequences_basic(data,sequence_length):
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
				sequences.append([np.array(prev_notes),i])
			prev_notes.append(i)

	random.shuffle(sequences)

	for i,o in sequences:
		s_in.append(i)
		s_out.append(o)

	return to_categorical(s_in,num_classes=88, dtype=np.bool), to_categorical(s_out,num_classes=88, dtype=np.bool)


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