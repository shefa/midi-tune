from mido import MidiFile
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from collections import deque

# important stuff
data_folder = "saved_data/"
typemap = ['basic', 'basic_velocity', 'delta', 'duration',  'delta_events']
data_split = ['train', 'validation', 'test']
input_data_type=0

if len(sys.argv)>1 and sys.argv[1] in set(typemap):
	input_data_type=typemap.index(sys.argv[1])

files = pd.read_csv('maestro-v2.0.0.csv')[['midi_filename','split']]
data_raw = [list(files[files['split']==x]['midi_filename']) for x in data_split]

lowest_note = 21
highest_note = 87

#notes=[0 for i in range(88)]
#velocities=[0 for i in range(127)]

def extract_notes_basic(x):
	return np.array([j.note-lowest_note for j in x if j.type=='note_on' and j.velocity])

def extract_notes_basic_velocity(x):
	return np.array([ [j.note-lowest_note, j.velocity] for j in x if j.type=='note_on' and j.velocity])

def extract_notes_delta(x):
	y = np.cumsum([j.time for j in x])
	y = [i if i<=300 else int(276+np.sqrt(i)) for i in y ] # retarded data smoothing with 95th quantile onward
	z = [ [j.note-lowest_note,j.velocity,i] for j,i in zip(x,y) if j.type=='note_on' and j.velocity] # notes with their time
	return np.array([z[0]] + [ [z[i-1][0],z[i-1][1],z[i][2]-z[i-1][2] ] for i in range(1,len(z))]) # make deltas
	# 0 for first item because its delta=0

def extract_notes_duration(x):
	z = np.cumsum([j.time for j in x])
	notes = [deque() for i in range(88)]
	y = []
	for j,i in zip(x,z):
		if j.type=='note_on':
			note = j.note-lowest_note
			if j.velocity: # if real note on event
				notes[note].append([len(y),i]) # insert position in queue
				y.append([note,j.velocity])
			else: # if note off event
				try:
					last=notes[note].popleft()
					y[last[0]].append(i-last[1])
				except:
					pass
	for i in notes:
		for j in i:
			del y[j[0]]

	return np.array(y)

def extract_notes_events(x):
	y = np.cumsum([j.time for j in x])
	y = [i if i<=300 else int(276+np.sqrt(i)) for i in y ] # retarded data smoothing with 95th quantile onward
	z = [ [j.note-lowest_note,j.velocity,i] for j,i in zip(x,y) if j.type=='note_on'] # notes with their time
	return np.array([z[0]] + [ [z[i-1][0],z[i-1][1],z[i][2]-z[i-1][2] ] for i in range(1,len(z))]) # make deltas
	# 0 for first item because its delta=0

def extract_notes(x,mode=0):
	if mode==0:
		return extract_notes_basic(x)
	elif mode==1:
		return extract_notes_basic_velocity(x)
	elif mode==2:
		return extract_notes_delta(x)
	elif mode==3:
		return extract_notes_duration(x)
	return extract_notes_events(x)

def toolbar_init(toolbar_width, cnt):
	sys.stdout.write("[%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1))
	return int((cnt+toolbar_width-1)/toolbar_width)

def parse(data):
	cnt = len(data)
	current_progress = 0.05
	thing = []
	
	toolbar_cnt = toolbar_init(60,cnt)
	begin = time.time()
	
	for i in range(cnt):
		x=MidiFile(data[i])
		if not i%toolbar_cnt:
			sys.stdout.write("-")
			sys.stdout.flush()
		thing.append(extract_notes(x.tracks[1],input_data_type))
	sys.stdout.write("] Done in %s sec!\n" % ('{0:.2f}'.format(time.time()-begin))) # this ends the progress barprint("done!")
	return np.array(thing)


print ("Generating dataset for "+typemap[input_data_type])
data_parsed=[]
for i in range(len(data_split)):
	print(data_split[i])
	data_parsed.append(parse(data_raw[i]))
	pickle.dump(data_parsed[i],open(f'../{data_folder}rick-{typemap[input_data_type]}-{data_split[i]}','wb'))


## --------------------- EOF ------------------------

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


# normalize data
