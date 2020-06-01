from mido import MidiFile
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from collections import deque
from bigbrain import toolbar_init, toolbar_tick

# important stuff
data_folder = "saved_data/"
typemap = ['basic', 'basic_velocity', 'delta', 'duration',  'delta_events']
data_split = ['train', 'validation', 'test']
input_data_type=0

if len(sys.argv)>1 and sys.argv[1] in set(typemap):
	input_data_type=typemap.index(sys.argv[1])

files = pd.read_csv('maestro/maestro-v2.0.0.csv')[['midi_filename','split']]
data_raw = [list(files[files['split']==x]['midi_filename']) for x in data_split]

lowest_note = 21
highest_note = 87

#notes=[0 for i in range(88)]
#velocities=[0 for i in range(127)]

def extract_notes_basic(x):
	return np.array([j.note-lowest_note for j in x if j.type=='note_on' and j.velocity], dtype=np.uint8)

def extract_notes_basic_velocity(x):
	return np.array([ [j.note-lowest_note, j.velocity] for j in x if j.type=='note_on' and j.velocity], dtype=np.uint8)

def extract_notes_delta(x):
	y = np.cumsum([j.time for j in x])
	y = [i if i<200 else int(200+np.sqrt(i)) for i in y ] # retarded data smoothing with .9999th quantile onward
	z = [ [j.note-lowest_note,j.velocity,i] for j,i in zip(x,y) if j.type=='note_on' and j.velocity] # notes with their time
	return np.array([z[0]] + [ [z[i-1][0],z[i-1][1],z[i][2]-z[i-1][2] ] for i in range(1,len(z))], dtype=np.uint8) # make deltas
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

	return np.array(y, dtype=np.uint16)

def extract_notes_events(x):
	y = np.cumsum([j.time for j in x])
	y = [i if i<200 else int(200+np.sqrt(i)) for i in y ] # retarded data smoothing with 95th quantile onward
	z = [ [j.note-lowest_note,j.velocity,i] for j,i in zip(x,y) if j.type=='note_on'] # notes with their time
	return np.array([z[0]] + [ [z[i-1][0],z[i-1][1],z[i][2]-z[i-1][2] ] for i in range(1,len(z))], dtype=np.uint8) # make deltas
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

def parse(data):
	cnt = len(data)
	current_progress = 0.05
	thing = []
	
	toolbar_cnt = toolbar_init(60,cnt)
	begin = time.time()
	
	for i in range(cnt):
		name = f'maestro/{data[i]}'
		x=MidiFile(name)
		toolbar_tick(i%toolbar_cnt)
		thing.append(extract_notes(x.tracks[1],input_data_type))
	sys.stdout.write("] Done in %s sec!\n" % ('{0:.2f}'.format(time.time()-begin))) # this ends the progress barprint("done!")
	return np.array(thing)


print ("Generating dataset for "+typemap[input_data_type])
data_parsed=[]
for i in range(len(data_split)):
	print(data_split[i])
	data_parsed.append(parse(data_raw[i]))
	pickle.dump(data_parsed[i],open(f'{data_folder}rick-{typemap[input_data_type]}-{data_split[i]}','wb'))


## --------------------- EOF ------------------------