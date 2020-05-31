from mido import MidiFile
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# important stuff
lowest_note = 21
highest_note = 87

typemap = ['basic', 'duration', 'delta', 'delta_events']
input_data_type=0

if len(sys.argv)>1 and sys.argv[1] in set(typemap):
	input_data_type=typemap.index(sys.argv[1])

files = pd.read_csv('maestro-v2.0.0.csv')[['midi_filename','split']]
data_split = ['train', 'validation', 'test']
data_raw = [list(files[files['split']==x]['midi_filename']) for x in data_split]


#notes=[0 for i in range(88)]
#velocities=[0 for i in range(127)]

def extract_notes_duration(x):
	abs_time=0
	y=[]
	notes=[0 for i in range(88)]
	for j in x:
		abs_time+=j.time
		if j.type=='note_on':
			note = (j.note-lowest_note)/highest_note
			if j.velocity==0:
				y[notes[note]].append(abs_time)
				notes[note]=0
			else:
				notes[note]=j.note-lowest_note
				y.append([note,j.velocity/126])
	return y

def extract_notes_delta(x):
	abs_time=0
	y=[]
	for j in x:
		abs_time+=j.time
		if j.type=='note_on' and j.velocity!=0:
			if abs_time > 300:  # retarded data smoothing
				abs_time=int(276+np.sqrt(abs_time)) # 95th quantile onward
			note = (j.note-lowest_note)/highest_note
			y.append([note,j.velocity/126,abs_time/480])
			abs_time=0
	return y


def extract_notes_events(x):
	abs_time=0
	y=[]
	for j in x:
		abs_time+=j.time
		if j.type=='note_on':
			j.note-=lowest_note
			if abs_time > 300:  # retarded data smoothing
				abs_time=int(299+np.sqrt(abs_time))
			y.append([j.note/highest_note,j.velocity/126,abs_time/480])
			abs_time=0
	y[0][2]=0
	return y


def extract_notes(x,mode=0):
	if mode==1:
		return extract_notes_duration(x)
	elif mode==2:
		return extract_notes_delta(x)
	elif mode==3:
		return extract_notes_events(x)

	#default = basic with no timing, only note on events
	y=[]
	for j in x:
		if j.type=='note_on' and j.velocity!=0:
			y.append([(j.note-lowest_note)/highest_note,j.velocity/126])
	return y

def parse(data):
	cnt = len(data)
	current_progress = 0.05
	thing = []

	toolbar_width = 60
	toolbar_cnt = int((cnt+toolbar_width-1) / toolbar_width)
	sys.stdout.write("[%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1))
	begin = time.time()
	
	for i in range(cnt):
		x=MidiFile(data[i])
		if i%toolbar_cnt==0:
			sys.stdout.write("-")
			sys.stdout.flush()
		thing.append(extract_notes(x.tracks[1],input_data_type))
	sys.stdout.write("] Done in %s sec!\n" % ('{0:.2f}'.format(time.time()-begin))) # this ends the progress barprint("done!")
	return thing

def make_sequences(data):
	s_in = []
	s_out = []
	prev_notes = deque(maxlen=hyperparams.sequence_length)

	for notes in data:
		for i in notes:
			if len(prev_notes) == hyperparams.sequence_length:
				s_in.append(np.array(prev_notes))
				s_out.append(i[0])
			prev_notes.append(i)
	return s_in, s_out


print ("Generating dataset for "+typemap[input_data_type])
data_parsed=[]
for i in range(len(data_split)):
	print(data_split[i])
	data_parsed.append(parse(data_raw[i]))
	pickle.dump(data_parsed[i],open(f'../rick-{typemap[input_data_type]}-{data_split[i]}','wb'))


## --------------------- EOF ------------------------

def time_vibe_check(data):
	d = [  x[2] for y in data for x in y  ]
	print(max(d), min(d))
	print(np.mean(d), np.median(d))
	print(np.quantile(d,.25), np.quantile(d,.75), np.quantile(d,.95))
	m_value=max(d)#1000
	plt.hist(d,range(m_value), density=True)
	plt.gca().set(title='Frequency histogram for Delta time values', ylabel='Frequency')
	plt.xlim(0,500)
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
