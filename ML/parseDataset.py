from mido import MidiFile
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from collections import deque
from smallbrain import toolbar_init, toolbar_tick, data_folder, typemap, data_split

# input data type is passed as argument
# default = 0 (basic notes extraction)
input_data_type=0
if len(sys.argv)>1 and sys.argv[1] in set(smallbrain.typemap):
    input_data_type=smallbrain.typemap.index(sys.argv[1])

files = pd.read_csv('maestro/maestro-v2.0.0.csv')[['midi_filename','split']]
data_raw = [list(files[files['split']==x]['midi_filename']) for x in smallbrain.data_split]

lowest_note = 21
highest_note = 87

delta_bins = [1,2,4,8,16,32,64,128,256,512,1024]
velocity_bins = [1,20,36,57,78,90,100]

def extract_notes_basic(x):
    return np.array([j.note-lowest_note for j in x if j.type=='note_on' and j.velocity], dtype=np.uint8)

def extract_notes_basic_velocity(x):
    return np.array([ [j.note-lowest_note, j.velocity] for j in x if j.type=='note_on' and j.velocity], dtype=np.uint8)

def extract_notes_delta(x):
    y = np.cumsum([j.time for j in x])
    #y = [i if i<200 else int(200+np.sqrt(i)) for i in y ] # retarded data smoothing with .9999th quantile onward
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

    return np.array(y, dtype=np.uint16)

def velocity_category(x):
    for i in range(len(velocity_bins)):
        if x < velocity_bins[i]:
            return i
    return len(velocity_bins)

def delta_category(x):
    return min(11, int(x).bit_length())
    #hack, basically what this one does:
    #for i in range(len(freq_bins)):
    #   if x < freq_bins[i]:
    #       return i
    #return len(freq_bins)

def extract_notes_events(x):
    y = np.cumsum([j.time for j in x])
    z = [ [j.note-lowest_note,velocity_category(j.velocity),i] for j,i in zip(x,y) if j.type=='note_on'] # notes with their time
    d = np.array([[z[0][0],z[0][1],delta_category(z[0][2])]] + [ [ z[i-1][0], z[i-1][1], delta_category(z[i][2]-z[i-1][2]) ] for i in range(1,len(z))])
    deltas_only = np.array([z[i][2]-z[i-1][2] for i in range(1,len(z))])
    min_steps = 500
    threshold = 2048 # ... took me some time
    # those guys  - data_parsed[0][379] to data_parsed[0][381]
    # Frédéric Chopin,"12 Etudes, Op. 25",train,2004,2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_02_Track02_wav.midi,2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_02_Track02_wav.wav,1409.28857763
    # Frédéric Chopin,"12 Etudes, Op. 25",train,2004,2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi,2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.wav,327.327025835
    # Frédéric Chopin,"24 Preludes, Op. 28",train,2004,2004/MIDI-Unprocessed_XP_06_R1_2004_01_ORIG_MID--AUDIO_06_R1_2004_01_Track01_wav.midi,2004/MIDI-Unprocessed_XP_06_R1_2004_01_ORIG_MID--AUDIO_06_R1_2004_01_Track01_wav.wav,2398.64035989
    # they have multiple pieces in the same file
    # Chopin is amazing and I like those performances very much, but, wtf
    sz = len(deltas_only)
    start = int((sz+9)/10)
    # ignore first and last 10 percent when doing the cuts
    # can't remember what i was doing here, but it works.
    # Dont touch!
    split_stuff = []
    last_split = 0
    for i in range(start,sz-start):
        if deltas_only[i]>threshold and (i-last_split) > min_steps:
            split_stuff.append(d[last_split:i])
            last_split=i

    split_stuff.append(d[last_split:len(d)])
    return split_stuff

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
    thing = []
    
    toolbar_cnt = toolbar_init(60,cnt)
    begin = time.time()
    
    for i in range(cnt):
        name = f'maestro/{data[i]}'
        x=MidiFile(name)
        toolbar_tick(i%toolbar_cnt)
        result = extract_notes(x.tracks[1],input_data_type)
        if input_data_type==4:
            #print (i, len(result))
            for songs_split in result:
                thing.append(songs_split)
        else:
            thing.append(result)
    sys.stdout.write("] Done in %s sec!\n" % ('{0:.2f}'.format(time.time()-begin))) # this ends the progress barprint("done!")
    return np.array(thing)


print ("Generating dataset for "+smallbrain.typemap[input_data_type])
data_parsed=[]
for i in range(len(smallbrain.data_split)):
    print(smallbrain.data_split[i])
    data_parsed.append(parse(data_raw[i]))
    pickle.dump(data_parsed[i],open(f'{smallbrain.data_folder}/rick-{smallbrain.typemap[input_data_type]}-{smallbrain.data_split[i]}','wb'))


## --------------------- EOF ------------------------