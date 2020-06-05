import mido
from keras.models import load_model
import numpy as np
import time

model = load_model('model-best.h5')
#warm up
model.predict(np.zeros([1,100,88]))
#default
notes_on=[0 for i in range(88)]
sequence = np.zeros([100,88])
temp = np.zeros([1,88])
treshold = 0.02
corrected = [0 for i in range(88)]

inputs = mido.get_input_names()
keyboards = [i for i in mido.get_input_names() if i.startswith('Digital Keyboard')]
if len(keyboards)==0:
    keyboards=inputs

with mido.open_input(mido.get_input_names()[0]) as port:
    with mido.open_output("stuffs", virtual=True) as out:
        for message in port:
            if message.type=='note_on':
                note = message.note - 21
                if message.velocity and not notes_on[note]:
                    notes_on[note]=1
                    temp[0][note]=1
                    sequence = np.append(sequence,temp,axis=0)[1:]
                    temp[0][note]=0
                    time_taken = time.time()
                    prob=model.predict(sequence.reshape([1,100,88]))[0]
                    time_taken = time.time() - time_taken
                    print(message,prob[note%12],time_taken)
                    if prob[note%12] <= treshold:
                        if prob[(note+1)%12]>prob[(note-1)%12]:
                            message.note+=1
                        else:
                            message.note-=1
                        print("boi, we are correcting", message)
                        corrected[note]=message.note
                if not message.velocity:
                    notes_on[note]=0
                    if corrected[note]:
                        message.note = corrected[note]
                    corrected[note]=0
                out.send(message)
