//const rnn = new mm.MusicRNN('https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/basic_rnn')
//const genie = new mm.PianoGenie(CONSTANTS.GENIE_CHECKPOINT);

const pressedKeys = Array(88).fill(new Map());
const lastSuch = Array(88).fill(0);

let sustaining = false;
let sustainingNotes = [];

let miditune;
let midistate = [];

let treshold = 0.02;

// load tensorflowJS model
tf.loadLayersModel("/model.json").then(
    model => {
      miditune = model;

      // Slow to start up, so do a fake prediction to warm up the model.
      tf.tidy(() => { miditune.predict(tf.zeros([1, 100, 88])); });
      for (var i = 0; i < 100; i++) { midistate.push(tf.zeros([88], "int32")); }
      //rnn.initialize().then(()=>{
      playBtn.textContent = "Play";
      playBtn.removeAttribute("disabled");
      playBtn.classList.remove("loading");
      var slider = document.getElementById("myRange");
      var output = document.getElementById("demo");
      output.innerHTML = slider.value;
      slider.oninput = function() {
        output.innerHTML = this.value;
        treshold = 1 - this.value / 100;
      };

      //});
    },
    err => console.log("helmp!")
  );

/*************************
 * Model logic & usage
 ************************/

function add_note_to_tensor(note){
  let tempHot = Array(88).fill(0);
  tempHot[note]=1;
  midistate.push(tf.tensor(tempHot,[88],'int32'));
  midistate[0].dispose();
  midistate.shift();
}

async function predict_next_note(){
  const asd = tf.stack(midistate);
  const bsd = asd.reshape([1,100,88]);
  const ret = await miditune.predict(bsd).array();
  asd.dispose();
  bsd.dispose();
  return ret[0];
}

async function correct_note(note){
  const prediction = await predict_next_note();

  if (prediction[note % 12] <= treshold/5) {
    if (prediction[(note + 1) % 12] > prediction[(note - 1) % 12]) return note+1;
    return note-1;
  }
  return note;
}

function midi_message_in(command, button, velocity){
  switch (command) {
      case 0x90: // note on
        window.note_on(button,velocity);
        break;
      case 0x80: // note off
        window.note_off(button);
        break;
      case 176:  // sustain
        window.sustain_pedal();
        break;
      case 177:  // sustain
        window.sustain_pedal();
        break;
      default:
        console.log("Error: not supported midi command", command, velocity);
        break;
  }
}

function sustain_pedal(){
  sustaining = !sustaining;
  if(!sustaining){
      stopSustaining(sustainingNotes);
      sustainingNotes = [];
  }
}

function note_on(note, velocity) {
  lastSuch[note]=performance.now();
  buttonDown(note,velocity,lastSuch[note]);
}

function note_off(note){
  if(lastSuch[note]!=0) buttonUp(note, lastSuch[note], performance.now()-lastSuch[note]);
  lastSuch[note]=0;
}

async function buttonDown(note,velocity,timeNow) {
  let modified = 0;
  const originalnote = note;
  const perf = performance.now();
  
  add_note_to_tensor(note);
  
  if (treshold<0.99){
    note = await correct_note(note);
    if (note!=originalnote) modified=1;
  }

  console.log(performance.now()-perf);

  const pitch = CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE + note;
  pressedKeys[originalnote].set(timeNow,playNoteDown(pitch,modified,note));
}

function buttonUp(button, timeNow, duration) {
  // if not yet registered the buttonDown completely, wait some time
  if (!pressedKeys[button].has(timeNow)){ setTimeout(function(){ buttonUp(button, timeNow, duration); }, 20); return; }

  let thing = pressedKeys[button].get(timeNow);
  const oshte_kolko = thing.timeNow + duration - performance.now();
    setTimeout(function(){
        playNoteUp(thing,sustaining);
        if (sustaining) {
          sustainingNotes.push(CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE + thing.note);
        }
    }, oshte_kolko);
  pressedKeys[button].delete(timeNow);
}


/* -------- Test miditune vs performanceRNN vs melodyRNN in their JS implementations -----------
ƒ test() { // around 100-110ms
t = performance.now();
     melody.continueSequenceAndReturnProbabilities(bsd,1).then((res)=>{console.log(performance.now()-t,res.probs[0]);}); 
}
test2
ƒ test2(){ //99-110 ms
    t=performance.now(); around 100-110ms
    rnn.continueSequenceAndReturnProbabilities(bsd,1).then((res)=>{console.log(performance.now()-t,res.probs[0]);});
}
test3
ƒ test3() {  //180-200
t = performance.now();
    miditune.predict(tf.zeros([1, 100, 88])).array().then((res)=>{console.log(performance.now()-t,res[0]); }); }
*/