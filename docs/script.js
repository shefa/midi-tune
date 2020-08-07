let OCTAVES = 7;

let keyWhitelist;

const heldButtonToVisualData = new Map();

// Which notes the pedal is sustaining.
let sustaining = false;
let sustainingNotes = [];

// Mousedown/up events are weird because you can mouse down in one element and mouse up
// in another, so you're going to lose that original element and never mouse it up.
let mouseDownButton = null;

const player = new Player();
//const rnn = new mm.MusicRNN('https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/basic_rnn')
//const genie = new mm.PianoGenie(CONSTANTS.GENIE_CHECKPOINT);
const painter = new FloatyNotes();
const piano = new Piano();

let miditune;
let midistate = [];

let treshold = 0.02;

initEverything();

/*************************
 * Basic UI bits
 ************************/
function initEverything() {
  tf.loadLayersModel("model.json").then(
    model => {
      miditune = model;
      //rnn.initialize().then(()=>{
      playBtn.textContent = "Play";
      playBtn.removeAttribute("disabled");
      playBtn.classList.remove("loading");
      //});
    },
    err => console.log("helmp!")
  );

  // Start the drawing loop.
  onWindowResize();
  updateButtonText();
  window.requestAnimationFrame(() => painter.drawLoop());

  window.addEventListener("resize", onWindowResize);
  window.addEventListener("orientationchange", onWindowResize);
}

function showMainScreen() {
  document.querySelector(".splash").hidden = true;
  document.querySelector(".loaded").hidden = false;

  // Output.
  radioMidiOutYes.addEventListener("click", () => {
    player.usingMidiOut = true;
    midiOutBox.hidden = false;
  });
  radioAudioYes.addEventListener("click", () => {
    player.usingMidiOut = false;
    midiOutBox.hidden = true;
  });
  // Input.
  radioMidiInYes.addEventListener("click", () => {
    player.usingMidiIn = true;
    midiInBox.hidden = false;
    updateButtonText();
  });
  radioDeviceYes.addEventListener("click", () => {
    player.usingMidiIn = false;
    midiInBox.hidden = true;
    updateButtonText();
  });

  var slider = document.getElementById("myRange");
  var output = document.getElementById("demo");
  output.innerHTML = slider.value; // Display the default slider value

  // Update the current slider value (each time you drag the slider handle)
  slider.oninput = function() {
    output.innerHTML = this.value;
    treshold = 1 - this.value / 100;
  };

  // Figure out if WebMidi works.
  if (navigator.requestMIDIAccess) {
    midiNotSupported.hidden = true;
    radioMidiInYes.parentElement.removeAttribute("disabled");
    radioMidiOutYes.parentElement.removeAttribute("disabled");
    navigator
      .requestMIDIAccess()
      .then(
        midi => player.midiReady(midi),
        err => console.log("Something went wrong", err)
      );
  } else {
    midiNotSupported.hidden = false;
    radioMidiInYes.parentElement.setAttribute("disabled", true);
    radioMidiOutYes.parentElement.setAttribute("disabled", true);
  }

  // Slow to start up, so do a fake prediction to warm up the model.
  tf.tidy(() => {
    miditune.predict(tf.zeros([1, 100, 88]));
  });
  for (var i = 0; i < 100; i++) {
    midistate.push(tf.zeros([88], "int32"));
  }
  // const note = genie.nextFromKeyWhitelist(0, keyWhitelist, TEMPERATURE);
  //genie.resetState();
}

/*************************
 * Button actions
 ************************/
function note_on(note, velocity, delta) {
  if (note == 128) sustaining = !sustaining;
  if (velocity) buttonDown(note);
  else buttonUp(note);
}
function buttonDown(note) {
  if (heldButtonToVisualData.has(note)) {
    return;
  }
  button = 0;
  originalnote = note;
  
  
  midistate.push(tf.oneHot(note, 88));
  midistate[0].dispose();
  midistate.shift();

  // const note = genie.nextFromKeyWhitelist(BUTTON_MAPPING[button], keyWhitelist, TEMPERATURE);
  const asd = tf.tidy(() => {
    return miditune
      .predict(tf.stack(midistate).reshape([1, 100, 88]))
      .arraySync()[0];
  });
  console.log(asd);
  if (asd[note % 12] < treshold) {
    if (asd[(note + 1) % 12] > asd[(note - 1) % 12]) note += 1;
    else note -= 1;
    button = 1;
  }
  
  const pitch = CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE + note;
  
  // Hear it.
  player.playNoteDown(pitch, button);

  // See it.
  const rect = piano.highlightNote(note, button);

  if (!rect) {
    debugger;
  }
  // Float it.
  const noteToPaint = painter.addNote(
    button,
    rect.getAttribute("x"),
    rect.getAttribute("width")
  );
  heldButtonToVisualData.set(originalnote, {
    rect: rect,
    note: note,
    noteToPaint: noteToPaint
  });
}

function buttonUp(button) {
  const thing = heldButtonToVisualData.get(button);
  if (thing) {
    // Don't see it.
    piano.clearNote(thing.rect);

    // Stop holding it down.
    painter.stopNote(thing.noteToPaint);

    // Maybe stop hearing it.
    const pitch = CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE + thing.note;
    if (!sustaining) {
      player.playNoteUp(pitch, button);
    } else {
      sustainingNotes.push(CONSTANTS.LOWEST_PIANO_KEY_MIDI_NOTE + thing.note);
    }
  }
  heldButtonToVisualData.delete(button);
}

/*************************
 * Events
 ************************/

function onWindowResize() {
  OCTAVES = window.innerWidth > 700 ? 7 : 3;
  const bonusNotes = OCTAVES > 6 ? 4 : 0; // starts on an A, ends on a C.
  const totalNotes = CONSTANTS.NOTES_PER_OCTAVE * OCTAVES + bonusNotes;
  const totalWhiteNotes =
    CONSTANTS.WHITE_NOTES_PER_OCTAVE * OCTAVES + (bonusNotes - 1);
  keyWhitelist = Array(totalNotes)
    .fill()
    .map((x, i) => {
      if (OCTAVES > 6) return i;
      // Starting 3 semitones up on small screens (on a C), and a whole octave up.
      return i + 3 + CONSTANTS.NOTES_PER_OCTAVE;
    });

  piano.resize(totalWhiteNotes);
  painter.resize(piano.config.whiteNoteHeight);
  piano.draw();
}

/*************************
 * Utils and helpers
 ************************/

function updateButtonText() {
  const btns = document.querySelectorAll(".controls button.color");
  for (let i = 0; i < btns.length; i++) {
    btns[i].innerHTML = `<span>${i + 1}</span><br><span>${
      BUTTONS_DEVICE[i]
    }</span>`;
  }
}
