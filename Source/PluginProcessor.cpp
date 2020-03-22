#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "PluginParameter.h"

//==============================================================================

MidiTuneAudioProcessor::MidiTuneAudioProcessor():
#ifndef JucePlugin_PreferredChannelConfigurations
    AudioProcessor (BusesProperties()
                    #if ! JucePlugin_IsMidiEffect
                     #if ! JucePlugin_IsSynth
                      .withInput  ("Input",  AudioChannelSet::stereo(), true)
                     #endif
                      .withOutput ("Output", AudioChannelSet::stereo(), true)
                    #endif
                   ),
#endif
    parameters (*this)
	, keyParam(parameters, "Key", {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"})
	, scaleParam(parameters, "Scale/Mode", { "Ionian (Major)", "Dorian", "Phrygian","Lydian","Mixolydian","Aeolian (Minor)","Locrian" }, 0, [&](float value) { return updateParams(value); })
    , correctionType (parameters, "Correction type", {"Random","Heuristic 1"})
	, correctionRate(parameters, "Min confidence", "", 0.0f, 1.0f, 1.0f)
{
    parameters.valueTreeState.state = ValueTree (Identifier (getName().removeCharacters ("- ")));
	for (int i = 0; i < 12; i++) allowedNotes[i] = defaultAllowed[i];
}

MidiTuneAudioProcessor::~MidiTuneAudioProcessor()
{
}

//==============================================================================

void MidiTuneAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    const double smoothTime = 1e-3;
    keyParam.reset (sampleRate, smoothTime);
    scaleParam.reset (sampleRate, smoothTime);
    correctionType.reset (sampleRate, smoothTime);
    correctionRate.reset (sampleRate, smoothTime);
}

void MidiTuneAudioProcessor::releaseResources()
{
}

float MidiTuneAudioProcessor::updateParams(float x) {
	DBG("changing shit");
	int offset = (int)x+1;
	for (int i = 0, j = 0; i < 12; i++) {
		if (defaultAllowed[i]) j++;
		if (j == offset) {
			offset = i;
			break;
		}
	}
	String sx = "";
	for (int i = 0; i < 12; i++) {
		sx += allowedNotes[i];
	}
	DBG(sx);
	for (int i = 0; i < 12; i++) {
		allowedNotes[i] = defaultAllowed[(i + offset) % 12];
	}
	sx = "";
	for (int i = 0; i < 12; i++) {
		sx += allowedNotes[i];
	}
	DBG(sx);
	return x;
}

void MidiTuneAudioProcessor::processBlock (AudioSampleBuffer& buffer, MidiBuffer& midiMessages)
{
    ScopedNoDenormals noDenormals;

    MidiBuffer processedMidi;
    MidiMessage message;
    int time;

	int note;
	int key=(int)keyParam.getCurrentValue();

    for (MidiBuffer::Iterator iter (midiMessages); iter.getNextEvent (message, time);) {
		note = ( message.getNoteNumber() - key )% 12;
		DBG(note);
		message = MidiMessage::noteOn(message.getChannel(), message.getNoteNumber(), message.getVelocity());
		if (!allowedNotes[note]) {
			DBG("error");
			message = MidiMessage::noteOn(message.getChannel(), message.getNoteNumber()+1, message.getVelocity());
		}
        processedMidi.addEvent (message, time);
    }

    midiMessages.swapWith (processedMidi);
}

//==============================================================================






//==============================================================================

void MidiTuneAudioProcessor::getStateInformation (MemoryBlock& destData)
{
    auto state = parameters.valueTreeState.copyState();
    std::unique_ptr<XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void MidiTuneAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.valueTreeState.state.getType()))
            parameters.valueTreeState.replaceState (ValueTree::fromXml (*xmlState));
}

//==============================================================================

AudioProcessorEditor* MidiTuneAudioProcessor::createEditor()
{
    return new MidiTuneAudioProcessorEditor (*this);
}

bool MidiTuneAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

//==============================================================================

#ifndef JucePlugin_PreferredChannelConfigurations
bool MidiTuneAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

//==============================================================================

const String MidiTuneAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool MidiTuneAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool MidiTuneAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool MidiTuneAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double MidiTuneAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

//==============================================================================

int MidiTuneAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int MidiTuneAudioProcessor::getCurrentProgram()
{
    return 0;
}

void MidiTuneAudioProcessor::setCurrentProgram (int index)
{
}

const String MidiTuneAudioProcessor::getProgramName (int index)
{
    return {};
}

void MidiTuneAudioProcessor::changeProgramName (int index, const String& newName)
{
}

//==============================================================================

// This creates new instances of the plugin..
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MidiTuneAudioProcessor();
}

//==============================================================================
