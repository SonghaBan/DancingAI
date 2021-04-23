import os
import json

def get_filelabel(filename):
    flabel = os.path.basename(filename).split('.')[0]
    if '-' in flabel:
        flabel = flabel.split('-')[-1]
    return flabel
#    return filename.split('/')[-1].split('.')[0]

def load_jsonfile(filename):
    with open(filename, 'r') as of:
        data = json.load(of)
    return data

def load_audio(filename):
	audio = wav.read(filename)[1]
	return audio

def split_audio(audio, s=5):
    audio_chunk = 16000 * s
    newaudio = []
    i = 0
    while True:
        if (i+1) * audio_chunk >= len(audio):
            newaudio.append(audio[-1*audio_chunk:].tolist())
            break
        newaudio.append(audio[i*audio_chunk:(i+1)*audio_chunk].tolist())
        i += 1
    return newaudio