import os
import json
import gizeh as gz
import numpy as np
import moviepy.editor as mpy
import scipy.io.wavfile as wav
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

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


l_pair_openpose = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    ]
#for plot usage
colors_openpose = np.array([
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]])/255

def make_pose_video(data, output_filename="../test.mp4", save_video=True, fps=10):
    width = 640
    height = 360
    joints = 18
    print("make! fps:",fps, len(data))
    
    result = []
    for frame in data:
#        print(cur_idx, frame['image_id'])
        
        surface = gz.Surface(width=width, height=height, bg_color=(1,1,1))
    
        line_cnt = 0
        l_pair = l_pair_openpose
        line_color = colors_openpose
        
        for limb in l_pair:
            x1, y1 = frame[limb[0]]
            x2, y2 = frame[limb[1]]
            line = gz.polyline(points=[(x1,y1), (x2,y2)], stroke_width = 5, stroke=line_color[line_cnt])
#            print('line', line_cnt)
            line_cnt += 1
            line.draw(surface)
        
        for idx in range(len(frame)):
#            print(idx)
            x1, y1 = frame[idx]
            if idx < 14:
                pcolor = np.array(BLACK) / 255
            else:
                pcolor = np.array(BLACK) / 255
            joint = gz.circle(3, xy=[x1,y1], fill=pcolor)
            joint.draw(surface)

        result.append(surface.get_npimage())
    
    if save_video:
        clip = mpy.ImageSequenceClip(result, fps=fps)
        clip.write_videofile(output_filename, fps=fps, codec='mpeg4') 

def attach_audio(videofile, audiofile):
    audio = mpy.AudioFileClip(audiofile)
#    audio = mpy.CompositeAudioClip([mpy.AudioFileClip(f"data/audio/{flabel}.wav")])
    video = mpy.VideoFileClip(videofile)
    video = video.set_audio(audio)
#    video.audio = audio
    video.write_videofile(videofile+".tmp.mp4")
    os.remove(videofile)
    os.rename(videofile+".tmp.mp4", videofile)