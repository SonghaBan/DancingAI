# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:03:55 2021

@author: songhama
"""
import json
import moviepy.editor as mpy
import scipy.io.wavfile as wav
from scipy import interpolate
from visframe import make_frame, make_frame_clean, make_video_openpose
from more_itertools.recipes import grouper
import pandas as pd
import glob, os
import time
import numpy as np
from pydub import AudioSegment, effects  
import multiprocessing as mp

def normalize_audio(filename):
    rawsound = AudioSegment.from_file(filename)  
    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export(filename.replace("audio","audio_norm"), format="wav")


class DataCleaner:
    '''
    recover missing frames and incorrect detections
    Inpput:
        - filename: filename of the pose json. ex) PoseEstimation/output_fast/alphapose-results-fast-14_Trim.json
    '''
    def __init__(self, filename, fps=30):
        self.flabel = filename.split('-')[-1].split('.')[0]
        with open(filename, 'r') as of:
            self.data = json.load(of)
        self.tcks = []
        self.x = []
        self.ys = [[] for i in range(34)]
        self.newdata = []
        self.missing= 0
        self.wrong = 0
        self.fps = fps
        
    def filter_data(self):
        prev = None
        missing = 0
        prevwrong = 0
        
        cur_idx = int(self.data[0]['image_id'].split('.')[0]) - 1
        # cur_idx should be one less than frame_idx
        for i,d in enumerate(self.data):
            frame_idx = int(d['image_id'].split('.')[0])
            if frame_idx == cur_idx: 
                # skip same frame
                continue
            
            #missing frames
            if (frame_idx - cur_idx) > 1:
#                print('missing frame')
                self.missing += (frame_idx - cur_idx -1)
                cur_idx += (frame_idx - cur_idx -1)
                missing += 1
                
            poselist = list(grouper(d['keypoints'], 3))
            pose = pd.DataFrame(poselist)
            
            cur_idx += 1
            if i == 0:
                prev = pose
            else:
                diff = pose - prev
                #misdetection
                absdiff = max(diff.max()[:2].max(), -1* diff.min()[:2].min())
                if absdiff > 50:
                    if prevwrong:
                        prevwrong += 1
                        prev=pose
                        continue
#                    if missing > 3:
#                        pass
#                    else:
#                    prev = pose #TODO: should I save the misdetection as prev pose (1)removed.
                    else:
                        self.wrong += 1
                        prevwrong += 1
#                        prev=pose
                        continue
            
            self.x.append(frame_idx)
            y_idx = 0
            for row in poselist:
                for value in row[:2]:
                    self.ys[y_idx].append(value)
                    y_idx += 1
            prev = pose
            missing = 0
            prevwrong = 0
            
    def interpolate(self):
#        for i in range(0,len(self.x),100):
        for y in self.ys:
#                tck = interpolate.splrep(self.x[i:(i+100)], y[i:(i+100)])
            tck = interpolate.splrep(self.x, y)
            self.tcks.append(tck)
            
    def combine_keypoints(self, idx):
        keypoints = []
        for coor_list in self.ys:
            keypoints.append(coor_list[idx])
        return keypoints
    
    def recover_frame(self, frame_idx):
        keypoints = []
        cnt = -1
        for i in range(0,len(self.x),100):
            if self.x[i] > frame_idx:
                cnt = max(0, cnt)
                break
            cnt+=1
        for tck in self.tcks:
#        for tck in self.tcks[cnt*len(self.ys):(cnt+1)*len(self.ys)]:
            kp = interpolate.splev(frame_idx, tck)
            keypoints.append(float(kp))
        return keypoints
    
    def fill_missing(self):
        frame_idx = 0
        for i,target_idx in enumerate(self.x):
            tmpdata = dict()
            if frame_idx == target_idx:
                tmpdata['image_id'] = frame_idx
                tmpdata['keypoints'] = self.combine_keypoints(i)
                self.newdata.append(tmpdata)
                frame_idx += 1
            
            #recover missing frames
            elif frame_idx < target_idx:
                while frame_idx <= target_idx:
                    tmpdata = dict()
                    tmpdata['image_id'] = frame_idx
                    tmpdata['keypoints'] = self.recover_frame(frame_idx)
                    self.newdata.append(tmpdata)
                    frame_idx += 1
            else:
                print("WRONG", frame_idx, target_idx)
                return
            
    def save(self):
        
        if self.fps!=30:
            newdata = []
            n = 30 // self.fps
            for i,d in enumerate(self.newdata):
                if i%n == 0:
                    newdata.append(d)
            self.newdata = newdata
        
        newfilename = f"data/fixed_pose/{self.flabel}.json"
        with open(newfilename, 'w') as of:
            json.dump(self.newdata, of)
        print("saved json")
        
        make_skeleton_video(newfilename, fps=self.fps)
        print("saved video")
        
            
    def run(self):
        print('cleaning ', self.flabel)
        print('filtering...')
        self.filter_data()
        print('interpolating...')
        self.interpolate()
        print('recovering...')
        self.fill_missing()
        print(f'done / {self.missing} missing frames, {self.wrong} misdetections fixed!')
        
        self.save()
        
def get_filelabel(filename):
    flabel = os.path.basename(filename).split('.')[0]
    if '-' in flabel:
        flabel = flabel.split('-')[-1]
    return flabel
#    return filename.split('/')[-1].split('.')[0]

def extract_audio(filename):
    '''extract audio from the video and save as wav
    Input:
        - filename: filename of the original dance video
    Output:
        - wav file saved in data/audio/ directory
    '''
    #"data/final_videos/29_Trim.mp4"
    video_clip = mpy.VideoFileClip(filename)
    flabel = get_filelabel(filename)
    newfilename = f"data/audio/{flabel}.wav"
    video_clip.audio.write_audiofile(newfilename, fps=16000)
    audio = wav.read(newfilename)
    #save single channel
    wav.write(newfilename, 16000, audio[1][:,0])
#    wav.write(newfilename, 16000, audio[1])
    
def extract_all_audio():
    files = glob.glob('data/final_videos/*.mp4')
    for filename in files:
        if 'multi' in filename:
            continue
        print(filename)
        extract_audio(filename)
        
def attach_audio(videofile, audiofile):
    audio = mpy.AudioFileClip(audiofile)
#    audio = mpy.CompositeAudioClip([mpy.AudioFileClip(f"data/audio/{flabel}.wav")])
    video = mpy.VideoFileClip(videofile)
    video = video.set_audio(audio)
#    video.audio = audio
    video.write_videofile(videofile+".tmp.mp4")
    os.remove(videofile)
    os.rename(videofile+".tmp.mp4", videofile)
    
    
def make_skeleton_video(filename, org=False, fps=30, directory="data/joint_videos/", joints=17):
    '''
    Input:
        - filename: filename of the pose json   
    Output:
        - mp4 video saved in data/joints_video/
    '''
    with open(filename,'r') as of:
        data = json.load(of)
    flabel = get_filelabel(filename)
    if org:
        output_file = f"data/org_joint_videos/{flabel}.mp4"
        make_frame(data, joints=joints, output_filename=output_file)
    else:
        output_file = directory + f"{flabel}.mp4"
        make_frame_clean(data, joints=joints, output_filename=output_file, fps=fps)
    
    audiofile = f"data/audio/{flabel}.wav"
    videofile = output_file
    attach_audio(videofile, audiofile)
    
def make_skeleton_short(pose, audio, fps=10, filename="test.mp4", initpose=None):
    wav.write('tmp.wav', 16000, np.array(audio, dtype=np.int16))
    visualize_normalized(pose, output_file='test.mp4', initpose=initpose)
    attach_audio(filename, 'tmp.wav')
#make_skeleton_short(a,b. initpose=d['pose'][0][0])


def load_jsonfile(filename):
    with open(filename, 'r') as of:
        data = json.load(of)
    return data

def split_pose(filename, s=5, fps=10):
    flabel = get_filelabel(filename)
    data = load_jsonfile(filename)
    audio = wav.read(f"data/audio_norm/{flabel}.wav")[1]
#    wav.write('tmp.wav',16000, audio)
    chunk = fps * s
    audio_chunk = 16000 * s
    newdata = []
    newaudio = []
    i = 0
    while True:
        if (i+1) * chunk >= len(data):
            newdata.append(data[-1*chunk:])
            newaudio.append(audio[-1*audio_chunk:].tolist())
            if len(audio[-1*audio_chunk:].tolist()) != 80000:
                print(flabel, "!", audio[-1*audio_chunk:].tolist())
            break
        newdata.append(data[i*chunk:(i+1)*chunk])
        if len(audio[i*audio_chunk:(i+1)*audio_chunk].tolist()) != 80000:
            print(flabel, i, len(audio), (i+1)*chunk, len(data), len(audio[i*audio_chunk:(i+1)*audio_chunk].tolist()))
        newaudio.append(audio[i*audio_chunk:(i+1)*audio_chunk].tolist())
        i += 1
    return newdata, newaudio

def split_all_files():
    files = glob.glob("data/fixed_pose_norm/*.json")
    data = dict()
    for f in files:
        flabel = get_filelabel(f).split('_')[0]
        pose, music = split_pose(f)
        tmpd = {"pose":[],"music":[]}
        tmpd['pose'].extend(pose)
        tmpd['music'].extend(music)
        data[flabel] = tmpd
    with open("dance_music_paired.json",'w') as of:
        json.dump(data, of)
    return data

def normalize_pose(filename):
    flabel = get_filelabel(filename)
    data = load_jsonfile(filename)
    width, height = get_video_size(f"data/final_videos/{flabel}.mp4")
    dancer_sizes = []
    prev = None
    diff = None
    newdata = []
    #nose: 0, 15,16: rl ankle
    for i, d in enumerate(data):
        pose = pd.DataFrame(list(grouper(d["keypoints"], 2)))
        #add 18th joint (neck)
        pose.loc[17] = (pose.loc[5] + pose.loc[6]) / 2 #neck
        
        nose = pose.loc[0]
        lankle = pose.loc[15]
        rankle = pose.loc[16]
        
        newpose = pose-nose
        if i > 0:
            if i == 1:
                diff = nose - prev
            else:
                diff = diff + nose - prev
            newpose = newpose + diff
            
        #convert into openpose format
        newpose = alpha2openpose(newpose)
        
        newdata.append(newpose)
        prev = nose
            
#        if i == 0:
#            prev = pose
#            newdata.append(pose-nose)
#        else:
#            newdata.append(pose-prev)
#            prev = pose
        dancer_sizes.append(max(abs(rankle-nose)[1], abs(lankle-nose)[1]))
    
    dancer_sizes.sort()
    
    scale = 240 / dancer_sizes[-20]
    print(scale)
    
    for i, d in enumerate(newdata):
        newdata[i] = (d*scale).values.tolist()
        
    newdata = np.array(newdata)
    v_add = 350 - max(newdata[:,10,1].max(), newdata[:,13,1].max()) #rankle lankle
    newdata[:,:,1] += v_add
    
    lankles = newdata[:,13,0].copy()
    lankles.sort()
    if lankles[-20] < 350:
        newdata[:,:,0] += 320
    else:
        newdata[:,:,0] += 320 - min(lankles[-20] - 340, 300)
    
    newdata = newdata.tolist()
    
    return newdata


def normalize_data(filename):
    print(filename)
    flabel = get_filelabel(filename)
    newdata = normalize_pose(filename)
    outjson = f"data/fixed_pose_norm/{flabel}.json"
    with open(outjson, 'w') as of:
        json.dump(newdata, of)
    
    #visualize
    output_file = f"data/joint_videos_norm/{flabel}.mp4"
    visualize_normalized(newdata, output_file=output_file)
#    make_frame_clean(new, joints=17, output_filename=output_file, fps=10)
    audiofile = f"data/audio/{flabel}.wav"
    attach_audio(output_file, audiofile)
    
def visualize_normalized(newdata, output_file='test.mp4', initpose=None):
    newdata = np.array(newdata)
#    v_add = 390 - max(newdata[:,15,1].max(), newdata[:,16,1].max())
#    newdata[:,:,0] += 300
#    newdata[:,:,1] += v_add
#    
#        if i == 0:
#            if not initpose:
#                initp = newdata[0]
#                initp[:,0] = initp[:,0] + 300
#                initp[:,1] = initp[:,1] + 100
#                new.append(initp)
#            else:
#                initp = np.array(initpose)
#                initp[:,0] = initp[:,0] + 300
#                initp[:,1] = initp[:,1] + 100
#                new.append(initp+d)
#        else:
#            new.append(new[i-1] + d)
#    new = np.array(new)
    make_video_openpose(newdata, output_filename=output_file, fps=10)

def alpha2openpose(pose):
    #pose is a dataframe
    newpose = pose.copy()
    # 0 nose
    newpose.loc[1] = pose.loc[17] #neck
    newpose.loc[2] = pose.loc[5] #LShoulder
    newpose.loc[3] = pose.loc[7] #LElbow
    newpose.loc[4] = pose.loc[9] #LWrist
    newpose.loc[5] = pose.loc[6] #RShoulder
    newpose.loc[6] = pose.loc[8] #RElbow 
    newpose.loc[7] = pose.loc[10] #RWrist 
    newpose.loc[8] = pose.loc[11] #LHip 
    newpose.loc[9] = pose.loc[13] #LKnee
    newpose.loc[10] = pose.loc[15] #LAnkle 
    newpose.loc[11] = pose.loc[12] #RHip 
    newpose.loc[12] = pose.loc[14] #RKnee 
    newpose.loc[13] = pose.loc[16] #RAnkle
    newpose.loc[14] = pose.loc[1] #LEye
    newpose.loc[15] = pose.loc[2] #REye
    newpose.loc[16] = pose.loc[3] #LEar
    newpose.loc[17] = pose.loc[4] #REar
    return newpose
    
def normalize_data_mp(filelist):
    for f in filelist:
        normalize_data(f)
        time.sleep(0.1)
  
def get_video_size(filename):
    video_clip = mpy.VideoFileClip(filename)
    return video_clip.size
    
def normalize_all_files(process=8):
    files = glob.glob('data/fixed_pose/*.json')
    print(files)
    jobs = []
    if len(files) % process == 0:
        chunk = int(len(files) / process)
    else:
        chunk = len(files)//process +1
    print(chunk)
    for i in range(process):
        print(i)
        filelist = files[i*chunk:(i+1)*chunk]
        print(i,len(filelist))
        p = mp.Process(target=normalize_data_mp, args=(filelist,))
        print(p)
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
        

def clean_all_files():
    files = glob.glob('PoseEstimation/output_fast/*.json')
    for filename in files:
    #    make_skeleton_video(filename, org=True)
        dc = DataCleaner(filename, fps=10)
        dc.run()
        
def normalize_all_audio():
    files = glob.glob('data/audio/*.wav')
    for filename in files:
        normalize_audio(filename)


def main():
    extract_all_audio()
    clean_all_files()
    normalize_all_audio()
    normalize_all_files()
    split_all_files()
'''
if __name__ == '__main__':
    main()
'''
'''
def get_mfcc(filename):
#    y1, sr1 = librosa.load(filename) 
    y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    mfcc1 = librosa.feature.mfcc(y,sr, n_mfcc=10)   #Computing MFCC values
    return mfcc1

mfcc2 = get_mfcc("data/audio/29_Trim.wav")
#data = librosa.feature.mfcc(y=y, sr=sr, n_fft=1012, hop_length=256, n_mfcc=20)
'''