#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:14:19 2021

@author: song-isong-i
"""
import numpy as np
from numpy import linalg as LA
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import os
import json
import torch
from scipy.linalg import sqrtm
from dataset.data_handler import DATA_SCORE, DanceDataset
from net.st_gcn_perceptual import Model as gcnmodel
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor

def calculate_rom(rot_quats, align=0):
    num_frames, num_axis = rot_quats.shape
    
    joints = int(num_axis / 2)

    # Calculate the speed of each frame and the angle between
    speed = np.zeros((num_frames, num_axis))
    angle = np.zeros((num_frames, joints))
    sprot_norm = np.zeros((num_frames, joints))
    for i in range(1, num_frames):
        speed[i] = rot_quats[i] - rot_quats[i - 1]
        for j in range(joints):
            v1 = speed[i - 1, j * 2:j * 2 + 2]
            v2 = speed[i, j * 2:j * 2 + 2]
            if not (np.sum(v1) == 0 and np.sum(v2) == 0):
                angle[i, j] = angle_between(v1, v2)
            sprot_norm[i, j] = LA.norm(v2)
    _beats = []
    active = []
    jnts_beat = []

    # Calculate a candidate beat by matching angle's peak and a
    # local minimum of the speed of each joint.
    for j in range(joints):
        if np.std(angle[:, j]) > 0:
            angle[:, j] /= np.amax(np.abs(angle[:, j]))
            sprot_norm[:, j] /= np.amax(np.abs(sprot_norm[:, j]))
            peak_angle = peak_detect(angle[:, j])
            zero_vel = closezero_detect(sprot_norm[:, j])
            joint_beat = []
            init_frame = 0
            for zero in zero_vel:
                for idx in range(init_frame, init_frame + 20):
                    if (zero >= peak_angle[idx] - align) or (zero <= peak_angle[idx] + align):
                        joint_beat += [zero]
                        init_frame = idx
                        break
            _beats += [joint_beat]
            jnts_beat += joint_beat
            active += [j]
    vel_drop = np.zeros((num_frames, len(active)))
    jnts_beat = np.array(jnts_beat)

    # Calculate the speed drop on each beat
    for j in range(len(active)):
        peak_vel = peak_detect(sprot_norm[:, active[j]])
        for vdp in _beats[j]:
            vpk = np.where(peak_vel < vdp)[0]
            if len(vpk) > 0:
                vpk = peak_vel[vpk[-1]]
                vel_drop[vdp, j] = sprot_norm[vpk, active[j]] - sprot_norm[vdp, active[j]]
    vel_drop = np.sum(vel_drop, axis=1)

    # Process Velocity drops
    min_drop = np.where(vel_drop < np.std(vel_drop))[0]
    vel_drop[min_drop] = -0.1
    drop_cross = np.where(np.diff(np.signbit(vel_drop)))[0]
    for i in range(0, len(drop_cross), 2):
        xi = drop_cross[i] - align
        xj = drop_cross[i + 1] + align + 1
        segment = vel_drop[xi:xj]
        maxs = np.where(segment > 0)[0]
        if len(maxs) > 1:
            max_id = np.where(segment == np.amax(segment))[0][0]
            for j in range(xi, xj):
                if not(j - xi == max_id) and j < vel_drop.shape[0]:
                    vel_drop[j] = -0.1
    drops = np.asarray(np.where(vel_drop > 0)[0])
    
    return vel_drop, drops

    # Match the speed drop with the candidate beats in frames.
#    candidate_beat = []
#    for dp in drops:
#        candidate = np.where(jnts_beat == dp)[0]
#        if len(candidate) > 0:
#            candidate_beat += [dp]
#    candidate_beat = np.sort(np.unique(np.asarray(candidate_beat)))
#    return candidate_beat

def peak_detect(signal):
    # signal in 1D
    gradient = np.gradient(signal)
    zero_cross = np.where(np.diff(np.signbit(gradient)))[0]
    peak = []
    for i in range(0, len(zero_cross) - 2):
        xss1, _, xss2 = zero_cross[i:i + 3]
        portion = signal[xss1:xss2]
        amax = np.amax(np.abs(portion))
        idx = np.where(np.abs(portion) == amax)[0]
        peak += [(xss1 + x) for x in idx]
    peak = np.sort(np.unique(np.asarray(peak)))
    return peak

def closezero_detect(signal):
    # signal in 1D
    gradient = np.gradient(signal)
    zero_cross = np.where(np.diff(np.signbit(gradient)))[0]
    closzero = []
    for i in range(len(zero_cross) - 2):
        xss1, _, xss2 = zero_cross[i:i + 3]
        portion = signal[xss1:xss2]
        amin = np.amin(np.abs(portion))
        idx = np.where(np.abs(portion) == amin)[0]
        closzero += [(xss1 + x) for x in idx]
    return np.asarray(closzero)

def unit_vector(vector):
    """Returns the unit vector of the vector.  """

    div = np.linalg.norm(vector)
    if div != 0:
        return vector / div
    else:
        return vector

def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def extract_beats(musicbeat, motionbeat, align=0):
    beats = []
    for beat in musicbeat:
        extracted = np.where(motionbeat >= beat - align)[0]
        if len(extracted) > 0 and (extracted[0] <= beat + align):
            beats += [motionbeat[extracted[0]]]
    return np.asarray(beats)

from scipy.signal import butter,filtfilt
def butter_highpass(data,cutoff, fs, order=5):
   """
   Design a highpass filter.
   Args:
   - cutoff (float) : the cutoff frequency of the filter.
   - fs     (float) : the sampling rate.
   - order    (int) : order of the filter, by default defined to 5.
   """
   # calculate the Nyquist frequency
   nyq = 0.5 * fs
   # design filter
   high = cutoff / nyq
   b, a = butter(order, high, btype='high', analog=False)
   # returns the filter coefficients: numerator and denominator
   y = filtfilt(b, a, data)
   return y


def get_music_beats(audio):
    wav.write('tmp.wav', 16000, np.int16(audio))
    x, sr = librosa.load('tmp.wav', sr=16000)
    #x=butter_highpass(x,1000, sr, order=5)
    o_env = librosa.onset.onset_strength(x, sr=16000)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=16000)
#    onset_frames = librosa.onset.onset_detect(x, sr=16000)
    onset_frames = librosa.util.peak_pick(o_env, 3, 3, 3, 3, 0.5, 2)
    
#    plt.figure(figsize=(12,5))
#    plt.plot(times, o_env, label='Onset strength')
#    plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
#    linestyle='--', label='Onsets')
#    plt.axis('tight')
#    plt.legend(frameon=True, framealpha=0.75)
#    plt.show()
    
    os.remove('tmp.wav')
    
    return times[onset_frames] 
    
def get_kinematic_beats(pose_sequence):
    pose_sequence = np.array(pose_sequence).reshape(-1,36)
    vel_drop, drops = calculate_rom(pose_sequence)
#    times = np.array(range(len(vel_drop)))/ 10
#    plt.figure(figsize=(12,5))
#    plt.plot(times,vel_drop)
    #plt.vlines(np.array(drops)/10, 0, vel_drop.max(), color='r', alpha=0.9, linestyle='--', label='Onsets') #drops
    return drops/50 * 5
    
def get_aligned_beats(musicbeat, motionbeat):
    align = 0.15
    beats = []
    for beat in musicbeat:
        e1 = np.where(motionbeat >= beat-align)[0]
        e2 = np.where(motionbeat <= beat+align)[0]
        beats.extend(list(np.intersect1d(e1,e2)))
    return list(set(beats))

def get_beat_coverage(audio, pose_sequence):
    musicbeat = get_music_beats(audio)
    motionbeat = get_kinematic_beats(pose_sequence)
    aligned = get_aligned_beats(musicbeat, motionbeat)
    
    Bk = len(motionbeat)
    Bm = len(musicbeat)
    Ba = len(aligned)
    
    beat_coverage = float(Bk) / float(Bm)
    beat_hitrate = float(Ba) / float(Bk)
    
    return beat_coverage, beat_hitrate


# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
    


graph_args={"layout": 'openpose',"strategy": 'spatial'}
gcn = gcnmodel(2,16,graph_args,edge_importance_weighting=True).cuda()
gcn.load_state_dict(torch.load('./pretrain_model/GCN.pth'))
gcn.eval()

from config import get_arguments
parser = get_arguments()
opt = parser.parse_args()

data=DanceDataset(opt)
dataloader = torch.utils.data.DataLoader(data,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=0,
                                     pin_memory=False,
                                     drop_last=True
                                    )

def compute_tr_embeddings(dataloader, count=300):
    embeddings = []
    for i, (x,target,initp) in enumerate(dataloader):
        #target is pose
        pose = Variable(target.type(Tensor))#1,50,18,2
        pose=pose.view(-1,50,36)
        e = gcn.extract_feature(pose)
        embeddings.extend(e)
        if i > count:
            break
    return np.array(embeddings)

def real_beatinfo():
    pass_score=5
    with open('dataset/dance_music_paired.json','r') as of:
        data = json.load(of)
    for k,v in DATA_SCORE.items():
        if v < pass_score:
            del data[k]
    bcs = []
    bhs = []
    for k,v in data.items():
        for i, audio in enumerate(v['music']):
            bc, bh = get_beat_coverage(audio, v['pose'][i])
            bcs.append(bc)
            bhs.append(bh)
    return np.array(bcs).mean(), np.array(bhs).mean()
    
    
    
    
#Bk/Bm : ratio of kinematic beats to musical beats
#Ba/Bk : ratio of aligned kinematic beats to total kinematic beats