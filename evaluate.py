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
from dataset.data_handler import DATA_SCORE
from net.st_gcn_perceptual import Model as gcnmodel
from model.HCN_D import seq_discriminator
from torch.autograd import Variable
import glob, random
from utils import load_jsonfile, load_audio, split_audio
Tensor = torch.cuda.FloatTensor

targets = ['baseline','original', 'lstm', 'gru']

# =============================================================================
# Beat Coverage and Beat hit rate
# =============================================================================
def evaluate_beats():
    '''
    beat coverage/hit rate of the test results
    '''
    for t in targets:
        bc, bh = get_beatinfo(f'../output/{t}.json')
        print(f'{t}: {bc} / {bh}')
        

def real_beatinfo():
    '''
    beat coverage/hit rate of the training dataset
    '''
    data = load_tr_data()
    bcs = []
    bhs = []
    for k,v in data.items():
        if int(k)>80:
            continue
        for i, audio in enumerate(v['music']):
            bc, bh = get_beat_coverage(audio, v['pose'][i])
            bcs.append(bc)
            bhs.append(bh)
    return np.array(bcs).mean(), np.array(bhs).mean()

def get_beatinfo(filename):
    data = load_jsonfile(filename)
    bcs = []
    bhs = []
    for i, audio in enumerate(data['music']):
        bc, bh = get_beat_coverage(audio, data['pose'][i])
        bcs.append(bc)
        bhs.append(bh)
    return np.array(bcs).mean(), np.array(bhs).mean()

def get_beat_coverage(audio, pose_sequence):
    musicbeat = get_music_beats(audio, aligned=False)
    motionbeat = get_kinematic_beats(pose_sequence)
    aligned = get_aligned_beats(audio, motionbeat)
    
    Bk = len(motionbeat)
    Bm = len(musicbeat)
    Ba = len(aligned)
    
    beat_coverage = float(Bk) / float(Bm)
    beat_hitrate = float(Ba) / float(Bk)
    
    return beat_coverage, beat_hitrate

def extract_beats(musicbeat, motionbeat, align=0):
    beats = []
    for beat in musicbeat:
        extracted = np.where(motionbeat >= beat - align)[0]
        if len(extracted) > 0 and (extracted[0] <= beat + align):
            beats += [motionbeat[extracted[0]]]
    return np.asarray(beats)
    
def calculate_rom(rot_quats, align=0):
    '''
    Calculate kinematic beats.
    This fucntion was adapted from: https://github.com/Fhrozen/motion_dance
    '''
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


def get_music_beats(audio, aligned=True):
    wav.write('tmp.wav', 16000, np.int16(audio))
    x, sr = librosa.load('tmp.wav', sr=16000)
    o_env = librosa.onset.onset_strength(x, sr=16000)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=16000)
    if aligned:
        onset_frames = librosa.onset.onset_detect(x, sr=16000)
    else:
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
    
def get_aligned_beats(audio, motionbeat):
    musicbeat = get_music_beats(audio, aligned=True)
    align = 0.15
    beats = []
    for beat in musicbeat:
        e1 = np.where(motionbeat >= beat-align)[0]
        e2 = np.where(motionbeat <= beat+align)[0]
        beats.extend(list(np.intersect1d(e1,e2)))
    return list(set(beats))



def load_tr_data(clean=False):
    pass_score=5
    with open('dataset/dance_music_paired.json','r') as of:
        data = json.load(of)
    for k,v in DATA_SCORE.items():
        if v < pass_score:
            del data[k]
    if clean:
        new = {'pose':[],'music':[]}
        for k,v in data.items():
            if int(k) > 80:
                continue
            new['pose'].extend(v['pose'])
            new['music'].extend(v['music'])
                
        return new
            
    return data


def merge_testfiles(files,flabel):
    new = {'music':[], 'pose':[]}
    for f in files:
        data = load_jsonfile(f)
        new['music'].extend(data['music'])
        new['pose'].extend(data['pose'])
    with open(f'../output/{flabel}.json', 'w') as of:
        json.dump(new, of)
        
def merge_all_testfiles():
    blfiles = glob.glob('../output/baseline/result_t*.json')
    orgfiles = glob.glob('../output/normal3/800/result_t*.json')
    lstmfiles = glob.glob('../output/musicinitplstmminng/800/result_t*.json')
    grufiles = glob.glob('../output/musicinitpmin/350/result_t*.json')
    merge_testfiles(blfiles, 'baseline')
    merge_testfiles(orgfiles, 'original')
    merge_testfiles(lstmfiles, 'lstm')
    merge_testfiles(grufiles, 'gru')


# =============================================================================
# FID MEASURE
# =============================================================================
def calculate_frechet_distance(act1, act2, eps=1e-6):
    """
    code from https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid
    
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
 
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

class FID:
    def __init__(self):
        self.D = seq_discriminator(16, 'musicinitpmin')
        self.D.load_state_dict(torch.load('log/musicinitpmin/sequence_0350.pth'))
        self.D.cuda()

        trdata = load_tr_data(clean=True)
        trdata = CustomDataLoader(trdata)
        dataloader = torch.utils.data.DataLoader(trdata,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=False)
        
        self.tr_embeddings = self.compute_embeddings(dataloader)
        trdata = None
        dataloader = None

    def compute_embeddings(self, dataloader):
        embeddings = []
        for i, x in enumerate(dataloader):
            pose = Variable(x.type(Tensor))#1,50,18,2
            if pose.size()[0] != 16:
                continue
            pose=pose.view(-1,50,36)
            e = self.D.extract_feature(pose)[5]
            embeddings.extend(e.cpu().data.numpy().reshape(e.size(0), -1))
        return np.array(embeddings)
    
    def compute_fid(self, filename):
        data = load_jsonfile(filename)
        data = CustomDataLoader(data)
        dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=False)
        embeddings = self.compute_embeddings(dataloader)
        meanfids = []
        minfids = []
        for i,e in enumerate(embeddings):
            tmpfids = []
            for te in self.tr_embeddings:
                tmpfids.append(calculate_frechet_distance(e, te))
            meanfids.append(np.mean(tmpfids))
            minfids.append(np.min(tmpfids))
        return meanfids, minfids
    
    def real_fid(self):
        meanfids = []
        minfids = []
        for i,e in enumerate(self.tr_embeddings):
            tmpfids = []
            for te in self.tr_embeddings:
                tmpfids.append(calculate_frechet_distance(e,te))
            meanfids.append(np.mean(tmpfids))
            minfids.append(np.min(tmpfids))
        return meanfids, minfids
    
    def diversity(self, filename):
        data = load_jsonfile(filename)
        data = CustomDataLoader(data)
        dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=False)
        embeddings = self.compute_embeddings(dataloader)
        meanfids = []
        for i,e in enumerate(embeddings):
            tmpfids = []
            for te in embeddings:
                tmpfids.append(calculate_frechet_distance(e, te))
            meanfids.append(np.mean(tmpfids))
        return meanfids
    
def get_diversity():
    fid = FID()
    for t in targets:
        filename = f'../output/{t}.json'
        means = fid.diversity(filename)
        print(filename, np.mean(means))
    
    
    
def print_fid_result(label, means, mins):
    print(label)
    print("mean of means: ", np.mean(means))
    print("min of means: ", np.min(means))
    print("mean of mins: ", np.mean(mins))
    print("min of mins: ", np.min(mins))
    print('-------------------------------')
    
def get_fids():
    fid = FID()
    means, mins = fid.real_fid()
    print_fid_result("REAL", means, mins)
    
    for t in targets:
        filename = f'../output/{t}.json'
        means, mins = fid.compute_fid(filename)
        print_fid_result(t, means, mins)
#        print(f'{t}: {r}')

class CustomDataLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        length = len(data['music'])
        self.length = length
#        music=torch.FloatTensor(length,50,1600).zero_() #music
        pose=torch.FloatTensor(length,50,18,2).zero_()
        
        for i, audio in enumerate(data['music']):
            temp_pose = np.array(data['pose'][i])
            temp_pose[:,:,0]=(temp_pose[:,:,0]/320)-1
            temp_pose[:,:,1]=(temp_pose[:,:,1]/180)-1
            pose[i] = torch.from_numpy(temp_pose)
            
#            temp_audio = np.array(audio)
#            music[i] = torch.from_numpy(temp_audio).type(torch.LongTensor).view(50,1600)
        self.pose = pose


    def __getitem__(self, idx):
        #print("idx:",idx)
        target=self.pose[idx]
        return target

    def __len__(self):
        return self.length

def tensorfy(data):
    length = len(data['music'])
    music=torch.FloatTensor(length,50,1600).zero_() #music
    pose=torch.FloatTensor(length,50,18,2).zero_()
    
    for i, audio in enumerate(data['music']):
        temp_pose = np.array(data['pose'][i])
        temp_pose[:,:,0]=(temp_pose[:,:,0]/320)-1
        temp_pose[:,:,1]=(temp_pose[:,:,1]/180)-1
        pose[i] = torch.from_numpy(temp_pose)
        
        temp_audio = np.array(audio)
        music[i] = torch.from_numpy(temp_audio).type(torch.LongTensor).view(50,1600)    
        
    return music, pose

# =============================================================================
# Analyze Survey Results for qualitative evaluation
# =============================================================================

from more_itertools.recipes import grouper
def match_answer(response):
    answer = ['B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
    evaluation = []
    for i,r in enumerate(response):
        if r == answer[i]:
            evaluation.append(1)
        else:
            evaluation.append(0)
    evaluation = list(grouper(evaluation, 4))
    return evaluation

def analyze_answers(response):
    surveyr = []
    for r in response:
        surveyr.append(match_answer(r))
    surveyr = np.array(surveyr)
    beatmatching = surveyr[:,:,0]
    natural = surveyr[:,:,1]
    dynamic = surveyr[:,:,2]
    like = surveyr[:,:,3]
    
    plot_result(beatmatching, 'Beat matching')
    plot_result(natural, 'Realistic')
    plot_result(dynamic, 'Dynamic')
    plot_result(like, 'Overall preference')
    
    return surveyr.mean()
    
#    print("Beat matching")
#    print(beatmatching)
#    print(f"Mean: {beatmatching.mean()}")
#    print(f"Mean/song: {beatmatching.mean(axis=0)} / {beatmatching.mean(axis=0).mean()}")
#    print(f"Mean/person: {beatmatching.mean(axis=1)} / {beatmatching.mean(axis=1).mean()}")
#    print('------------------------------')
#    print("Natural")
#    print(natural)
#    print(f"Mean: {natural.mean()}")
#    print(f"Mean/song: {natural.mean(axis=0)} / {natural.mean(axis=0).mean()}")
#    print(f"Mean/person: {natural.mean(axis=1)} / {natural.mean(axis=1).mean()}")
#    print('------------------------------')
#    print("Dynamic")
#    print(dynamic)
#    print(f"Mean: {dynamic.mean()}")
#    print(f"Mean/song: {dynamic.mean(axis=0)} / {dynamic.mean(axis=0).mean()}")
#    print(f"Mean/person: {dynamic.mean(axis=1)} / {dynamic.mean(axis=1).mean()}")
#    print('------------------------------')
#    print("Like")
#    print(like  )
#    print(f"Mean: {like.mean()}")
#    print(f"Mean/song: {like.mean(axis=0)} / {like.mean(axis=0).mean()}")
#    print(f"Mean/person: {like.mean(axis=1)} / {like.mean(axis=1).mean()}")
    
def plot_result(sr,label):
    sr = sr*100
    pos = sr.mean(axis=0)
    neg = 100-pos
    dances = [1,2,3,4,5,6]
    plt.barh(dances, neg,  label='original model', left=pos, color='#7979d2')
    plt.barh(dances, pos, label='my model', color='#ff9933')
    for i,pv in enumerate(pos):
        x = pv-pv/3
        if pv < 15:
            x = 3
        elif pv < 40:
            x = x - 10
        plt.text(x,i+1-0.1,f"{pv:.2f}%")
    for i,nv in enumerate(neg):
        plt.text(100-nv+nv/5,i+1-0.1,f"{nv:.2f}%")
        
    plt.ylabel("Dance")
#    plt.xlabel(label)
    plt.title(label)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    
    plt.show()
    
def main():
    #merge test files
    #merge_all_testfiles()
    #evaluate_beats()
    #get_fids()
    #response = []
    #analyze_answers(response)
    return
    
    
#main()
    
