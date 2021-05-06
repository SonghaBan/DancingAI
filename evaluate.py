#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:14:19 2021

@author: song-isong-i
"""
import numpy as np
from numpy import linalg as LA

def calculate_rom(rot_quats, align=0, fps=30):

    """The code was implemented based on the following papers:

      [1] Chieh Ho, Wei-Tze Tsai, Keng-Sheng Lin, and Homer H Chen, National Taiwan University,
          Extraction and Alignment Evaluation of Motion Beats for Street Dance, ICASSP 2013

      [2] Wei-Ta Chu, Member, IEEE, and Shang-Yin Tsai, National Chung Cheng University,
          Rhythm of Motion Extraction and Rhythm-Based Cross-Media Alignment for Dance Videos,
          IEEE Transactions on Multimedia ( Volume: 14, Issue: 1, Feb. 2012 )

    [1]'s algorithm was modified for rotations

    """

    num_frames, num_axis = rot_quats.shape
    joints = int(num_axis / 3)

    # Calculate the speed of each frame and the angle between
    speed = np.zeros((num_frames, num_axis))
    angle = np.zeros((num_frames, joints))
    sprot_norm = np.zeros((num_frames, joints))
    for i in range(1, num_frames):
        speed[i] = rot_quats[i] - rot_quats[i - 1]
        for j in range(joints):
            v1 = speed[i - 1, j * 3:j * 3 + 3]
            v2 = speed[i, j * 3:j * 3 + 3]
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

    # Match the speed drop with the candidate beats in frames.
    candidate_beat = []
    for dp in drops:
        candidate = np.where(jnts_beat == dp)[0]
        if len(candidate) > 0:
            candidate_beat += [dp]
    candidate_beat = np.sort(np.unique(np.asarray(candidate_beat)))
    return candidate_beat

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


class SimpleBeatDetection:
    """
    Simple beat detection algorithm from
    http://archive.gamedev.net/archive/reference/programming/features/beatdetection/index.html
    """
    def __init__(self, history = 43):
        self.local_energy = np.zeros(history) # a simple ring buffer
        self.local_energy_index = 0 # the index of the oldest element

    def detect_beat(self, signal):

        samples = signal.astype(np.int) # make room for squares
        # optimized sum of squares, i.e faster version of (samples**2).sum()
        instant_energy = np.dot(samples, samples) / float(0xffffffff) # normalize

        local_energy_average = self.local_energy.mean()
        local_energy_variance = self.local_energy.var()

        beat_sensibility = (-0.0025714 * local_energy_variance) + 1.15142857
        beat = instant_energy > beat_sensibility * local_energy_average

        self.local_energy[self.local_energy_index] = instant_energy
        self.local_energy_index -= 1
        if self.local_energy_index < 0:
            self.local_energy_index = len(self.local_energy) - 1

        return beat