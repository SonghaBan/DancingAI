# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:34:29 2021

@author: songhama
"""
import numpy as np
import time
import json
from more_itertools.recipes import grouper
import pandas as pd
import glob

alpha_files = glob.glob('PoseEstimation/output_fast/*.json')
alpha_26 = glob.glob('output_halpe26/*.json')
open_files = glob.glob('output_openpose/*.json')

def analyze_models(files):
    wrong_list = []
    c_min_list = []
    c_mean_list = []
    missing_list = []
    total_f = 0
    for filename in files[:50]:
        print(filename)
        with open(filename,'r') as of:
            data = json.load(of)
            total_f += len(data)
        openp=False
        if 'output_openpose' in filename:
            openp=True
        missing, wrong, c_min, c_mean = analyze_data(data, openp)
        wrong_list.append(wrong)
        c_min_list.append(c_min)
        c_mean_list.append(c_mean)
        missing_list.append(missing)
    return missing_list, wrong_list, c_min_list, c_mean_list, total_f

def chk_openpose_zero(prev, pose):
#    a,b,c = prev.isin([0])[:19].all()
#    if a or b:
#        return False
#    a,b,c = pose.isin([0])[:19].all()
#    if a or b:
#        return False
    a,b,c = prev.isin([0])[19:].any()
    if a or b:
        return True
    a,b,c = pose.isin([0])[19:].any()
    if a or b:
        return True
    return False

def analyze_data(data, openp=False):
    wrong = 0
    prev = None
    c_min = 0
    c_mean = 0
    cur_idx = 0
    missing = 0
    if not openp:
        cur_idx = int(data[0]['image_id'].split('.')[0]) - 1
    for i,d in enumerate(data):
        if not openp:
            frame_idx = int(d['image_id'].split('.')[0])
#            print(frame_idx, cur_idx)
            if frame_idx == cur_idx: #same frame
                continue
            if (frame_idx - cur_idx) > 1:
                print('missing frame')
                missing += (frame_idx - cur_idx -1)
                cur_idx += (frame_idx - cur_idx -1)
                
            pose = pd.DataFrame(list(grouper(d['keypoints'], 3)))
        else:
            if len(d['people']) == 0: #missing frame
                missing += 1
                continue
            pose = pd.DataFrame(list(grouper(d['people'][0]["pose_keypoints_2d"], 3)))
        cur_idx += 1
        if i == 0:
            prev = pose
            continue
        diff = pose - prev
        #misdetection
        if diff.max()[:2].max() > 100 or diff.min()[:2].min() < -100:
            if openp:
                chkr = chk_openpose_zero(prev, pose)
                if not chkr:
                    wrong += 1
            else:
                wrong += 1
        prev = pose
        c_min = min(pose.min()[2], c_min)
        c_mean = c_mean + pose.mean()[2]
        
    c_mean = c_mean / len(data)
    return missing, wrong, c_min, c_mean

    
from PIL import Image
#r = make_frame(data[:1000])
#img = Image.fromarray(r) 
#img.show()

'''

missin3, wrong3, cmin3, ceman3 = analyze_models(open_files)
missing1, wrong1, cmin1, cmean1 = analyze_models(alpha_files)
missing2, wrong2, cmin2, ceman2 = analyze_models(alpha_26)

def showall(func):
    print('missing')
    print(func(missing1), func(missing2), func(missin3))
    print('incorrect')
    print(func(wrong1), func(wrong2), func(wrong3))
    print('cmin')
    print(func(cmin1), func(cmin2), func(cmin3))
    print('c mean')
    print(func(cmean1), func(ceman2), func(ceman3))

print("MEAN")
showall(np.mean)
print('================================')
print("MEDIAN")
showall(np.median)
print('================================')
print("MAX")
showall(max)
'''
a,b,c,d,e = analyze_models(alpha_files)

