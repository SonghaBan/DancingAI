#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
import cv2
import numpy as np
import time
import gizeh as gz
import json
from more_itertools.recipes import grouper, pairwise
import moviepy.editor as mpy
import pandas as pd

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}


l_pair_17 = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 11), (17, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16),(17,0)
]

colors17 = np.array([
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], 
    [0, 255, 170], [255,200,100],
    [0, 170, 255], [255,130,100], [0, 0, 255], [255,150,80], [0, 50, 255],
    [255,0,0]])/255

p_color_17 = np.array([(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)])/255  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
#p_color_17 = np.array([(255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 170), (170, 0, 255), (255, 0, 255), (85, 0, 255)])/255
line_color_17 = np.array([(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (191, 255, 77), (77, 255, 77), (77, 196, 255), (77, 135, 255),
              (255, 80, 120), (77, 222, 255), 
              (255, 0,80), (0, 80, 255), (255, 0,0), (0, 0, 255),(77, 255, 222)])/255

l_pair_26 = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 15), (21, 16), (23, 16), (22, 15), (15, 24), (16, 25),# Foot
        ]
p_color_26 = np.array([(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
           (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
           (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)])/255 # foot)

line_color_26 = np.array([(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
              (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)])/255

l_pair_openpose = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    ]

l_pair_openpose25 = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], 
    [8, 9], [9, 10], [10,11], [11,24], [11,22], [22,23],
    [8,12], [12, 13], [13,14], [14,21], [14,19], [19,20],
    [1, 0], 
    [0, 16], [18, 16], 
    [0, 15], [15, 17],
    ]
colors_openpose25 = np.array([
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [255,0,0], 
    [0, 255, 0], [0, 255, 85], [0, 255, 85], [0, 255, 85], [0, 255, 85], [0, 255, 85], 
    [0, 255, 170], [0, 255, 255],[0, 170, 255],[0, 170, 255],[0, 170, 255],[0, 170, 255],
    [0, 85, 255], 
#    [0, 0, 255], [85, 0, 255], 
#    [170, 0, 255],
    [255, 0, 255], [255, 0, 170], 
    [255, 0, 85], [255, 0, 0]])/255
#for plot usage
colors_openpose = np.array([
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]])/255



def build_graph(lines):
    graph = {}
    for line in lines:
        for n1, n2 in pairwise(line):
            if n1 > n2:
                n1, n2 = n2, n1
            graph.setdefault(n1, set()).add(n2)
    return graph


#BODY_25_GRAPH = build_graph(BODY_25_LINES)


def max_dim(doc, dim):
    return max((
        val
        for person in doc["people"]
        for numarr in person.values()
        for val in numarr[dim::3]
    ))
    
def make_video_openpose(data, save_video=True, output_filename="../test.mp4",fps=10):
    width = 640
    height = 360
    joints = 18
    print("make! fps:",fps, len(data))
    
    arr = True if isinstance(data, np.ndarray) else False
    
    result = []
    for frame in data:
#        print(cur_idx, frame['image_id'])
        
        surface = gz.Surface(width=width, height=height, bg_color=(1,1,1))
    
        if arr:
            pose = frame.tolist()
        else:
            n_group = int(len(frame['keypoints']) / joints)
            pose = list(grouper(frame["keypoints"], n_group))
        line_cnt = 0
        l_pair = l_pair_openpose
        line_color = colors_openpose
        
        for limb in l_pair:
            x1, y1 = pose[limb[0]]
            x2, y2 = pose[limb[1]]
            line = gz.polyline(points=[(x1,y1), (x2,y2)], stroke_width = 6, stroke=line_color[line_cnt])
#            print('line', line_cnt)
            line_cnt += 1
            line.draw(surface)
        
        for idx in range(len(pose)):
#            print(idx)
            x1, y1 = pose[idx]
            joint = gz.circle(4, xy=[x1,y1], fill=(0,0,0))
            joint.draw(surface)

        result.append(surface.get_npimage())
    
    if save_video:
        clip = mpy.ImageSequenceClip(result, fps=fps)
        clip.write_videofile(output_filename, fps=fps, codec='mpeg4')
        
def visualize_frame(frame, joints=17, label=0):
    
    surface = gz.Surface(width=640, height=360, bg_color=(1,1,1))
    
    if joints == 25:
        frame = frame['people'][0]
        n_group = int(len(frame['pose_keypoints_2d']) / joints)
        pose = list(grouper(frame["pose_keypoints_2d"], n_group))
    else:
        n_group = int(len(frame['keypoints']) / joints)
        pose = list(grouper(frame["keypoints"], n_group))
#    print(pose)
    
    if joints==17:
        l_pair = l_pair_17
        line_color = colors17
    elif joints==25:
        l_pair = l_pair_openpose25
        line_color = colors_openpose25
    elif joints == 26:
        l_pair = l_pair_26
        line_color = line_color_26
        
    line_cnt = 0
    
    for limb in l_pair:
#        print(limb)
        if limb[0] == joints and (joints == 17 or joints == 26):
            x1, y1, _ = (np.array(pose[5]) + np.array(pose[6])) / 2 #neck
        else:
            x1, y1, _ = pose[limb[0]]
        x2, y2, _ = pose[limb[1]]
        line = gz.polyline(points=[(x1,y1), (x2,y2)], stroke_width = 6, stroke=line_color[line_cnt])
#            print('line', line_cnt)
        line_cnt += 1
        line.draw(surface)
    
    for idx in range(len(pose)):
#            print(idx)
        x1, y1,_ = pose[idx]
        joint = gz.circle(4, xy=[x1,y1], fill=(0,0,0))
        joint.draw(surface)

    show_img(surface.get_npimage(), label)


def make_frame_clean(data, save_video=True, joints=17,output_filename="../test.mp4",fps=30):
    width = 640
    height = 360
    print("make! fps:",fps, len(data))
    
    arr = True if isinstance(data, np.ndarray) else False
    
    result = []
#    BODY_17_GRAPH = build_graph(BODY_17_LINES)
    cur_idx = -1
    for frame in data:
#        print(cur_idx, frame['image_id'])
        
        surface = gz.Surface(width=width, height=height, bg_color=(1,1,1))
    
        cur_idx += 1
        
        if arr:
            pose = frame.tolist()
        else:
            n_group = int(len(frame['keypoints']) / joints)
            pose = list(grouper(frame["keypoints"], n_group))
        line_cnt = 0
        if joints==17 or joints == 18:
            l_pair = l_pair_17
            line_color = line_color_17
            p_color = p_color_17
        elif joints == 26:
            l_pair = l_pair_26
            line_color = line_color_26
            p_color = p_color_26
        
        for limb in l_pair:
            
            if limb[0] == joints and joints == 17:
                x1, y1 = (np.array(pose[5]) + np.array(pose[6])) / 2 #neck
            else:
                x1, y1 = pose[limb[0]]
            x2, y2 = pose[limb[1]]
            line = gz.polyline(points=[(x1,y1), (x2,y2)], stroke_width = 6, stroke=line_color[line_cnt])
#            print('line', line_cnt)
            line_cnt += 1
            line.draw(surface)
        
        for idx in range(len(pose)):
#            print(idx)
            x1, y1 = pose[idx]
            joint = gz.circle(4, xy=[x1,y1], fill=(0,0,0))
            joint.draw(surface)

        result.append(surface.get_npimage())
    
    if save_video:
        clip = mpy.ImageSequenceClip(result, fps=fps)
        clip.write_videofile(output_filename, fps=fps, codec='mpeg4')
# =============================================================================
#     TODO: multi-person
# =============================================================================
def make_frame(data, save_video=True, joints=17,output_filename="../test.mp4"):
    width = 720
    height = 540
    
    
    result = []
#    BODY_17_GRAPH = build_graph(BODY_17_LINES)
    cur_idx = -1
    for frame in data:
        frame_idx = int(frame['image_id'].split('.')[0])
#        print(cur_idx, frame['image_id'])
        if frame_idx == cur_idx: #same frame
#            print('cont', cur_idx)
            continue
        
        surface = gz.Surface(width=width, height=height, bg_color=(1,1,1))
        
        if (frame_idx - cur_idx) > 1: #missing frame
#            print('missing frame')
            time.sleep(0.5)
            #missing += (frame_idx - cur_idx -1)
            cur_idx += (frame_idx - cur_idx)
            result.append(surface.get_npimage())
            continue
        cur_idx += 1
        
        n_group = int(len(frame['keypoints']) / joints)
        pose = list(grouper(frame["keypoints"], n_group))
        line_cnt = 0
        if joints==17 or joints == 18:
            l_pair = l_pair_17
            line_color = line_color_17
            p_color = p_color_17
        elif joints == 26:
            l_pair = l_pair_26
            line_color = line_color_26
            p_color = p_color_26
        
        for limb in l_pair:
            
            if limb[0] == joints and joints == 17:
                x1, y1, c1 = (np.array(pose[5]) + np.array(pose[6])) / 2 #neck
            else:
                x1, y1, c1 = pose[limb[0]]
            x2, y2, c2 = pose[limb[1]]
            line = gz.polyline(points=[(x1,y1), (x2,y2)], stroke_width = 3, stroke=line_color[line_cnt])
#            print('line', line_cnt)
            line_cnt += 1
            line.draw(surface)
        
        for idx in range(len(pose)):
#            print(idx)
            x1, y1, c1 = pose[idx]
            joint = gz.circle(3, xy=[x1,y1], fill=(0,0,0))
            joint.draw(surface)
#            
#            for g_idx in BODY_17_GRAPH.get(idx, set()):
#                
#                if g_idx == 17:    
#                    x2, y2, c2 = (np.array(pose[5]) + np.array(pose[6])) / 2 #neck
#                    joint = gz.circle(3, xy=[x2,y2], fill=p_color[g_idx])
#                    joint.draw(surface)
#                else:
#                    x2, y2, c2 = pose[g_idx]
#                c = min(c1, c2)
#                
##                if c == 0:
##                    continue
#                line = gz.polyline(
#                    points=[(x1, y1), (x2, y2)], stroke_width=5 * c,
#                    stroke=line_color[line_cnt]
#                )
#                print('color ',line_cnt, line_color[line_cnt])
#                line_cnt += 1
#                line.draw(surface)
        result.append(surface.get_npimage())
    
    if save_video:
        clip = mpy.ImageSequenceClip(result, fps=30)
        clip.write_videofile(output_filename, fps=30, codec='mpeg4')
        #clip.ipython_display(fps=30)
#    return result




def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval

#filename = 'output_fast/alphapose-results-fast-29_Trim.json'
#filename = 'output_halpe26/alphapose-results-halpe26-29_Trim.json'

#with open(filename,'r') as of:
#    data = json.load(of)
    
from PIL import Image
#r = make_frame(data[:1200],joints=17)
#r = make_frame(data,joints=26, output_filename="../test29_2.mp4")



def show_img(arr,idx=0):
    img = Image.fromarray(arr)
#    img.show()
    img.save(f'testframes/clean/{idx}.jpg')

