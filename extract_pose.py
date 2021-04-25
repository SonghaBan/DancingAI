# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 03:10:46 2021

@author: songhama
"""

import subprocess
import glob
import time
import os
import json
import moviepy.editor as mpy
'''
compare three pose estimatino methods
'''

cmd_fast = "python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --video ../../{} --outdir ../output_fast/"

cmd_26 = "python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video ../../{} --outdir ../output_halpe26/"

cmd_openpose = "bin\OpenPoseDemo.exe --video ../../{} --disable_blending --write_json ../output_tmp/"


def get_alphapose_17(filename, label):
    subprocess.call(cmd_fast.format(filename))
    try:
        os.rename('../output_fast/alphapose-results.json', '../output_fast/alphapose-results-fast-{}.json'.format(label))
    except FileExistsError:
        os.remove('../output_fast/alphapose-results-fast-{}.json'.format(label))
        os.rename('../output_fast/alphapose-results.json', '../output_fast/alphapose-results-fast-{}.json'.format(label))
        
def get_alphapose_26(filename, label):
    subprocess.call(cmd_26.format(filename))
    os.rename('../output_halpe26/alphapose-results.json', '../output_halpe26/alphapose-results-halpe26-{}.json'.format(label))

def get_openpose(filename, label):
    subprocess.call(cmd_openpose.format(filename))
    tmpfiles = glob.glob('../output_tmp/*.json')
    data = []
    for tf in tmpfiles:
        with open(tf,'r') as of:
            td = json.load(open(tf,'r'))
            data.append(td)
        os.remove(tf)
    with open('../output_openpose/{}.json'.format(label), 'w') as of:
        json.dump(data, of)


def rename_all():
    files = glob.glob("data/final_videos/me/*.mp4")
    for f in files:
        os.rename(f, f.replace('.mp4','_Trim.mp4'))
        
def update_fps():
    files = glob.glob("data/final_videos/me/*.mp4")
    for f in files:
        print(f)
        video = mpy.VideoFileClip(f)
        fps = video.fps
        if fps < 29.9:
            print('!!!',fps, f)
            video.write_videofile('tmp.mp4', fps=30)
            video.close()
            
            os.remove(f)
            os.rename('tmp.mp4', f)
            
        
        else:
            video.close()
            
        os.chdir('PoseEstimation/AlphaPose')
        label = os.path.basename(f).split('.')[0]
        get_alphapose_17(f, label)
        os.chdir('../../')
        time.sleep(0.1)
        
        
def main():
    files = glob.glob("data/final_videos/me/*.mp4")
    for f in files:
        label = os.path.basename(f).split('.')[0]
        os.chdir('PoseEstimation/AlphaPose')
        print(f)
        get_alphapose_17(f, label)
        time.sleep(1)
        
#        get_alphapose_26(f, label)
#        time.sleep(1)
#        
#        os.chdir('../openpose')
#        get_openpose(f, label)
        
        print(f, "done")
#        time.sleep(3)
        os.chdir('../../')
    
if __name__=="__main__":
#    main()
    update_fps()