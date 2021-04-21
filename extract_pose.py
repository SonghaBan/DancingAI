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

'''
compare three pose estimatino methods
'''

cmd_fast = "python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --video ../../{} --outdir ../output_fast/"

cmd_26 = "python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video ../../{} --outdir ../output_halpe26/"

cmd_openpose = "bin\OpenPoseDemo.exe --video ../../{} --disable_blending --write_json ../output_tmp/"


files = glob.glob("data/final_videos/*.mp4")

for f in files:
    if f == 'data/final_videos\\10_Trim.mp4' or f == 'data/final_videos\\11_Trim.mp4':
        continue
    label = os.path.basename(f).split('.')[0]
    os.chdir('PoseEstimation/AlphaPose')
    print(f)
    subprocess.call(cmd_fast.format(f))
    os.rename('../output_fast/alphapose-results.json', '../output_fast/alphapose-results-fast-{}.json'.format(label))
    time.sleep(10)
    subprocess.call(cmd_26.format(f))
    os.rename('../output_halpe26/alphapose-results.json', '../output_halpe26/alphapose-results-halpe26-{}.json'.format(label))
    time.sleep(10)
    
    os.chdir('../openpose')
    subprocess.call(cmd_openpose.format(f))
    tmpfiles = glob.glob('../output_tmp/*.json')
    data = []
    for tf in tmpfiles:
        with open(tf,'r') as of:
            td = json.load(open(tf,'r'))
            data.append(td)
        os.remove(tf)
    with open('../output_openpose/{}.json'.format(label), 'w') as of:
        json.dump(data, of)
    print(f, "done")
    time.sleep(30)
    os.chdir('../../')
    