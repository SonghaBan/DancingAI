# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:04:21 2021

@author: songhama
"""
import json
filename = 'lisa_revised_pose_pairs.json'

with open(filename,'r') as f:
    data = json.loads(json.load(f))
    
data.keys()

newdata = {'000':data['000'],'001':data['001']}

d = data['001']
#d.keys ['010', '020', '030', '040', '050', '060', '070', '080', '090', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190'

with open('hana.json','w') as f:
    json.dump(json.dumps(d), f)
    
d10 = d['010']
#d10.keys ['audio_sequence', 'joint_coors']

joints = d10['joint_coors'] #(100,18,2) 18joints 100 frames
audio = d10['audio_sequence'] #a list of integers (160000,) 100 frames, 1600 per frame. 

keys = ['047','049']


import numpy as np

length = 0

for key in keys:
    sub_keys=sorted(data[str(key)].keys())
    print('subkeys',sub_keys)
    for sub_key in sub_keys:
        temp_pose=np.array(data[str(key)][str(sub_key)]["joint_coors"])
        if(temp_pose.shape==(100,)):
            print("boy"+key+" "+sub_key+" is wrong")
            continue
        length+=1
print(length)