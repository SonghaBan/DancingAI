import sys
import torch
import torch.nn as nn
import numpy as np
from model.pose_generator_norm import Generator
from dataset.data_handler import DanceDataset, AudioLoader
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import numpy as np
import math
import itertools
import time
import datetime

join = os.path.join
cur_d = os.path.dirname(__file__)

from matplotlib import pyplot as plt
#import cv2
from dataset.output_helper import save_2_batch_images
import argparse
import scipy.io.wavfile as wav

parser = argparse.ArgumentParser()

parser.add_argument(
        "--model",
        default=join(cur_d, "log/test/generator_0400.pth"),
        metavar="FILE",
        help="path to pth file",
        type=str
    )

parser.add_argument(
        "--data",
        default=join(cur_d,"dataset/dance_music_paired.json"),
        metavar="FILE",
        help="path to pth file",
        type=str
    )

parser.add_argument("--count", type=int, default=100)#100
parser.add_argument(
        "--output",
        default=join(cur_d,"output"),
        metavar="FILE",
        help="path to output",
        type=str
    )
args = parser.parse_args()

file_path=args.model
counter=args.count
print(file_path)
output_dir=args.output
try:
    os.makedirs(output_dir)
except OSError:
    pass

audio_path=output_dir + "/audio"
try:
    os.makedirs(audio_path)
except OSError:
    pass

Tensor = torch.cuda.FloatTensor
generator = Generator(1)
generator.eval()
#generator.load_state_dict(torch.load(file_path, map_location=device))
generator.load_state_dict(torch.load(file_path))
generator.cuda()

def generate_dance(audio):
    fake = generator(audio)
    fake = fake.contiguous().cpu().detach().numpy()#1,50,36 
    fake = fake.reshape([50,36])
    return fake


audiofile = "C:/Users/songhama/Documents/_School/Spring2021/Thesis/data/audio/7.wav"
data=AudioLoader()
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=False)
criterion_pixelwise = torch.nn.L1Loss()
count = 0
total_loss=0.0

for i, x in enumerate(dataloader):
    audio_out = x.view(-1) #80000
    scaled = np.int16(audio_out)

    audio = Variable(x.type(Tensor).transpose(1,0)) #50,1,1600
    
    fake = generate_dance(audio)

    wav.write("tmp.wav", 16000, scaled)
    fake_coors = fake.reshape([-1,18,2])
    
    fake_coors[:,:,0] = (fake_coors[:,:,0]+1) * 320
    fake_coors[:,:,1] = (fake_coors[:,:,1]+1 ) * 180
    fake_coors = fake_coors.astype(int)


    if(count <= counter):
        write(output_dir+"/audio/{}.wav".format(i),16000,scaled)
        fake_coors = fake
        fake_coors = fake_coors.reshape([-1,18,2])
        
        fake_coors[:,:,0] = (fake_coors[:,:,0]+1) * 320
        fake_coors[:,:,1] = (fake_coors[:,:,1]+1 ) * 180
        fake_coors = fake_coors.astype(int)
        
        save_2_batch_images(real_coors,fake_coors,batch_num=count,save_dir_start=output_dir)
    count += 1

