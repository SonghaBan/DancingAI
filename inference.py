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
import json
from utils import make_pose_video, attach_audio, get_filelabel

join = os.path.join
cur_d = os.path.dirname(__file__)

from matplotlib import pyplot as plt
#import cv2
from dataset.output_helper import save_2_batch_images
import argparse
import scipy.io.wavfile as wav

parser = argparse.ArgumentParser()

parser.add_argument(
        "--input",
        default="2,4,7,56,68,116,125,117,13,41",
        metavar="FILE",
        help="path to pth file",
        type=str
    )

parser.add_argument(
        "--model",
        default=join(cur_d, "log/lstm/generator_0400.pth"),
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

parser.add_argument("--count", type=int, default=10)#100
parser.add_argument(
        "--output",
        default=join(cur_d,"../output"),
        metavar="FILE",
        help="path to output",
        type=str
    )
parser.add_argument(
        "--encoder",
        default=join(cur_d,"gru"),
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

#audio_path=output_dir + "/audio"
#try:
#    os.makedirs(audio_path)
#except OSError:
#    pass

Tensor = torch.cuda.FloatTensor
generator = Generator(1, encoder=args.encoder)
generator.eval()
#generator.load_state_dict(torch.load(file_path, map_location=device))
generator.load_state_dict(torch.load(file_path))
generator.cuda()

def generate_dance(audio, initp):
    fake = generator(audio, initp)
    fake = fake.contiguous().cpu().detach().numpy()#1,50,36 
    fake = fake.reshape([50,36])
    return fake

def save_output(data, filename):
    with open(filename, 'w') as of:
        json.dump(data, of)

def test(audiofile="C:/Users/songhama/Documents/_School/Spring2021/Thesis/data/audio/7_Trim.wav"):
    data=AudioLoader(audiofile)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=False)

    result = []
    outaudio = []
    prevp = None
    for i, x in enumerate(dataloader):
        if i > counter and counter:
            break
        audio_out = x.view(-1) #80000
        outaudio.extend(audio_out)

        audio = Variable(x.type(Tensor).transpose(1,0)) #50,1,1600
        
        if i == 0:
            initp = torch.zeros(18,2)
        else:
            initp = torch.from_numpy(prevp)
        initp = Variable(initp.type(Tensor))


        fake = generate_dance(audio, initp)

        #wav.write("tmp.wav", 16000, scaled)
        fake_coors = fake.reshape([-1,18,2])
        prevp = np.copy(fake_coors[-1])
        
        fake_coors[:,:,0] = (fake_coors[:,:,0]+1) * 320
        fake_coors[:,:,1] = (fake_coors[:,:,1]+1 ) * 180
        fake_coors = fake_coors.astype(int).tolist()

        save_output(fake_coors, join(output_dir, f"{i}.json"))

        result.extend(fake_coors)
        

    outaudio = np.int16(outaudio)
    wav.write("tmp.wav", 16000, outaudio)

    outfile = join(output_dir, f"result_{get_filelabel(audiofile)}.mp4")
    make_pose_video(result, output_filename=outfile)
    attach_audio(outfile, "tmp.wav")
    os.remove("tmp.wav")
            
if __name__ == '__main__':
    input_files = args.input.split(',')
    for input_file in input_files:
        filepath = join(cur_d, f"C:/Users/songhama/Documents/_School/Spring2021/Thesis/data/audio/{input_file}_Trim.wav")
        test(filepath)