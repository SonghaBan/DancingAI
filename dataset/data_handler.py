import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import bisect
import json

def load_jsonfile(filename):
    with open(filename, 'r') as of:
        data = json.load(of)
    return data

class DanceDataset(torch.utils.data.Dataset):
    def __init__(self, opt, train=True):
        file_location=opt.data
        pose_dict=load_jsonfile(file_location)
        
        length=0
        for k,v in pose_dict.items():
            length += len(v['pose'])

        self.length=length
        print(self.length)
        
        target=torch.FloatTensor(length,50,1600).zero_() #music
        label=torch.FloatTensor(length,50,18,2).zero_() #dance
        index=0
        
        keys=sorted(pose_dict.keys())
        for key in keys:
            #if int(key) != 2:
            #    continue
            pose_sequences = np.array(pose_dict[key]['pose'])
            audio_sequences = np.array(pose_dict[key]['music'])

            for i, temp_pose in enumerate(pose_sequences):
                temp_pose[:,:,0]=(temp_pose[:,:,0]/320)-1
                temp_pose[:,:,1]=(temp_pose[:,:,1]/180)-1

                label[index] = torch.from_numpy(temp_pose)
                
                temp_audio = np.array(audio_sequences[i])
                d = torch.from_numpy(temp_audio).type(torch.LongTensor)
                target[index] = d.view(50, 1600)

                index += 1
        
        self.audio=target
        self.label=label
        
        self._length = 80000
        
        self.train = train
        print("load the json file to dictionary (5s raw data)" )
        # assign every *test_stride*th item to the test set


    def __getitem__(self, idx):
        #print("idx:",idx)
        one_hot=self.audio[idx]
        target=self.label[idx]          
        return one_hot, target

    def __len__(self):
        return self.length


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj) 
    
def save_to_json(dic,target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)  
    file = open(target_dir, 'w')  
    json.dump(dumped, file)
    file.close()
    
def read_from_json(target_dir):
    f = open(target_dir,'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data 