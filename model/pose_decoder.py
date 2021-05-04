import torch
import torch.nn as nn
from model.audio_encoder import RNN


class res_linear_layer(nn.Module):
    def __init__(self, linear_hidden = 1024,time=1024):
        super(res_linear_layer,self).__init__()
        self.layer = nn.Sequential(        
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU()
        )
    def forward(self,input):
        output = self.layer(input) #51,1024
        return output


class pose_decoder(nn.Module):
    def __init__(self,batch,hidden_channel_num=64,input_c = 266,linear_hidden = 1024,encoder='gru'):
        super(pose_decoder,self).__init__()
        self.batch=batch
        self.encoder = encoder
        
        self.tmpsize = 128 if 'min' in encoder else 256
        if 'initp' in self.encoder:
            self.size = self.tmpsize+10+5
        else:
            self.size = self.tmpsize+10
        #self.relu = nn.ReLU()
        #self.decoder = nn.GRU(bidirectional=True,hidden_size=36, input_size=266,num_layers= 3, batch_first=True)
        #self.fc=nn.Linear(72,36)
        # TODO! change here!!!!!
        if 'lstm' in encoder:
            self.rnn_noise = nn.LSTM(10, 10, batch_first=True)
        else:
            self.rnn_noise = nn.GRU( 10, 10, batch_first=True)
            
        self.rnn_noise_squashing = nn.Tanh()
        # state size. hidden_channel_num*8 x 360 x 640

        if 'initp' in self.encoder:
            self.layeri = nn.Linear(36, 250)

        self.layer0 = nn.Linear(36, linear_hidden)
        self.layer1 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer2 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer3 = res_linear_layer(linear_hidden = linear_hidden)
        self.final_linear = nn.Linear(linear_hidden, self.tmpsize)

    def forward(self,input):
        #initpose : 18,2
        # input: -1, 50, 36
        # output: -1, 50, 128(or 256)
        input = input.view(-1, 36)
        output = self.layer0(input)
        output = self.layer1(output) + output
        output = self.layer2(output) + output
        output = self.layer3(output) + output
        output = self.final_linear(output)#,36

        output = output.view(self.batch,50,self.tmpsize)
            # output = output[:,1:,:]
            # print('output',output.size())
        #output = self.rnn_noise_squashing(output)
        return output