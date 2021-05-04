import torch
import torch.nn as nn
from model.audio_encoder import RNN

#model for generator decoder

    
# use standard conv-relu-pool approach
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
        

# class conv_layer(nn.Module):
#     def __init__(self, in_channels=266, out_channels=512, kernel_size=(5,5)):
#         super(conv_layer,self).__init__()
#         self.layer = nn.Sequential(
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size
#             ),
#             nn.BatchNorm2d(out_channels),
#         )
#     def forward(self, input):
#         return self.layer(input)

class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(conv_layer,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, input):
        return self.layer(input)
        
class hr_pose_generator(nn.Module):
    def __init__(self,batch,hidden_channel_num=64,input_c = 266,linear_hidden = 1024,encoder='gru'):
        super(hr_pose_generator,self).__init__()
        self.batch=batch
        self.encoder = encoder
        self.tmpsize = 128 if 'min' in encoder else 256
        if 'initp2' in self.encoder:
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

        if 'initp2' in self.encoder:
            self.layeri = nn.Linear(36, 250)
        elif 'initp' in self.encoder:
            self.layeri = nn.Linear(36, self.size)
        self.layer0 = nn.Linear(self.size,linear_hidden)
        #self.relu = nn.ReLU()
        #self.bn=nn.BatchNorm1d(50)
        self.layer1 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer2 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer3 = res_linear_layer(linear_hidden = linear_hidden)
        self.dropout =  nn.Dropout(p=0.5)
        self.final_linear = nn.Linear(linear_hidden,36)
        
    def forward(self,input, initpose):
        #initpose : 18,2
        noise = torch.FloatTensor(self.batch, 50, 10).normal_(0, 0.33).cuda()
        aux, h = self.rnn_noise(noise)
        aux = self.rnn_noise_squashing(aux) #1 50 10

        # print('initp',initpose.size())        

        input = torch.cat([input, aux], 2)
        if 'initp2' in self.encoder:
            initp = self.layeri(initpose.view(-1,36)) #-1, 250
            initp = initp.view(-1,50,5)
            # print(input.size(), initp.size())
            input = torch.cat([input, initp], 2) #-1,50,231
        elif 'initp' in self.encoder:
            initp = self.layeri(initpose.view(-1,36))
            initp = initp.view(-1, 1, self.size)
            input = torch.cat([input, initp], 1) #-1,51, 266
        #print(input.shape)
        #input=input.squeeze().view(1,50,266)
        #input=input.squeeze().view(50,266)
            
        input = input.view(-1,self.size) #50 266
        output = self.layer0(input)
            #output = self.relu(output)
        #output = self.bn(output)
        output = self.layer1(output) + output
        output = self.layer2(output) + output
        output = self.layer3(output) + output
        output = self.dropout(output)

        output = self.final_linear(output)#,36
        
        if 'initp2' not in self.encoder and 'initp' in self.encoder:
            output = output.view(self.batch,51,36)
            output = output[:,:-1,:]
        else:
            output = output.view(self.batch,50,36)
            # output = output[:,1:,:]
            # print('output',output.size())
        output = self.rnn_noise_squashing(output)
        return output
    
        

class Generator(nn.Module):
    def __init__(self,batch, encoder='gru'):
        super(Generator,self).__init__()    
        self.audio_encoder=RNN(batch, encoder)
        self.pose_generator=hr_pose_generator(batch, encoder=encoder)
        self.batch=batch

    def forward(self,input,initp=None):
        output=self.audio_encoder(input)#input 50,1,1600 // output 1,50,256
        output=self.pose_generator(output,initp)#1，50，36
        return output#1,50,36

    def extract_music_features(self, audio):
        return self.audio_encoder(audio)
        