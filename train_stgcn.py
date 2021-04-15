import torch
import torch.nn as nn
import data
import numpy as np
from net.st_gcn_perceptual import Model

epochs = 800
graph_args={"layout": 'openpose',"strategy": 'spatial'}
stgcn = Model(2, 16, graph_args, edge_importance_weighting=True).cuda()
with open('dataset/dance_music_paired.json', 'r') as f:
    data = json.load(f)


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

train_x = []
for k,d in data.items():
    pose = d['pose']
    #bsz,time,17,2
    train_x.extend(pose)
    for pose_5s in pose:
        train_x.append(pose_5s)

th = int(len(train_x) * 0.8) + 1
train_x = np.array(train_x[:th])
val_x = np.array(train_x[th:])

loss = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimiaer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs+1):
    l_sum, n = 0.0, 0
    model.train()
    for x in train_x:
        y = stgcn(x)
        bsz,time,feature = y.size()
        y = y.view(bsz, time, 17, 2)
        l = loss(y, x)
        l.backward()
        optimizer.step()
        l_sum += l.item() * x.shape[0]
        n += x.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_x)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)



'''
weights = [20.0 ,5.0 ,1.0 ,1.0 ,1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

class GCNLoss(nn.Module):
    def __init__(self,opt):
        super(GCNLoss, self).__init__()
        dict_path=opt.pretrain_GCN
        graph_args={"layout": 'openpose',"strategy": 'spatial'}
        self.gcn = Model(2,16,graph_args,edge_importance_weighting=True).cuda()
        self.gcn.load_state_dict(torch.load(dict_path))
        self.gcn.eval()
        self.criterion = nn.L1Loss()
        self.weights = [20.0 ,5.0 ,1.0 ,1.0 ,1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  #10 output      

    def forward(self, x, y):              
        x_gcn, y_gcn = self.gcn.extract_feature(x), self.gcn.extract_feature(y)
        loss = 0
        for i in range(len(x_gcn)):
            loss_state = self.weights[i] * self.criterion(x_gcn[i], y_gcn[i].detach())  
            #print("VGG_loss "+ str(i),loss_state.item())
            loss += loss_state       
        return loss
'''

