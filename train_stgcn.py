import torch
import torch.nn as nn
import numpy as np
from net.st_gcn_perceptual import Model
import json
from dataset.data_handler import DanceDataset
from config import get_arguments


epochs = 800
graph_args={"layout": 'openpose',"strategy": 'spatial'}
stgcn = Model(2, 16, graph_args, edge_importance_weighting=True).cuda()
# with open('dataset/dance_music_paired.json', 'r') as f:
#     data = json.load(f)


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

# Xs = []
# for k,d in data.items():
#     pose = d['pose']
#     #bsz,time,17,2
#     Xs.append(np.array(pose))
    #for pose_5s in pose:
    #    train_x.append(pose_5s)

# th = int(len(Xs) * 0.8) + 1
# train_x = np.array(Xs[:th])
# val_x = np.array(Xs[th:])


parser = get_arguments()
opt = parser.parse_args()
data=DanceDataset(opt)
dataloader = torch.utils.data.DataLoader(data,
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     pin_memory=False,
                                     drop_last=True
                                    )

save_path = "log/stgcn.pt"

loss = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.RMSprop(stgcn.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs+1):
    l_sum, n = 0.0, 0
    stgcn.train()
    for (x,target)in dataloader:
        bsz, time, _, _ = target.size()
        y = stgcn(target.view(bsz, time, 34).cuda())
        print('ysize', y.size())
        print(y)
        bsz,time,feature = y.size()
        y = y.view(bsz, time, 18, 2)
        l = loss(y, target)
        l.backward()
        optimizer.step()
        l_sum += l.item() * target.shape[0]
        n += target.shape[0]
    scheduler.step()
    # val_loss = evaluate_model(stgcn, loss, val_x)
    # if val_loss < min_val_loss:
    #     min_val_loss = val_loss
    if epoch % 50 == 0:
        torch.save(stgcn.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n)



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

