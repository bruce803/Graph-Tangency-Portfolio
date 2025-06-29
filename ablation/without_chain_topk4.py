import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import pickle
from datetime import *
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader

import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.nn import GATConv, ChebConv, GCNConv, GeneralConv, SAGEConv
from sklearn.preprocessing import normalize, LabelEncoder
from torch_geometric import graphgym
from torch_geometric.graphgym import loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

seed_value = 42
device = torch.device("cuda")
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed_value)

relation = pd.read_csv(r'dataset/edge_data.csv')
relation.drop(relation.columns[0], axis=1, inplace=True)
sliced_data = torch.load("dataset/sliced_data.pth")
sliced_r = torch.load("dataset/sliced_r.pth")

lr = 1e-6
lo = 10


class get_dataset(Dataset):

    # def __init__(self, data, target):
    #     data = data.to(device)
    #     mean=torch.mean(data,dim=1)
    #     std=torch.std(data,dim=1)
    #     self.data = (data - mean) / std
    #     self.target = target.to(device)
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __getitem__(self, index):
        # x = self.data[index,:,:,:]
        # y = self.target[index,:,:]
        return self.data[index],self.target[index]

    def __len__(self):
        return len(self.data)


def dataset_split(data, target, ratio1,ratio2):
    # data是tensor
    spilt_index_1 = int(ratio1*data.shape[0])
    spilt_index_2 = int(ratio2*data.shape[0])
    train_data = data[:spilt_index_1,:,:,:]
    train_target = target[:spilt_index_1,:,:]
    valid_data = data[spilt_index_1:spilt_index_2,:,:,:]
    valid_target = target[spilt_index_1:spilt_index_2,:,:]
    test_data = data[spilt_index_2:,:,:,:]
    test_target = target[spilt_index_2:,:,:]
    return train_data, train_target, valid_data, valid_target, test_data, test_target


train_data, train_target, valid_data, valid_target, test_data, test_target = dataset_split(sliced_data,sliced_r,0.7,0.9)
train_dataset = get_dataset(train_data, train_target)
valid_dataset = get_dataset(valid_data, valid_target)
test_dataset = get_dataset(test_data, test_target)

def get_data_loader(batch_size, trainFolder, validFolder, testFolder):

    train_loader = torch.utils.data.DataLoader(trainFolder,
                                               batch_size=batch_size,
                                               shuffle=False, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(validFolder,
                                               batch_size=1,
                                               shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testFolder,
                                            batch_size=1,
                                            shuffle=False, drop_last=True)

    
    return train_loader, valid_loader,test_loader


batch_size=1
train_loader, valid_loader,test_loader=get_data_loader(batch_size, train_dataset, valid_dataset, test_dataset)
class AGNN(torch.nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        self.fc_0_1 = nn.Linear(161, 100)
        self.fc_0_2 = nn.Linear(100, 50)
        self.fc_1_1 = nn.Linear(50, 64)
        self.fc_1_2 = nn.Linear(64, 32)
        self.fc_1_3 = nn.Linear(32, 1)
        self.fc_2 = nn.Linear(30, 50, bias=False)
        self.bn1=nn.BatchNorm2d(2912)
        self.layer40to1=nn.Linear(40,1)
        self.lstm = nn.LSTM(input_size=50, hidden_size=30, num_layers=1, batch_first=False)

        # 条件beta
        self.batch1 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.batch2 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.batch3 = nn.BatchNorm2d(1, eps=1e-5, affine=True)
        self.beta_layer1 = nn.Linear(50, 100)
        self.beta_layer2 = nn.Linear(100, 60)
        self.beta_layer3 = nn.Linear(60, 20)
        self.beta_layer4 = nn.Linear(20, 1)

        #深度因子
        self.df1 = nn.Linear(2912,100)
        self.df2 = nn.Linear(100,50)
        self.df3 = nn.Linear(50,1)

    def w_softmax(self, w):
        w = torch.clamp(w, min=-3, max=3)
        w_1 = -50 * torch.exp(-8 * w)  # [B,N,1]
        w_2 = -50 * torch.exp(8 * w)
        h_1 = F.softmax(w_1, dim=1)
        h_2 = F.softmax(w_2, dim=1)
        w_new = h_1 - h_2
        return w_new

    
    

    def forward(self, data, r, h, c):

        x, edge_index = data.x, data.edge_index
        # x[batch,T,stock,feature]
        x = torch.transpose(x, 1, 2)
        x=self.bn1(x)#在t的维度上做batchnorm
        x = torch.transpose(x, 1, 2)
        
        # 首先输入邻接矩阵A和特征数据X，得到嵌入矩阵H
        x = self.fc_0_1(x)# x[batch,T,stock,feature]
        x = torch.tanh(x)
        x = self.fc_0_2(x)# x[batch,T,stock,feature]
        x = torch.tanh(x)
        x = torch.transpose(x, 1, 3)
        x = self.layer40to1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.tanh(x)# x[batch,1,stock,feature]
        x = x.squeeze(1)# x[batch,stock,feature]
        _, (H,_) = self.lstm(x,(h,c))# H[batch(1),stock(2912),feature(30)]
        

        # 使用嵌入矩阵H估计均值向量μ
        mu = self.fc_1_1(x)  # H[batch,stock,feature]
        mu = torch.tanh(mu)
        mu = self.fc_1_2(mu)  # H[batch,stock,feature]
        mu = torch.tanh(mu)
        mu = self.fc_1_3(mu)  # H[batch,stock,feature] mu[batch,stock,1]
        mu = torch.tanh(mu)

        # 使用嵌入矩阵H计算得到HWH^T来估计sigma的逆
        hw = self.fc_2(H)  # x[batch,stock,feature]
        # temp = torch.matmul(temp, torch.transpose(temp, 1, 2))
        temp = torch.matmul(hw, torch.transpose(hw, 1, 2))
        temp = torch.layer_norm(temp, normalized_shape=temp.shape[-1:])
        temp = torch.tanh(temp)  # temp[batch,stock,stock]

        
        # 使用H计算beta
        beta = torch.tanh(self.batch1(self.beta_layer1(x.unsqueeze(1))))
        beta = torch.tanh(self.batch2(self.beta_layer2(beta)))
        beta = torch.tanh(self.batch3(self.beta_layer3(beta)))
        beta = self.beta_layer4(beta).squeeze(1)

        # spar = torch.mul(torch.matmul(hw,torch.transpose(hw,-1,-2)),(ones-eye))
        # R_beta = 0.001*(torch.norm(spar, p='fro')**2)

        # 根据论文中的公式，w等于协方差逆矩阵乘上均值向量,再经过softmax()非线性激活函数来近似经典资产定价排序操作
        w = torch.matmul(temp, mu)  # w[batch,stock,1]
        # W = self.df1(W.float())
        # W = self.df2(W)
        # W = self.df3(W)
        # w = torch.tanh(W)

        # 对时间维度预测

        w = self.w_softmax(w)  # w[batch,stock,1]

        # 相乘得到深度因子
        R = torch.matmul(r, w)
        R = R.squeeze(-1)  # R[batch,T]

        pr = R.unsqueeze(2)
        pre_r = torch.matmul(beta, torch.transpose(pr, -1, -2))
        # 重新调整结果的形状，去除最后一个维度的大小为 1
        pre_r = torch.transpose(pre_r, 1, 2)
        return R, w, x, mu, temp,  pre_r


model_agnn = AGNN()
print(model_agnn)

model_agnn.to(device)
optimizer = torch.optim.Adam(model_agnn.parameters(), lr=lr)
criterion = nn.MSELoss().to(device)

def l0_loss(input, target):
    loss = torch.sum(torch.abs(input - target) > 0).float()
    return 0.01*loss

def loss_1(R_beta):
    loss = 0.001*R_beta
    return loss

col_relation = ['primary_code', 'related_code']
re = np.array(relation[col_relation])
x_l = 40
r_l = 10

num_epoch = 10


train_LOSS = []
train_LOSS_ORDER = []

valid_LOSS = []
valid_LOSS_ORDER = []

for epoch in range(num_epoch):
    print('Epoch：{}'.format(epoch+1))
    train_losses = []
    train_loss_orders = []
    # 训练过程
    model_agnn.train()
    for i,(x,r) in enumerate(train_loader):
        x = x.to(device)
        r = r.to(device)
        edge_index = torch.tensor(re, dtype=torch.long)
        edge_index=edge_index.to(device)
        Gdata = Data(x=x,edge_index=edge_index.t().contiguous())
        h = torch.zeros(1, 2912, 30).to(device)
        c = torch.zeros(1, 2912, 30).to(device)
        R, w, x, mu, temp,  pre_r = model_agnn(Gdata, r, h, c)#b,t,n
        r_mean = torch.mean(r,dim=1)
        mu = mu.squeeze(-1)
        mu_arg = torch.argsort(mu,dim=1,descending=True).float()
        r_arg = torch.argsort(r_mean,dim=1,descending=True).float()
        loss_order = criterion(mu_arg,r_arg)/(2912*batch_size)
        
        optimizer.zero_grad()
        mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)
        loss = mse + lo * loss_order
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_loss_orders.append(loss_order.item())

          
     
    model_agnn.eval()
    valid_losses = []
    valid_loss_orders = []

    Rs_vail=[]
    for i,(x,r) in enumerate(valid_loader):
        x = x.to(device)
        r = r.to(device)
        edge_index = torch.tensor(re, dtype=torch.long)
        edge_index=edge_index.to(device)
        Gdata = Data(x=x,edge_index=edge_index.t().contiguous())
        with torch.no_grad():
            h = torch.zeros(1, 2912, 30).to(device)
            c = torch.zeros(1, 2912, 30).to(device)
           
            R, w, x, mu, temp,  pre_r = model_agnn(Gdata, r, h, c)#b,t,n
        r_mean = torch.mean(r,dim=1)
        mu = mu.squeeze(-1)
        mu_arg = torch.argsort(mu,dim=1,descending=True).float()
        r_arg = torch.argsort(r_mean,dim=1,descending=True).float()
        loss_order = criterion(mu_arg,r_arg)/(2912*batch_size)
        optimizer.zero_grad()
        
        mse = torch.sum(torch.square(r - pre_r))/ (2912 * r_l * batch_size)
        loss = mse + lo * loss_order 
        
        optimizer.zero_grad()
        
        valid_losses.append(loss.item())
        valid_loss_orders.append(loss_order.item())
        Rs_vail.append(torch.mean(R, dim=1))
    
    train_loss = np.average(train_losses)
    train_loss = np.average(train_losses)
    train_loss_order = np.average(train_loss_orders)

    valid_loss = np.average(valid_losses)
    valid_loss_order = np.average(valid_loss_orders)

    train_LOSS.append(train_loss)
    train_LOSS_ORDER.append(train_loss_order)

    valid_LOSS.append(valid_loss)
    valid_LOSS_ORDER.append(valid_loss_order)

    print('Epoch {}:'.format(epoch + 1))
    print(' Train平均mse：{:.4f}'.format(train_loss), ' Train Loss：{:.6f}'.format(train_loss),
          'Train order loss:{:.6f}'.format(train_loss_order))
    print(' Valid平均mse：{:.4f}'.format(valid_loss), ' Valid Loss：{:.6f}'.format(valid_loss),
          'Valid order loss:{:.6f}'.format(valid_loss_order))
    


model_agnn.eval()
test_LOSS = []
test_LOSS_ORDER = []
test_MSE=[]
test_R2 = []
Rs_TEST=[]
STDRs_TEST = []
for i,(x,r) in enumerate(test_loader):
    x = x.to(device)
    r = r.to(device)
    edge_index = torch.tensor(re, dtype=torch.long)
    edge_index=edge_index.to(device)
    Gdata = Data(x=x,edge_index=edge_index.t().contiguous())
    with torch.no_grad():
        h = torch.zeros(1, 2912, 30).to(device)
        c = torch.zeros(1, 2912, 30).to(device)
        R, w, x, mu, temp,  pre_r = model_agnn(Gdata, r, h, c)#b,t,n
    r_mean = torch.mean(r,dim=1)
    mu = mu.squeeze(-1)
    mu_arg = torch.argsort(mu,dim=1,descending=True).float()
    r_arg = torch.argsort(r_mean,dim=1,descending=True).float()
    loss_order = criterion(mu_arg,r_arg)/(2912*batch_size)
    optimizer.zero_grad()
    mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)
    R2 = 1 - (torch.sum(torch.square(r - pre_r)) / torch.sum(torch.square(r)))
    loss = mse + lo * loss_order
    
    test_LOSS.append(loss.item())
    test_LOSS_ORDER.append(loss_order.item())
    test_R2.append(R2.item())
    test_MSE.append(mse.item())

    Rs_TEST.append(torch.mean(R*math.sqrt(252), dim=1))
    STDRs_TEST.append(torch.std(R*math.sqrt(252), dim=1))    
    



test_RR = np.average(test_R2)
test_loss_order = np.average(test_LOSS_ORDER)
test_loss = np.average(test_LOSS)
test_mse = np.average(test_MSE)

print('  test MSE：{:.8f}'.format(test_mse), '  test Loss：{:.8f}'.format(test_loss),
      'test_loss_order{:.8f}'.format(test_loss_order), 'test RR{:.8f}'.format(test_RR))