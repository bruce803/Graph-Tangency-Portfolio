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
from torch_geometric.nn import GATConv, ChebConv, GCNConv, GeneralConv, SAGEConv, AntiSymmetricConv
from sklearn.preprocessing import normalize, LabelEncoder
from torch_geometric import graphgym
from torch_geometric.graphgym import loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

seed_value = 42
device = torch.device("cuda:2")
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed_value)

relation = pd.read_csv(r'/home/huquan/industrychain/dataset/edge_data.csv')
relation.drop(relation.columns[0], axis=1, inplace=True)
sliced_data = torch.load("/home/huquan/industrychain/dataset/sliced_data.pth")
sliced_r = torch.load("/home/huquan/industrychain/dataset/sliced_r.pth")

lr = 1e-3
k=6
pm1=1

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
                                               shuffle=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(validFolder,
                                               batch_size=1,
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testFolder,
                                            batch_size=1,
                                            shuffle=False, drop_last=True)

    
    return train_loader, valid_loader,test_loader


batch_size=1
train_loader, valid_loader,test_loader=get_data_loader(batch_size, train_dataset, valid_dataset, test_dataset)
class AGNN(torch.nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        # 图滤波层，使用AntiSymmetric
        self.AntiSymmetric_1 = AntiSymmetricConv(161)
        self.AntiSymmetric_2 = AntiSymmetricConv(161)
        self.fc_0 = nn.Linear(161, 50)
        self.fc_1_1 = nn.Linear(50, 64)
        self.fc_1_2 = nn.Linear(64, 32)
        self.fc_1_3 = nn.Linear(32, 1)
        self.fc_2 = nn.Linear(50, 50, bias=False)
        # self.beta_layer1 = nn.Linear(100,240)
        # self.beta_layer2 = nn.Linear(240,50)
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
        w_1 = -50*torch.exp(-8*w) # [B,N,1]
        w_2 = -50*torch.exp(8*w)
        h_1 = F.softmax(w_1,dim=1)
        h_2 = F.softmax(w_2,dim=1)
        w_new = h_1 - h_2
        return w_new

    def compute_topk_binary_matrix_gpu(self, emb, k):
    
        # 计算两两欧氏距离矩阵 (n, n)
        D = torch.cdist(emb, emb, p=2)  # 使用 torch.cdist 计算欧氏距离
    
        n = emb.shape[0]
        W = torch.zeros((n, n), device=emb.device, dtype=torch.int32)  # 初始化二值矩阵，存储在 GPU 上

        # 找到每一行的 Top-k 最近邻
        _, topk_indices = torch.topk(-D, k=k+1, dim=1)  # 取负排序得到最小距离（包括自身）
        topk_indices = topk_indices[:, 1:]  # 排除自身（每行第一个索引）

         # 构造二值化矩阵
        row_indices = torch.arange(n, device=emb.device).unsqueeze(1).expand_as(topk_indices)  # 行索引
        W[row_indices, topk_indices] = 1  # 将 Top-k 索引位置置为 1

        return W
    
    def forward(self, data, r, h, c):

        x, edge_index = data.x, data.edge_index
        # x[batch,T,stock,feature]
        x = torch.transpose(x, 1, 2)
        x=self.bn1(x)#在t的维度上做batchnorm
        x = torch.transpose(x, 1, 2)
        
        # 首先输入邻接矩阵A和特征数据X，得到嵌入矩阵H
        x = self.AntiSymmetric_1(x, edge_index)# x[batch,T,stock,feature]
        x = torch.tanh(x)
        x = self.AntiSymmetric_2(x, edge_index)# x[batch,T,stock,feature]
        x = torch.tanh(x)
        x = self.fc_0(x)
        x = torch.tanh(x)
        x = torch.transpose(x, 1, 3)
        x = self.layer40to1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.tanh(x)# x[batch,1,stock,feature]
        x = x.squeeze(1)# x[batch,stock,feature]
        _, (H,_) = self.lstm(x,(h,c))# H[batch(1),stock(2912),feature(30)]
        
        # 使用嵌入矩阵H估计均值向量μ
        mu = self.fc_1_1(x)# H[batch,stock,feature]
        mu = torch.tanh(mu)
        mu = self.fc_1_2(mu)# H[batch,stock,feature]
        mu = torch.tanh(mu)
        mu = self.fc_1_3(mu)# H[batch,stock,feature] mu[batch,stock,1]
        mu = torch.tanh(mu)
        
        # 使用嵌入矩阵H计算得到HWH^T来估计sigma的逆
        hw = self.fc_2(x)  # x[batch,stock,feature]
        # temp = torch.matmul(temp, torch.transpose(temp, 1, 2))
        temp = torch.matmul(hw, torch.transpose(hw, 1, 2))
        temp = torch.tanh(temp)  # temp[batch,stock,stock]

        # 使用H计算W
        W = self.compute_topk_binary_matrix_gpu(H.squeeze(0), k)
        W = W.unsqueeze(0)
        # 使用H计算beta
        beta = torch.tanh(self.batch1(self.beta_layer1(x.unsqueeze(1))))
        beta = torch.tanh(self.batch2(self.beta_layer2(beta)))
        beta = torch.tanh(self.batch3(self.beta_layer3(beta)))
        beta = self.beta_layer4(beta).squeeze(1)
        
        # 根据论文中的公式，w等于协方差逆矩阵乘上均值向量,再经过softmax()非线性激活函数来近似经典资产定价排序操作
        w = torch.matmul(W.float(), mu)  # w[batch,stock,1]
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
        return R, w, x, mu, temp, W, pre_r


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

# writer=SummaryWriter("tensorboard/3")
train_LOSSes = []
train_LOSSes1 = []
train_LOSSes2 = []
train_LOSSes3 = []
train_srs = []
valid_LOSSes = []
valid_LOSSes1 = []
valid_LOSSes2 = []
valid_LOSSes3 = []
valid_srs = []
for epoch in range(num_epoch):
    print('Epoch：{}'.format(epoch+1))
    train_losses = []
    train_losses1 = []
    train_losses2 = []
    train_losses3 = []
    train_SRs = []
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
        
        R, w, x, mu, temp, W, pre_r = model_agnn(Gdata, r, h, c)#b,t,n
        r_mean = torch.mean(r,dim=1)
        mu = mu.squeeze(-1)
        mu_arg = torch.argsort(mu,dim=1,descending=True).float()
        r_arg = torch.argsort(r_mean,dim=1,descending=True).float()
        loss_order = criterion(mu_arg,r_arg)/2912/batch_size
        # if epoch == num_epoch-1:
        #     if i == len(train_loader)-1:
        #         torch.save(w, "results/GNNs/AntiSymmetricConv_1/train_w.pth")
        #         torch.save(R, "results/GNNs/AntiSymmetricConv_1/train_R.pth")
        #         # torch.save(x, "results/GNNs/AntiSymmetricConv_1/train_H.pth")
        #         torch.save(mu, "results/GNNs/AntiSymmetricConv_1/train_mu.pth")
        #         torch.save(temp, "results/GNNs/AntiSymmetricConv_1/train_sigma的逆.pth")
        #         torch.save(r, "results/GNNs/AntiSymmetricConv_1/train_r.pth")
        optimizer.zero_grad()
        
        mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)

        loss = mse + pm1 * loss_order 
       
       
        
        # print('  Train 夏普比率：{:.4f}'.format(SR.item()),'  Train Loss：{:.6f}'.format(loss.item()))
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_losses1.append(mse.item())
        train_losses3.append(loss_order.item())
    # with open("results/40epochwsrrh/xl{}_rl{}_batchsize{}_epoch{}_train_losses.txt".format(x_l,r_l,batch_size,epoch), 'wt') as f:
    #     for i in train_losses:
    #         print(i, file=f)
    # with open("results/40epochwsrrh/xl{}_rl{}_batchsize{}_epoch{}_train_SRs.txt".format(x_l,r_l,batch_size,epoch), 'wt') as f:
    #     for i in train_SRs:
    #         print(i, file=f)
            
          
     
    model_agnn.eval()
    valid_losses = []
    valid_losses1 = []
    valid_losses2 = []
    valid_losses3 = []
    valid_SRs = []
    Rs_vail=[]
    STDRs_vail = []
    for i,(x,r) in enumerate(valid_loader):
        x = x.to(device)
        r = r.to(device)
        edge_index = torch.tensor(re, dtype=torch.long)
        edge_index=edge_index.to(device)
        Gdata = Data(x=x,edge_index=edge_index.t().contiguous())
        with torch.no_grad():
            h = torch.zeros(1, 2912, 30).to(device)
            c = torch.zeros(1, 2912, 30).to(device)
            
            R, w, x, mu, temp, W, pre_r = model_agnn(Gdata, r, h, c)#b,t,n
        r_mean = torch.mean(r,dim=1)
        mu = mu.squeeze(-1)
        mu_arg = torch.argsort(mu,dim=1,descending=True).float()
        r_arg = torch.argsort(r_mean,dim=1,descending=True).float()
        loss_order = criterion(mu_arg,r_arg)/2912/batch_size
        optimizer.zero_grad()
        #if epoch == num_epoch-1:
            # if i == len(valid_loader)-1:
            #     torch.save(w, "results/GNNs/AntiSymmetricConv_1/valid_w.pth")
            #     torch.save(R, "results/GNNs/AntiSymmetricConv_1/valid_R.pth")
            #     # torch.save(x, "results/GNNs/AntiSymmetricConv_1/valid_H.pth")
            #     torch.save(mu, "results/GNNs/AntiSymmetricConv_1/valid_mu.pth")
            #     torch.save(temp, "results/GNNs/AntiSymmetricConv_1/valid_sigma的逆.pth")
            #     torch.save(r, "results/GNNs/AntiSymmetricConv_1/valid_r.pth")
        
        mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)

        loss = mse + pm1 * loss_order 
        
        # print('  Valid 夏普比率：{:.4f}'.format(SR.item()),'  Valid Loss：{:.6f}'.format(loss.item()))
        optimizer.zero_grad()
        valid_losses.append(loss.item())
        valid_losses1.append(mse.item())
        valid_losses3.append(loss_order.item())
        if epoch == num_epoch-1:
            Rs_vail.append(torch.mean(R, dim=1))
            STDRs_vail.append(torch.std(R, dim=1))
    
    train_loss = np.average(train_losses)
    train_loss1 = np.average(train_losses1)
    
    train_loss3 = np.average(train_losses3)
    
    valid_loss = np.average(valid_losses)
    valid_loss1 = np.average(valid_losses1)
    
    valid_loss3 = np.average(valid_losses3)
    train_LOSSes.append(train_loss)
    train_LOSSes1.append(train_loss1)
    
    train_LOSSes3.append(train_loss3)
    
    valid_LOSSes.append(valid_loss)
    valid_LOSSes1.append(valid_loss1)
    
    valid_LOSSes3.append(valid_loss3)
    
    print('Epoch {}:'.format(epoch + 1))
    print(' Train Loss：{:.6f}'.format(train_loss),' Train mse：{:.6f}'.format(train_loss1),' Train Lossorder：{:.6f}'.format(train_loss3))
    print(' Valid Loss：{:.6f}'.format(valid_loss),' Valid mse：{:.6f}'.format(valid_loss1),' Valid Lossorder：{:.6f}'.format(valid_loss3))
    # with open("results/30epochwsrrh/4xl{}_rl{}_batchsize{}_epoch{}_train_losses.txt".format(x_l,r_l,batch_size,epoch), 'wt') as f:
    #     for i in train_LOSSes:
    #         print(i, file=f)
    # with open("results/30epochwsrrh/4xl{}_rl{}_batchsize{}_epoch{}_train_SRs.txt".format(x_l,r_l,batch_size,epoch), 'wt') as f:
    #     for i in train_srs:
    #         print(i, file=f)
    # with open("results/30epochwsrrh/4xl{}_rl{}_batchsize{}_epoch{}_valid_losses.txt".format(x_l,r_l,batch_size,epoch), 'wt') as f:
    #     for i in valid_LOSSes:
    #         print(i, file=f)
    # with open("results/30epochwsrrh/4xl{}_rl{}_batchsize{}_epoch{}_valid_SRs.txt".format(x_l,r_l,batch_size,epoch), 'wt') as f:
    #     for i in valid_srs:
    #         print(i, file=f)
    # if epoch == num_epoch-1:
    #     combined_tensor1 = torch.stack(Rs_vail)
    #     combined_tensor2 = torch.stack(STDRs_vail)
    #     # print(combined_tensor)
    #     # 保存合并后的张量为文件
    #     torch.save(combined_tensor1, 'results/GNNs/AntiSymmetricConv_1/R_VAILD.pt')
    #     torch.save(combined_tensor2, 'results/GNNs/AntiSymmetricConv_1/stdR_VAILD.pt')


model_agnn.eval()
test_losses = []
test_losses1 = []
test_losses2 = []
test_losses3 = []
test_SRs = []
Rs_TEST=[]
STDRs_TEST = []
Rs_test=[]
for i,(x,r) in enumerate(test_loader):
    x = x.to(device)
    r = r.to(device)
    edge_index = torch.tensor(re, dtype=torch.long)
    edge_index=edge_index.to(device)
    Gdata = Data(x=x,edge_index=edge_index.t().contiguous())
    with torch.no_grad():
        h = torch.zeros(1, 2912, 30).to(device)
        c = torch.zeros(1, 2912, 30).to(device)
        
        R, w, x, mu, temp, W, pre_r = model_agnn(Gdata, r, h, c)#b,t,n
    r_mean = torch.mean(r,dim=1)
    mu = mu.squeeze(-1)
    mu_arg = torch.argsort(mu,dim=1,descending=True).float()
    r_arg = torch.argsort(r_mean,dim=1,descending=True).float()
    loss_order = criterion(mu_arg,r_arg)/2912/batch_size
    optimizer.zero_grad()
    
    mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)
    loss = mse + pm1*loss_order
    R2 = 1 - (torch.sum(torch.square(r - pre_r)) / torch.sum(torch.square(r)))
    
    # print('  Test 夏普比率：{:.4f}'.format(SR.item()),'  Test Loss：{:.6f}'.format(loss.item()))
    
    test_losses.append(loss.item())
    test_losses1.append(mse.item())
    test_losses2.append(R2.item())
    test_losses3.append(loss_order.item())
    Rs_TEST.append(torch.mean(R*math.sqrt(252), dim=1))
    Rs_test.append(R)
    STDRs_TEST.append(torch.std(R*math.sqrt(252), dim=1))
    if i == len(test_loader)-1:
        # torch.save(w, "results/GNNs/AntiSymmetricConv_1/test_w.pth")
        # torch.save(R, "results/GNNs/AntiSymmetricConv_1/test_R.pth")
        # torch.save(x, "results/GNNs/AntiSymmetricConv_1/test_H.pth")
        # torch.save(mu, "results/GNNs/AntiSymmetricConv_1/test_mu.pth")
        # torch.save(temp, "results/GNNs/AntiSymmetricConv_1/test_sigma的逆.pth")
        torch.save(r,"/home/huquan/industrychain/lhl/backbone_result/A/1/test_r.pth")
        torch.save(pre_r,"/home/huquan/industrychain/lhl/backbone_result/A/1/test_pre_r.pth")
# with open("results/30epochwsrrh/4xl{}_rl{}_batchsize{}_test_losses.txt".format(x_l,r_l,batch_size), 'wt') as f:
#     for i in test_losses:
#         print(i, file=f)
# with open("results/30epochwsrrh/4xl{}_rl{}_batchsize{}_test_SRs.txt".format(x_l,r_l,batch_size), 'wt') as f:
#     for i in test_SRs:
#         print(i, file=f)
combined_tensor1 = torch.stack(Rs_TEST)
combined_tensor2 = torch.stack(STDRs_TEST)
combined_tensor3 = torch.stack(Rs_test)
    # 保存合并后的张量为文件
# print(combined_tensor)
# torch.save(combined_tensor1, 'results/GNNs/AntiSymmetricConv_1/R_test.pt')
# torch.save(combined_tensor2, 'results/GNNs/AntiSymmetricConv_1/stdR_test.pt')
# torch.save(combined_tensor3, 'results/GNNs/AntiSymmetricConv_1/R_test1.pt')


test_loss = np.average(test_losses)
test_loss1 = np.average(test_losses1)
test_loss2 = np.average(test_losses2)
test_loss3 = np.average(test_losses3)
print(' Test Loss：{:.6f}'.format(test_loss),' Test mse：{:.6f}'.format(test_loss1),' Test R2：{:.6f}'.format(test_loss2),' Test Loss3：{:.6f}'.format(test_loss3))

#torch.save(model_agnn.state_dict(), 'model/AntiSymmetric.pth')