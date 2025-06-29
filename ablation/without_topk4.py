import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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
import random
from torch_geometric.utils import degree





lr = 1e-6
pm1 = 10

criterion = nn.MSELoss()


def l0_loss(input, target):
    # 计算张量之间的差值
    input = input.float()
    target = target.float()

    l2_norm = criterion(input, target) / (batch_size * 2912)
    return l2_norm


# R方的计算
def r_squared(r, hat_r):
    numerator = torch.sum((r - hat_r) ** 2)
    denominator = torch.sum(r ** 2)
    return 1 - (numerator / denominator)


# 计算度向量
def compute_degree_vector(adj_matrix):
    # 每个节点的度是对应行的元素之和
    degree_vector = torch.sum(adj_matrix, dim=1)  # 按行求和，得到一个长度为 n_samples 的向量

    return degree_vector


relation = pd.read_csv(r'dataset/edge_data.csv')
relation.drop(relation.columns[0], axis=1, inplace=True)
sliced_data = torch.load("dataset/sliced_data.pth")
sliced_r = torch.load("dataset/sliced_r.pth")


class get_dataset(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        # x = self.data[index,:,:,:]
        # y = self.target[index,:,:]
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def dataset_split(data, target, ratio1, ratio2):
    # data是tensor
    spilt_index_1 = int(ratio1 * data.shape[0])
    spilt_index_2 = int(ratio2 * data.shape[0])
    train_data = data[:spilt_index_1, :, :, :]
    train_target = target[:spilt_index_1, :, :]
    valid_data = data[spilt_index_1:spilt_index_2, :, :, :]
    valid_target = target[spilt_index_1:spilt_index_2, :, :]
    test_data = data[spilt_index_2:, :, :, :]
    test_target = target[spilt_index_2:, :, :]
    return train_data, train_target, valid_data, valid_target, test_data, test_target


train_data, train_target, valid_data, valid_target, test_data, test_target = dataset_split(sliced_data, sliced_r, 0.7,
                                                                                           0.9)
train_dataset = get_dataset(train_data, train_target)
valid_dataset = get_dataset(valid_data, valid_target)
test_dataset = get_dataset(test_data, test_target)


def get_data_loader(batch_size, trainFolder, validFolder, testFolder):
    train_loader = torch.utils.data.DataLoader(trainFolder,
                                               batch_size=batch_size,
                                               shuffle=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(validFolder,
                                               batch_size=batch_size,
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testFolder,
                                              batch_size=batch_size,
                                              shuffle=False, drop_last=True)

    return train_loader, valid_loader, test_loader


batch_size = 1
train_loader, valid_loader, test_loader = get_data_loader(batch_size, train_dataset, valid_dataset, test_dataset)


class AGNN(torch.nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        # 图滤波层，使用GCN
        self.conv_1 = GCNConv(161, 100)
        self.conv_2 = GCNConv(100, 50)
        self.fc_1_1 = nn.Linear(50, 64)
        self.fc_1_2 = nn.Linear(64, 32)
        self.fc_1_3 = nn.Linear(32, 1)
        self.fc_2 = nn.Linear(30, 50, bias=False)
        self.beta_layer1 = nn.Linear(100,240)
        self.beta_layer2 = nn.Linear(240,50)
        self.bn1 = nn.BatchNorm2d(2912)
        self.layer40to1 = nn.Linear(40, 1)
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
        x = self.bn1(x)  # 在t的维度上做batchnorm
        x = torch.transpose(x, 1, 2)

        # 首先输入邻接矩阵A和特征数据X，得到嵌入矩阵H
        x = self.conv_1(x, edge_index)  # x[batch,T,stock,feature]
        x = torch.tanh(x)
        x = self.conv_2(x, edge_index)  # x[batch,T,stock,feature]
        x = torch.tanh(x)
        x = torch.transpose(x, 1, 3)
        x = self.layer40to1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.tanh(x)  # x[batch,1,stock,feature]
        x = x.squeeze(1)  # x[batch,stock,feature]
        _, (H, _) = self.lstm(x, (h, c))  # H[batch(1),stock(2912),feature(30)]

        

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
device = torch.device("cuda:2")
model_agnn.to(device)
optimizer = torch.optim.Adam(model_agnn.parameters(), lr=lr)

col_relation = ['primary_code', 'related_code']
re = np.array(relation[col_relation])
x_l = 40
r_l = 10
num_epoch = 10

train_LOSS = []
train_LOSS_ORDER = []
train_MSE = []
train_MQ = []

valid_LOSS = []
valid_MSE = []
valid_MQ = []
valid_LOSS_ORDER = []


for epoch in range(num_epoch):
    print('Epoch：{}'.format(epoch + 1))

    train_losses = []
    train_mses = []
    train_loss_orders = []
    train_Qs = []
    # 训练过程
    model_agnn.train()
    for  i, (x, r) in enumerate(train_loader):
        x = x.to(device)
        r = r.to(device)
        edge_index = torch.tensor(re, dtype=torch.long)
        edge_index = edge_index.to(device)
        Gdata = Data(x=x, edge_index=edge_index.t().contiguous())
        h = torch.zeros(1, 2912, 30).to(device)
        c = torch.zeros(1, 2912, 30).to(device)
        R, w, x, mu, temp,  pre_r = model_agnn(Gdata, r, h, c)  # b,t,n
        r_mean = torch.mean(r, dim=1)
        mu = mu.squeeze(-1)
        mu_arg = torch.argsort(mu, dim=1, descending=True)
        r_arg = torch.argsort(r_mean, dim=1, descending=True)

        loss_order = l0_loss(r_arg, mu_arg) / (2912 * batch_size)

        # degree_k = compute_degree_vector(W)
        # edge_m = torch.sum(degree_k) / 2
        # degree_k = degree_k.unsqueeze(1)
        # # degree_k = degree_k.float()
        # # edge_m = edge_m.float()
        # # W = W.float()
        # K = W - (1 / (2 * edge_m)) * torch.matmul(degree_k.float(), degree_k.t().float())
        # CK = torch.matmul(C.transpose(1, 2), K)
        # tr_ckc = torch.trace(torch.matmul(CK, C).squeeze(0))
        # # R22 = torch.sum(torch.square(r-pre_r))
        # # R33 = torch.sum(torch.square(r))
        # # RRRR = (torch.sum(torch.square(r-pre_r))/torch.sum(torch.square(r)))
        # # R2 = 1-(torch.sum(torch.square(r-pre_r))/torch.sum(torch.square(r)))
        # Q = ((1 / 2) / edge_m) * tr_ckc

        # if i == len(train_loader)-1:
                # torch.save(w, "industrychain/results/1e-3/模型中间输出/train_w.pth")
                # torch.save(R, "industrychain/results/1e-3/模型中间输出/train_R.pth")
                # torch.save(x, "industrychain/results/1e-3/模型中间输出/train_H.pth")
                # torch.save(mu, "industrychain/results/1e-3/模型中间输出/train_mu.pth")
                # torch.save(temp, "industrychain/results/1e-3/模型中间输出/train_sigma的逆.pth")
                # torch.save(r,"industrychain/results/1e-3/模型中间输出/train_r的值.pth")
                #torch.save(pre_r,"D:/stu_perfect/毕业论文/数据集/result/GCN-LSTM/模型中间输出/train_pre_r.pth")
        optimizer.zero_grad()

        mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)

        loss = mse + pm1 * loss_order 
        
        
        if torch.isnan(mse).any():
            print("MSE became NaN at step:", i)  # 用 i 代替 step
            print("mu:", mu)
            print("pre_r:", pre_r)
            print("r:", r)
            break


        # print('  Train mse：{:.4f}'.format(mse.item()),'  Train Loss：{:.6f}'.format(loss.item()))
        loss.backward()
        
        optimizer.step()
        train_mses.append(mse.item())
        train_loss_orders.append(loss_order.item())
        train_losses.append(loss.item())
        #train_Qs.append(Q.item())

    model_agnn.eval()
    valid_losses = []
    valid_mses = []
    valid_loss_orders = []
    valid_Qs = []

    Rs_vail = []
    for i, (x, r) in enumerate(valid_loader):
        x = x.to(device)
        r = r.to(device)
        edge_index = torch.tensor(re, dtype=torch.long)
        edge_index = edge_index.to(device)
        Gdata = Data(x=x, edge_index=edge_index.t().contiguous())
        with torch.no_grad():
            h = torch.zeros(1, 2912, 30).to(device)
            c = torch.zeros(1, 2912, 30).to(device)
            R, w, x, mu, temp,  pre_r= model_agnn(Gdata, r, h, c)  # b,t,n
            r_mean = torch.mean(r, dim=1)
            mu = mu.squeeze(-1)
            mu_arg = torch.argsort(mu, dim=1, descending=True)
            r_arg = torch.argsort(r_mean, dim=1, descending=True)
            loss_order = l0_loss(r_arg, mu_arg) / (2912 * batch_size)

            # degree_k = compute_degree_vector(W)
            # edge_m = torch.sum(degree_k) / 2
            # degree_k = degree_k.unsqueeze(1)
            # K = W - (1 / (2 * edge_m)) * torch.matmul(degree_k.float(), degree_k.t().float())
            # CK = torch.matmul(C.transpose(1, 2), K)
            # tr_ckc = torch.trace(torch.matmul(CK, C).squeeze(0))
            # Q = ((1 / 2) / edge_m) * tr_ckc

        optimizer.zero_grad()
        # if epoch == num_epoch-1:
        #     if i == len(valid_loader)-1:
                # torch.save(w, "industrychain/results/1e-3/模型中间输出/valid_w.pth")
                # torch.save(R, "industrychain/results/1e-3/模型中间输出/valid_R.pth")
                # torch.save(x, "industrychain/results/1e-3/模型中间输出/valid_H.pth")
                # torch.save(mu, "industrychain/results/1e-3/模型中间输出/valid_mu.pth")
                # torch.save(temp, "industrychain/results/1e-3/模型中间输出/valid_sigma的逆.pth")
                # torch.save(r,"industrychain/results/1e-3/模型中间输出/valid_r的值.pth")
                # torch.save(pre_r,"D:/stu_perfect/毕业论文/数据集/result/GCN-LSTM/模型中间输出/valid_pre_r.pth")

        # test1 = torch.sum(torch.square(r-pre_r))
        # test2 = torch.sum(torch.square(r))

        mse = torch.sum(torch.square(r - pre_r))/ (2912 * r_l * batch_size)
        loss = mse + pm1 * loss_order 
        #+ pm2 * Q

        
        
        optimizer.zero_grad()
        valid_mses.append(mse.item())
        valid_losses.append(loss.item())
        valid_loss_orders.append(loss_order.item())
        #valid_Qs.append(Q.item())
    train_mse = np.average(train_mses)
    train_loss = np.average(train_losses)
    train_loss_order = np.average(train_loss_orders)
    #train_Q = np.average(train_Qs)

    valid_mse = np.average(valid_mses)
    valid_loss = np.average(valid_losses)
    valid_loss_order = np.average(valid_loss_orders)
    #valid_Q = np.average(valid_Qs)

    train_LOSS.append(train_loss)
    train_MSE.append(train_mse)
    #train_MQ.append(train_Q)
    train_LOSS_ORDER.append(train_loss_order)

    valid_LOSS.append(valid_loss)
    valid_MSE.append(valid_mse)
    #valid_MQ.append(valid_Q)
    valid_LOSS_ORDER.append(valid_loss_order)

    print('Epoch {}:'.format(epoch + 1))
    print(' Train平均mse：{:.4f}'.format(train_mse), ' Train Loss：{:.6f}'.format(train_loss),
          'Train order loss:{:.6f}'.format(train_loss_order))
    print(' Valid平均mse：{:.4f}'.format(valid_mse), ' Valid Loss：{:.6f}'.format(valid_loss),
          'Valid order loss:{:.6f}'.format(valid_loss_order))
    # with open(
    #         "lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_epoch{}_train_LOSS.txt".format(x_l, r_l, batch_size, epoch),
    #         'wt') as f:
    #     for i in train_LOSS:
    #         print(i, file=f)
    # with open(
    #         "lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_epoch{}_train_MSE.txt".format(x_l, r_l, batch_size, epoch),
    #         'wt') as f:
    #     for i in train_MSE:
    #         print(i, file=f)
    # with open(
    #         "lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_epoch{}_valid_LOSS.txt".format(x_l, r_l, batch_size, epoch),
    #         'wt') as f:
    #     for i in valid_LOSS:
    #         print(i, file=f)
    # with open(
    #         "lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_epoch{}_valid_MSE.txt".format(x_l, r_l, batch_size, epoch),
    #         'wt') as f:
    #     for i in valid_MSE:
    #         print(i, file=f)
    # combined_tensor = torch.stack(Rs_vail)
    # torch.save(combined_tensor, 'lhlrun/results/1e-3/模型中间输出/R_VAILD.pt')

model_agnn.eval()
test_LOSS = []
test_MSE = []
test_MQ = []
test_LOSS_ORDER = []
test_R2 = []

Rs_TEST=[]
# pre_TEST1=[]
# pre_TEST2=[]
#dm_TEST=[]

for i, (x, r) in enumerate(test_loader):
    x = x.to(device)
    r = r.to(device)
    edge_index = torch.tensor(re, dtype=torch.long)
    edge_index = edge_index.to(device)
    Gdata = Data(x=x, edge_index=edge_index.t().contiguous())
    with torch.no_grad():
        h = torch.zeros(1, 2912, 30).to(device)
        c = torch.zeros(1, 2912, 30).to(device)
        R, w, x, mu, temp,  pre_r = model_agnn(Gdata, r, h, c)  # b,t,n
        r_mean = torch.mean(r, dim=1)
        mu = mu.squeeze(-1)
        mu_arg = torch.argsort(mu, dim=1, descending=True)
        r_arg = torch.argsort(r_mean, dim=1, descending=True)
        loss_order = l0_loss(r_arg, mu_arg) / (2912 * batch_size)


    optimizer.zero_grad()

    mse = torch.sum(torch.square(r - pre_r)) / (2912 * r_l * batch_size)
    loss = mse + pm1 * loss_order
    R2 = 1 - (torch.sum(torch.square(r - pre_r)) / torch.sum(torch.square(r)))
    test_MSE.append(mse.item())
    
    test_LOSS.append(loss.item())
    Rs_TEST.append(torch.mean(R, dim=1))
    test_LOSS_ORDER.append(loss_order.item())
    
    Rs_TEST.append(torch.mean(R*math.sqrt(252), dim=1))
    
    test_R2.append(R2.item())
    

    if i == len(test_loader)-1:
        # torch.save(w, "industrychain/results/1e-3/模型中间输出/test_w.pth")
        #torch.save(R, "/home/huquan/industrychain/lhl/GCN_LSTM/模型中间输出/test_R.pth")
        # torch.save(x, "industrychain/results/1e-3/模型中间输出/testmu.pth")
        # torch.save(temp, "industrychain/results/1e-3/模型中间输出/test__H.pth")
        # torch.save(mu, "industrychain/results/1e-3/模型中间输出/test_sigma的逆.pth")
        torch.save(r,"/home/huquan/industrychain/lhl/ablation_result/w_c/test_r.pth")
        torch.save(pre_r,"/home/huquan/industrychain/lhl/ablation_result/w_c/test_pre_r.pth")
# with open("lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_test_LOSS.txt".format(x_l, r_l, batch_size), 'wt') as f:
#     for i in test_LOSS:
#         print(i, file=f)
# with open("lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_test_MSE.txt".format(x_l, r_l, batch_size), 'wt') as f:
#     for i in test_MSE:
#         print(i, file=f)
# with open("lhl/GCN_LSTM/1e-3-top6/xl{}_rl{}_batchsize{}_test_R2.txt".format(x_l, r_l, batch_size), 'wt') as f:
#     for i in test_R2:
#         print(i, file=f)
# combined_tensor = torch.stack(Rs_TEST)
# #保存合并后的张量为文件
# torch.save(combined_tensor, '/home/huquan/industrychain/lhl/GCN_LSTM/收益R/R_test.pt')

test_mse = np.average(test_MSE)
test_RR = np.average(test_R2)

test_loss_order = np.average(test_LOSS_ORDER)
test_loss = np.average(test_LOSS)
#test_Q = np.average(test_MQ)
print('  test MSE：{:.8f}'.format(test_mse), '  test Loss：{:.6f}'.format(test_loss),
      'test_loss_order{:.6f}'.format(test_loss_order), 'test RR{:.6f}'.format(test_RR))