import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class PtModel(nn.Module):
    def __init__(self, args):
        super(PtModel, self).__init__()

        self.feat = PointNetEncoder(args, 3)

        self.fc0 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, args.out_channel)

        self.dropout = nn.Dropout(p=args.dropout)
        self.bn0 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x):    # x  （batch，channl，npoint）
        with torch.cuda.amp.autocast():
            x,trans,trans_feat = self.feat(x)
            x = F.relu(self.bn0(self.fc0(x)))
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)  # 输出k维数据

        return x,trans_feat

class PtLoss_euler(torch.nn.Module):  # 重写损失函数
    def __init__(self, Lamda):
        super(PtLoss_euler, self).__init__()
        self.ratio = Lamda

    def forward(self, pred, target,arg,trans_feat):

        if trans_feat!=None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        else:
            mat_diff_loss =0
        # pred 是概率向量  target是目标标签
        loss_pitch = nn.MSELoss().cuda()
        loss_yaw  = nn.MSELoss().cuda()
        loss_roll = nn.MSELoss().cuda()
        loss  = loss_pitch(pred[0], target[0]).cuda() + loss_roll(pred[1], target[1]).cuda() + loss_yaw(pred[2], target[2]).cuda()
        return loss

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):   #到global feature 前部分
    def __init__(self,args, channel=3,feature_transform=False,stn=False,global_feature=False):
        super(PointNetEncoder, self).__init__()
        self.global_feature=global_feature
        self.stn = stn
        self.c = c = 16
        self.channels = c * 47 + channel

        if stn:
            self.STN3d = STN3d(channel)

        self.conv0 = torch.nn.Conv1d(channel, c, 1)
        self.conv1 = torch.nn.Conv1d(c, c*2, 1)
        self.conv2 = torch.nn.Conv1d(c*2, c*4, 1)
        self.conv3 = torch.nn.Conv1d(c*4, c*8, 1)
        self.conv4 = torch.nn.Conv1d(c*8,c*16,1)

        self.bn0 = nn.BatchNorm1d(c)
        self.bn1 = nn.BatchNorm1d(c*2)
        self.bn2 = nn.BatchNorm1d(c*4)
        self.bn3 = nn.BatchNorm1d(c*8)

        self.conv5 = torch.nn.Conv1d(c*16,1024,1)
        self.bn5 = nn.BatchNorm1d(c*16)

        self.Dropout = nn.Dropout(args.dropout)

        self.conv6 = torch.nn.Conv1d(self.channels,1024, 1)
        self.bn6 = nn.BatchNorm1d(2048)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        if self.stn:
            trans = self.STN3d(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        else:
            trans =None

        out0 = F.relu(self.bn0(self.conv0(x)))

        if self.feature_transform:
            trans_feat = self.fstn(out0)
            out0 = out0.transpose(2, 1)
            out0 = torch.bmm(out0, trans_feat)
            out0 = out0.transpose(2, 1)
        else:
            trans_feat = None

        out1 = F.relu(self.bn1(self.conv1(out0)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = self.conv4(out3)

        out5 = torch.max(out4, 2, keepdim=True)[0].view(-1, self.c*16)  #  [8, 1024, 1])

        if self.global_feature:
            global_feature = out5.view(-1, self.c*16, 1).repeat(1, 1, N)
            new_feature = torch.cat([global_feature, out4, out3, out2, out1, out0,x], 1)

            new_feature = self.conv6(new_feature)
            new_feature = torch.max(new_feature, 2, keepdim=True)[0].view(-1, 1024)  # 取x中每一行即4096个数中的最大值 maxpooling

            return new_feature,trans,trans_feat
        else:

            # out4 = F.relu(self.bn4(self.conv4(out3)))
            # out5 = self.conv5(out4)
            return out5,trans,trans_feat

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss