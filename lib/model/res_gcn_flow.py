import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,GATConv

from lib.model.Resnets import *

from lib.core.guassian import Gaussian_Map



class GCNModule(torch.nn.Module):
    def __init__(self):
        super(GCNModule, self).__init__()
        self.conv1 = GATConv(6, 128)
        self.conv2 = GATConv(128, 1)
        
        self.relu=nn.ReLU(True)
        self.sigmoid=nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        #x = self.conv2(x, edge_index)
        #x = self.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return self.sigmoid(x)

class RES_GCN(nn.Module):
    def __init__(self):
        super(RES_GCN, self).__init__()
        
        self.backbone=resnet18(pretrained=True)
        self.backbone_flow=resnet18(pretrained=True)
        pre_stage_channels=512
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(int(pre_stage_channels/1), int(pre_stage_channels/2), (4,4), stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(int(pre_stage_channels/2)),
                                    nn.ReLU(True),)
                                    
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(int(pre_stage_channels/2), int(pre_stage_channels/4), 4, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(int(pre_stage_channels/4)),
                                    nn.ReLU(True),)
                                    
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(int(pre_stage_channels/4), int(pre_stage_channels/8), 4, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(int(pre_stage_channels/8)),
                                    nn.ReLU(True),)
        
        
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(int(pre_stage_channels/8), int(pre_stage_channels/8), 4, stride=2, padding=1,bias=False),
                                    nn.BatchNorm2d(int(pre_stage_channels/8)),
                                    nn.ReLU(True),)
        
        self.final_layer_1 = nn.Sequential(nn.Conv2d(int(pre_stage_channels/8), int(pre_stage_channels/8),3,stride=1, padding=1),
                                           nn.BatchNorm2d(int(pre_stage_channels/8)),
                                           nn.ReLU(True),
                                           nn.Conv2d(int(pre_stage_channels/8), 1,3,stride=1, padding=1),)
        
        self.gcn=GCNModule()
        
        self.sigmoid=nn.Sigmoid()
        self.gaussian=Gaussian_Map()

    def forward(self, x, flow,data):
        
        #encoding
        res_list=self.backbone(x)
        res_list_flow=self.backbone_flow(flow)
        
        x=res_list[-1]+res_list_flow[-1]
        
        #decoding
        x=self.deconv1(x)
        x=x+res_list[-2]+res_list_flow[-2]
        
        #gcn
        node=self.gcn(data)
        g_map=self.gaussian(node,data.x,data.batch,x.shape[2],x.shape[3],x.shape[0])
        x=g_map*x
        
        
        x=self.deconv2(x)
        x=x+res_list[-3]+res_list_flow[-3]
        
        
        x=self.deconv3(x)
        x=x+res_list[-4]+res_list_flow[-4]
        
        x=self.deconv4(x)
        #x=self.deconv5(x)

        x = self.final_layer_1(x)
        
        x=self.sigmoid(x)
        
        return x,node
