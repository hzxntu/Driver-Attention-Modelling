# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

import torch
import torch.nn as nn
import math

class Gaussian_Map(nn.Module):
     def __init__(self,constraint=True):
         super(Gaussian_Map,self).__init__()
         self.epsilon=0#1e-4#0#1e-5
     
     def forward(self,pred,nodes,batch_list,h,w,batch_num):
     
        
        g_maps=torch.zeros((batch_num,1,h,w)).cuda()
        
        for i,idx in enumerate(batch_list):
        
                x_t = torch.mm(torch.ones((h, 1)), self._linspace(0, 1.0, w).view(1,-1)).cuda()
                y_t = torch.mm(self._linspace(0, 1.0, h).view(-1,1), torch.ones((1, w))).cuda()
                
                mu_x=nodes[i,0]
                mu_y=nodes[i,1]
                sigma_x=nodes[i,2]/1
                sigma_y=nodes[i,3]/1
                
                #eps=self.epsilon.repeat(self.num_gaussian)
                
                
                gaussian = 1 / (torch.add(2 * math.pi * sigma_x * sigma_y, self.epsilon)) * \
                           torch.exp(-((x_t - mu_x) ** 2 / (torch.add(2 * sigma_x ** 2, self.epsilon)) +
                                   (y_t - mu_y) ** 2 / (torch.add(2 * sigma_y ** 2 , self.epsilon))))

        
                max_gauss = torch.max(gaussian)
                gaussian = gaussian / max_gauss * pred[i]
                
                g_maps[idx,0,:,:]+=gaussian
        

        return g_maps
        
     def _linspace(self,start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        step = (stop - start) / (num - 1)
        return torch.arange(num) * step + start