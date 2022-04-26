
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class SalGraphLoss(nn.Module):
    def __init__(self):
        super(SalGraphLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.kl_div_1=nn.KLDivLoss()
        self.kl_div_2=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        self.bce=nn.BCELoss()
        self.sigmoid=nn.Sigmoid()
        self.eps=1e-10

    def forward(self, output, target_map,output_node, target_node):
        loss=0
        #target_map=torch.unsqueeze(target_map,dim=1)
        output_norm=output/ (torch.sum(output,(2,3),keepdim=True) + self.eps)
        target_norm=target_map/ (torch.sum(target_map,(2,3),keepdim=True) + self.eps)
        
        loss += 2.0 * self.kl_div_1(output_norm.log(),target_norm)
        loss += 1.0 * self.cc(output,target_map)
        loss += 2.0 * self.kl_div_2(output_node,target_node)

        return loss


class MyCorrCoef(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyCorrCoef, self).__init__()

    def forward(self, pred, target):

        size = pred.size()
        new_size = (-1, size[-1] * size[-2])
        pred = pred.reshape(new_size)
        target = target.reshape(new_size)
    
        cc = []
        for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
            xm, ym = x - x.mean(), y - y.mean()
            r_num = torch.mean(xm * ym)
            r_den = torch.sqrt(
                torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
            r = -1.0*r_num / r_den
            cc.append(r)
    
        cc = torch.stack(cc)
        cc = cc.reshape(size[:2]).mean()
        return cc  # 1 - torch.square(r)
