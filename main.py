import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch_geometric.data import Dataset, DataLoader
import torch_geometric

from lib.dataset.bdda import BDDA_OBJ
from lib.model.res_gcn_flow import RES_GCN
from lib.core.loss import SalGraphLoss,SalKLCCLoss
from saliency_metrics import *

from PIL import Image
import cv2


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training
doPrediction= True # Only prediction, no testing no training #hzx added

workers = 16
epochs = 200
#batch_size = torch.cuda.device_count()*32 # Change if out of cuda memory
batch_size=16

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 0.5
lr = base_lr

count_test = 0
count = 0



def main():
    global args, best_prec1, weight_decay, momentum
    
    
    cudnn.benchmark = True 
    
    model = RES_GCN()
    
    criterion = SalGraphLoss().cuda()
   
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    model = torch.nn.DataParallel(model,device_ids=[0]).cuda()  
    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state,strict=True)
            except:
                model.load_state_dict(state,strict=True)
            
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')
    
    
    dataTrain = BDDA_OBJ(split='train')
    dataVal = BDDA_OBJ(split='test')
   
    train_loader = torch_geometric.loader.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers)

    val_loader = torch_geometric.loader.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers)

    # Quick test
    if doTest:
        validate(val_loader, model, criterion, epoch)
        return

    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch ,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        break


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input_img,target_map,_,img_flow,box_node,_) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        input_img=input_img.cuda()
        target_map = target_map.cuda()#.float()
        box_node=box_node.cuda()
        img_flow=img_flow.cuda()
        

        outputs,output_nodes = model(input_img,img_flow,box_node)
        
        loss = criterion(outputs, target_map,output_nodes,box_node.y)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input_img.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i%100==0:
            print('Epoch (train): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))

def validate(val_loader, model, criterion, epoch):
    #global count_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc=AverageMeter()
    cc=AverageMeter()
    nss=AverageMeter()
    kl=AverageMeter()
    ig=AverageMeter()
    sim=AverageMeter()
    
    
    auc_all=[]
    cc_all=[]
    nss_all=[]
    kl_all=[]
    ig_all=[]
    sim_all=[]
    
    end = time.time()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
         for i, (input_img, target_map,target_map_mean,img_flow,box_node,img_id) in enumerate(val_loader):
    
            input_img=input_img.cuda()
            target_map = target_map.cuda()
            box_node=box_node.cuda()
            img_flow=img_flow.cuda()
            target_map_mean=target_map_mean.cuda()
            
            input_img = torch.autograd.Variable(input_img, requires_grad = False)
            target_map = torch.autograd.Variable(target_map, requires_grad = False)
            img_flow = torch.autograd.Variable(img_flow, requires_grad = False)
            
            # compute output
            outputs,output_nodes = model(input_img,img_flow,box_node)
            
            loss = criterion(outputs, target_map,output_nodes,box_node.y)
            #loss = criterion(outputs, target_map)
            

            num_images = input_img.size(0)
            losses.update(loss.item(), num_images)
            
            
            for idx in range(outputs.shape[0]):
                
                kl_avg=cal_kldiv_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]))
                cc_avg=cal_cc_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]))
                ig_avg=cal_infogain_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]),torch.squeeze(target_map_mean[idx,:,:,:]))
                nss_avg=cal_nss_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]))
                sim_avg=cal_similarity_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]))
                
                cnt=1
                cc.update(cc_avg, cnt)
                kl.update(kl_avg,cnt)
                ig.update(ig_avg,cnt)
                sim.update(sim_avg, cnt)
                nss.update(nss_avg, cnt)
                
                
                #auc_all.append(auc_avg)
                cc_all.append(cc_avg)
                nss_all.append(nss_avg)
                kl_all.append(kl_avg)
                ig_all.append(ig_avg)
                sim_all.append(sim_avg)
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % 100 == 0:
                print('Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'KLDiv {kl.val:.4f} ({kl.avg:.4f})\t' \
                      'Accuracy {cc.val:.3f} ({cc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          kl=kl, cc=cc))
                
    perf_indicator=cc.avg
    
    #print('auc_j: ',auc.avg)
    print('cc: ',cc.avg)
    print('kldiv:', kl.avg)
    print('nss: ',nss.avg)
    print('ig: ',ig.avg)
    print('sim: ',sim.avg)
    
    return perf_indicator

CHECKPOINTS_PATH = './output/'

def load_checkpoint(filename='best_checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    
    main()
    print('DONE')
