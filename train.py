#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.autograd.functional import jacobian as J

from utils import *




# misc
device = 'cpu'
    
# keypoints / graph
d = 5 # 3 (for refinement stage)
N_P = 2048 # 3072 (for refinement stage)
k = 15

# displacements
l_max = 14 # 8 (for refinement stage)
l_width = l_max * 2 + 1
q = 2 # 1 (for refinement stage)

# model
base = 4
sigma2 = 1

# training
num_epochs = 150
init_lr = 0.1
save_iter = 10

# load data

D, H, W = 192, 160, 192


# displacement space

disp = torch.stack(torch.meshgrid(torch.arange(- q * l_max, q * l_max + 1, q * 2),
                                  torch.arange(- q * l_max, q * l_max + 1, q * 2),
                                  torch.arange(- q * l_max, q * l_max + 1, q * 2))).permute(1, 2, 3, 0).contiguous().view(1, -1, 3).float()

#! torch.arange(- q * l_max, q * l_max + 1, q * 2)
#!    -> torch.arange(-28, 29, 4) -> [-28, -24, -20, ... , 20, 24, 28], len 15 

#! stack(meshgrid(...)).permute(1,2,3,0) -> (15, 15, 15, 3)
#! permute() moves (3, 15, 15, 15) -> (15, 15, 15, 3)

#! disp (1, 3375, 3), 15**3 = 3375

# tensor.contiguous() -> Returns a contiguous in memory tensor containing the same data as self tensor

disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(device) 
#! (1, 3375, 3)


# graphregnet

class GaussianSmoothing(nn.Module):
    def __init__(self, sigma):
        super(GaussianSmoothing, self).__init__()
        
        sigma = torch.tensor([sigma]).to(device)
        N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
        weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
        weight /= weight.sum()
        
        self.weight = weight
        
    def forward(self, x):
        
        x = filter1D(x, self.weight, 0)
        x = filter1D(x, self.weight, 1)
        x = filter1D(x, self.weight, 2)
        
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, base=4):
        super(Encoder, self).__init__()
    
        self.conv_in = nn.Sequential(nn.Conv3d(in_channels, base, 3, stride=2, padding=1, bias=False),
                                     nn.InstanceNorm3d(base),
                                     nn.LeakyReLU())
        
        self.conv1 = nn.Sequential(nn.Conv3d(base, 2*base, 3, stride=2, padding=1, bias=False),
                                   nn.InstanceNorm3d(2*base),
                                   nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(nn.Conv3d(2*base, 4*base, 3, stride=2, padding=1, bias=False),
                                   nn.InstanceNorm3d(4*base),
                                   nn.LeakyReLU())
        
    def forward(self, x):
        
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        
        return x1, x2, x3
            
class Decoder(nn.Module):
    def __init__(self, out_channels=1, base=4):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv3d(4*base, 2*base, 3, stride=1, padding=1, bias=False),
                                   nn.InstanceNorm3d(2*base),
                                   nn.LeakyReLU())
    
        self.conv1a = nn.Sequential(nn.Conv3d(4*base, 2*base, 3, stride=1, padding=1, bias=False),
                                    nn.InstanceNorm3d(2*base),
                                    nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(nn.Conv3d(2*base, base, 3, stride=1, padding=1, bias=False),
                                   nn.InstanceNorm3d(base),
                                   nn.LeakyReLU())
        
        self.conv2a = nn.Sequential(nn.Conv3d(2*base, base, 3, stride=1, padding=1, bias=False),
                                    nn.InstanceNorm3d(base),
                                    nn.LeakyReLU())
        
        self.conv_out = nn.Sequential(nn.Conv3d(base, 1, 3, padding=1))
        
    def forward(self, x1, x2, x3):
        x = F.interpolate(x3, size=x2.shape[-3:], mode='trilinear')
        x = self.conv1(x)
        x = self.conv1a(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=x1.shape[-3:], mode='trilinear')
        x = self.conv2(x)
        x = self.conv2a(torch.cat([x, x1], dim=1))
        x = self.conv_out(x)
        return x
    
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
        self.conv = nn.Sequential(
            nn.Conv3d(self.in_channels * 2, self.out_channels, 1, bias=False),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x, ind):
        B, N, C, D, _, _ = x.shape
        k = ind.shape[2]
        
        y = x.view(B*N, C, D*D*D)[ind.view(B*N, k)].view(B, N, k, C, D*D*D)
        x = x.view(B, N, C, D*D*D).unsqueeze(2).expand(-1, -1, k, -1, -1)
        
        x = torch.cat([y - x, x], dim=3).permute(0, 3, 1, 2, 4)
        
        x = self.conv(x)
    
        x = x.mean(dim=3).permute(0, 2, 1, 3).view(B, N, -1, D, D, D)
        return x
    
class GCN(nn.Module):
    def __init__(self, base=4):
        super(GCN, self).__init__()
        
        self.base = base
        
        self.conv1 = EdgeConv(4*self.base + 3, 4*self.base)
        self.conv2 = EdgeConv(2*4*self.base + 3, 4*self.base)
        self.conv3 = EdgeConv(3*4*self.base + 3, 4*self.base)
        
    def forward(self, x1, x2, x3, kpts, ind):
        expand = x3.shape[-1]
        xa = self.conv1(torch.cat([x3, kpts.view(-1, 3, 1, 1, 1).expand(-1, -1, expand, expand, expand)], dim=1).unsqueeze(0), ind).squeeze(0)
        xb = self.conv2(torch.cat([torch.cat([x3, kpts.view(-1, 3, 1, 1, 1).expand(-1, -1, expand, expand, expand)], dim=1), xa], dim=1).unsqueeze(0), ind).squeeze(0)
        xc = self.conv3(torch.cat([torch.cat([x3, kpts.view(-1, 3, 1, 1, 1).expand(-1, -1, expand, expand, expand)], dim=1), xa, xb], dim=1).unsqueeze(0), ind).squeeze(0)
        return x1, x2, xc
    
class GraphRegNet(nn.Module):
    def __init__(self, base, smooth_sigma):
        super(GraphRegNet, self).__init__()
        
        self.base = base
        self.smooth_sigma = smooth_sigma
        
        self.pre_filter1 = GaussianSmoothing(self.smooth_sigma)
        self.pre_filter2 = GaussianSmoothing(self.smooth_sigma)
            
        self.encoder1 = Encoder(2, self.base)
        self.gcn1 = GCN(self.base)
        self.decoder1 = Decoder(1, self.base)
        
        self.encoder2 = Encoder(4, self.base)
        self.gcn2 = GCN(self.base)
        self.decoder2 = Decoder(1, self.base)
        
    def forward(self, x, kpts, kpts_knn):
        
        x1 = self.encoder1(torch.cat([x, self.pre_filter1(x)], dim=1))
            
        x1 = self.gcn1(*x1, kpts, kpts_knn)
        x1 = self.decoder1(*x1)
        
        x1 = F.interpolate(x1, size=x.shape[-3:], mode='trilinear')
        
        x2 = self.encoder2(torch.cat([x, self.pre_filter1(x), x1, self.pre_filter2(x1)], dim=1))
                
        x2 = self.gcn2(*x2, kpts, kpts_knn)
        x2 = self.decoder2(*x2)
        
        return x2
                           
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)



# differentiable sparse-to-dense supervision

class InverseGridSample(Function):
    
    @staticmethod
    def forward(ctx, input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=None):
        B, C, N = input.shape
        D = grid.shape[-1]
        device = input.device
        dtype = input.dtype
        
        ctx.save_for_backward(input, grid)
        
        if D == 2:
            input_view = [B, C, -1, 1]
            grid_view = [B, -1, 1, 2]
        elif D == 3:
            input_view = [B, C, -1, 1, 1]
            grid_view = [B, -1, 1, 1, 3]
            
        ctx.grid_view = grid_view
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        with torch.enable_grad():
            output = J(lambda x: InverseGridSample.sample(input.view(*input_view), grid.view(*grid_view), x, mode, padding_mode, align_corners), (torch.zeros(B, C, *shape).to(dtype).to(device)))

        return output

    @staticmethod
    def backward(ctx, grad_output):        
        input, grid = ctx.saved_tensors
        grid_view = ctx.grid_view
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        
        grad_input = F.grid_sample(grad_output, grid.view(*grid_view), mode, padding_mode, align_corners)
        
        return grad_input.view(*input.shape), None, None, None, None, None
        
    @staticmethod
    def sample(input, grid, accu, mode='bilinear', padding_mode='zeros', align_corners=None):
        sampled = F.grid_sample(accu, grid, mode, padding_mode, align_corners)
        return -0.5 * ((input - sampled) ** 2).sum()
    
def inverse_grid_sample(input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=None):
    return InverseGridSample.apply(input, grid, shape, mode, padding_mode, align_corners)

def densify(kpts, kpts_disp, shape, smooth_iter=3, kernel_size=5, eps=0.0001):
    '''
    Takes the keypoints, keypoint displacement and shape,
    returns the grid

    E.g.
    keypoints: (1, 2048, 3),
    keypoint displacement: (1, 2048, 3375, 3)
    shape: (64, 53, 64)
    returns grid: (1, 3, 64, 53, 64)
    '''

    B, N, _ = kpts.shape
    device = kpts.device
    D, H, W = shape
    
    grid = inverse_grid_sample(kpts_disp.permute(0, 2, 1), kpts, shape, padding_mode='border', align_corners=True)
    grid_norm = inverse_grid_sample(torch.ones(B, 1, N).to(device), kpts, shape, padding_mode='border', align_corners=True)
    
    avg_pool = nn.AvgPool3d(kernel_size, stride=1, padding=kernel_size // 2).to(device)
    for i in range(smooth_iter):
        grid = avg_pool(grid)
        grid_norm = avg_pool(grid_norm)
        
    grid = grid / (grid_norm + eps)
    
    return grid


# model
graphregnet = GraphRegNet(base, sigma2).to(device)
graphregnet.apply(init_weights)
parameter_count(graphregnet)

# optimizer
optimizer = optim.Adam(graphregnet.parameters(), init_lr)

# criterion
def criterion(feat_fixed, feat_moving, disp, mask):
    mse_loss = nn.MSELoss(reduction='none')
    loss = (mse_loss(feat_fixed, warp_img(feat_moving, disp.permute(0, 2, 3, 4, 1))) * mask).sum() / mask.float().sum()
    return loss

# statistics
losses = []

if True:

    # train mode
    graphregnet.train()
    
    # statistics
    running_loss = 0.0
    
    
    # for all training cases
    while True:
        
        # zero out gradients
        optimizer.zero_grad()
    
        # load data
        img_fixed = torch.rand(1, 1, D, H, W).to(device) #! (1, 1, 192, 160, 192). Same for the other three
        mask_fixed = torch.rand(1, 1, D, H, W).to(device)
        img_moving = torch.rand(1, 1, D, H, W).to(device)
        mask_moving = torch.rand(1, 1, D, H, W).to(device)

        # extract kpts and generate knn graph
        kpts_raw, kpts_fixed = foerstner_kpts(img_fixed, mask_fixed, d=d, num_points=N_P)
        #! (1, 2048, 3), (1, 2048, 3)
        kpts_fixed_knn = knn_graph(kpts_fixed, k, include_self=True)[0]
        # ! (1, 2048, 15)

        # extract mind features 
        mind_fixed = mindssc(img_fixed)
        #! (1, 12, 192, 160, 192)
        
        mind_moving = mindssc(img_moving) 
        #! (1, 12, 192, 160, 192)

        # displacement cost computation
        cost = ssd(kpts_fixed, mind_fixed, mind_moving, (D, H, W), l_max, q).view(-1, 1, l_width, l_width, l_width)
        #! (2048, 1, 29, 29, 29)
        # forward
        
        kpts_fixed_disp_pred = graphregnet(cost, kpts_fixed, kpts_fixed_knn)
        #! (2048, 1, 15, 15, 15)

        # sparse to dense
        disp_pred = densify(kpts_fixed, (disp.unsqueeze(1) * F.softmax(kpts_fixed_disp_pred.view(1, N_P, -1), 2).unsqueeze(3)).sum(2), (D//3, H//3, W//3)) 
        #! kpts_fixed (1, 2048, 3)
        #! (disp.unsqueeze(1) * F.softmax(kpts_fixed_disp_pred.view(1, N_P, -1), 2).unsqueeze(3)).shape (1, 2048, 3375, 3)
        #! (D//3, H//3, W//3) (64, 53, 64)
        
        #! disp_pred (1, 3, 64, 53, 64)
        
        disp_pred = F.interpolate(disp_pred, size=(D, H, W), mode='trilinear')
        #! (1, 3, 192, 160, 192)
        
        # loss
        loss = criterion(mind_fixed, mind_moving, disp_pred, mask_moving)
        #! scala
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()

        
    