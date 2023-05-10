# From the author's GitHub
# Modified and updated for file paths by Price

import nibabel as nib
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

from pathlib import Path
import pdb

from utils import *
from models import *

"""
Start Interactive Session with:

sinteractive -p gpu --mem 64G --gres=gpu:V100 --time=0-12:00
source /physical_sciences/Lachlan/venvs/test1/bin/activate
cd /physical_sciences/GraphRegNet_PMCC
python train_pj_v3_192.py
"""

# settings

# data
# cases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
# fold = 0
# if fold == 0:  
#     test_cases = [0, 5, 10, 15, 20, 25, 30, 35, 40]
# elif fold == 1:
#     test_cases = [1, 6, 11, 16, 21, 26, 31, 36, 41]
# elif fold == 2:
#     test_cases = [2, 7, 12, 17, 22, 27, 32, 37, 42]
# elif fold == 3:
#     test_cases = [3, 8, 13, 18, 23, 28, 33, 38, 43]
# elif fold == 4:
#     test_cases = [4, 9, 14, 19, 24, 29, 32, 39]

cases = [0]
fold = 0
test_cases = [3]

train_cases = [i for i in cases if not i in test_cases]

# misc
device = 'cuda'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
model_dir = 'fold{}'.format(fold)

phys_dir = Path('/physical_sciences/GraphRegNet_PMCC')
data_dir = phys_dir / 'data_hifive_bbox_crop_and_res192'
model_dir = Path(f'./models192/{fold}')
model_dir.mkdir(exist_ok=True, parents=True)

# CONFIGS
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
imgs_fixed = {}
masks_fixed = {}
imgs_moving = {}
masks_moving = {}

def nii2tensor(fname):
    nib_obj = nib.load(fname)
    tensor = torch.from_numpy(nib_obj.get_fdata())
    tensor = tensor.view((1,1) + tensor.shape)
    return tensor
    
for case in train_cases:
    print('loading case {} ...'.format(case + 1), end=' ')
    t0 = time.perf_counter()
    fname = str(case+1).zfill(4)
    input_img_fixed = data_dir / f'case{fname}_img_fixed.nii.gz'
    input_mask_fixed = data_dir / f'case{fname}_mask_fixed.nii.gz'
    input_img_moving = data_dir / f'case{fname}_img_moving.nii.gz'
    input_mask_moving = data_dir / f'case{fname}_mask_moving.nii.gz'
    img_fixed = (nii2tensor(input_img_fixed).float().clamp_(-1000, 1500) + 1000) / 2500
    mask_fixed = nii2tensor(input_mask_fixed).bool()
    img_moving = (nii2tensor(input_img_moving).float().clamp_(-1000, 1500) + 1000) / 2500
    mask_moving = nii2tensor(input_mask_moving).bool()
    imgs_fixed[case] = img_fixed
    masks_fixed[case] = mask_fixed
    imgs_moving[case] = img_moving
    masks_moving[case] = mask_moving
    t1 = time.perf_counter()
    print('{:.2f} s'.format(t1-t0))

*_, D, H, W = imgs_fixed[train_cases[0]].shape

# displacement space

disp = torch.stack(
    torch.meshgrid(
        torch.arange(- q * l_max, q * l_max + 1, q * 2),
        torch.arange(- q * l_max, q * l_max + 1, q * 2),
        torch.arange(- q * l_max, q * l_max + 1, q * 2))
    ).permute(1, 2, 3, 0).contiguous().view(1, -1, 3).float()

disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(device)
                           
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)
    


# training

# model
graphregnet = GraphRegNet(base, sigma2).to(device)
graphregnet.apply(init_weights)
parameter_count(graphregnet)

# optimizer
optimizer = optim.Adam(graphregnet.parameters(), init_lr)

# loss = criterion(mind_fixed, mind_moving, disp_pred, mask_moving)
# criterion
def criterion(feat_fixed, feat_moving, disp, mask):
    mse_loss = nn.MSELoss(reduction='none')
    loss = (mse_loss(feat_fixed, warp_img(feat_moving, disp.permute(0, 2, 3, 4, 1))) * mask).sum() / mask.float().sum()
    return loss

# statistics
losses = []
torch.cuda.synchronize()
t0 = time.perf_counter()

# for num_epochs epochs
for epoch in range(num_epochs):

    # train mode
    graphregnet.train()
    
    # statistics
    running_loss = 0.0
    
    # shuffle training cases
    train_cases_perm = random.sample(train_cases, len(train_cases))
    
    # for all training cases
    for case in train_cases_perm:
        
        # zero out gradients
        optimizer.zero_grad()
    
        # load data
        img_fixed = imgs_fixed[case].to(device)
        mask_fixed = masks_fixed[case].to(device)
        img_moving = imgs_moving[case].to(device)
        mask_moving = masks_moving[case].to(device)

        # extract kpts and generate knn graph
        kpts_raw, kpts_fixed = foerstner_kpts(img_fixed, mask_fixed, d=d, num_points=N_P)
        kpts_fixed_knn = knn_graph(kpts_fixed, k, include_self=True)[0]


        # Save the feature maps ##########
        kpts_dir = Path(f'kpts')
        kpts_dir.mkdir(parents=True, exist_ok=True)
        kpts_raw = kpts_raw.squeeze().int().tolist()
        nii = nib.load(data_dir / f'case{str(case+1).zfill(4)}_img_fixed.nii.gz')
        points = np.zeros_like(nii.get_fdata())
        for x, y, z in kpts_raw:
            points[x, y, z] = 1
        kpts_nii = nib.Nifti1Image(points, nii.affine, nii.header)
        nib.save(kpts_nii, kpts_dir/f'key_point_epoch_{epoch}.nii')
        ##################################
        
        # extract mind features 
        mind_fixed = mindssc(img_fixed)
        mind_moving = mindssc(img_moving)

        # displacement cost computation
        cost = ssd(kpts_fixed, mind_fixed, mind_moving, (D, H, W), l_max, q).view(-1, 1, l_width, l_width, l_width)
        
        # forward
        kpts_fixed_disp_pred = graphregnet(cost, kpts_fixed, kpts_fixed_knn)
        
        # sparse to dense
        disp_pred = densify(kpts_fixed, (disp.unsqueeze(1) * F.softmax(kpts_fixed_disp_pred.view(1, N_P, -1), 2).unsqueeze(3)).sum(2), (D//3, H//3, W//3))
        disp_pred = F.interpolate(disp_pred, size=(D, H, W), mode='trilinear')
        
        # loss
        loss = criterion(img_fixed, img_moving, disp_pred, mask_moving)
        # loss = criterion(mind_fixed, mind_moving, disp_pred, mask_moving)
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()

        

    running_loss /= (len(train_cases))
    losses.append(running_loss)
    print(epoch+1,running_loss)
        
    if ((epoch + 1) % save_iter) == 0:
    
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        print('epoch: ', epoch + 1)
        print('loss: {:.4f}'.format(running_loss))
        print('time (epoch): {:.4f} s'.format((t1 - t0) / save_iter))
        gpu_usage()
        print('---')
        
        torch.save(graphregnet.cpu().state_dict(), model_dir / f'epoch{epoch}.pth')
        
        graphregnet.to(device)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()

torch.save(graphregnet.cpu().state_dict(), model_dir / 'final.pth')
