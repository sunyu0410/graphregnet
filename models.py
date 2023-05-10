import torch
import torch.nn as nn
from torch.autograd import Function

from utils import *
from torch.autograd.functional import jacobian as J

device = 'cuda'

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