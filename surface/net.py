import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class LinearInterpolator(nn.Module):
    """
    Differentiable linear volume-to-surface interpolation.
    
    Args:
    - inshape: input shape, (D1,D2,D3)
    - device: [cpu, cuda]
    
    Inputs:
    - vert: mesh vertices, (B,|V|,3)
    - vol: 3D volume, (B,C,D1,D2,D3)
    
    Returns:
    - value: interpolated values, (B,|V|,C)
    """
    
    def __init__(self, inshape=[112,224,160], device='cpu'):
        super(LinearInterpolator, self).__init__()
        linear_grid = [torch.arange(0,2) for s in range(3)]
        linear_grid = torch.meshgrid(linear_grid, indexing='ij')
        self.linear_grid = torch.stack(
            linear_grid).unsqueeze(0).to(device)
        self.min_bound = torch.Tensor(
            [1e-5, 1e-5, 1e-5]).to(device)
        self.max_bound = torch.Tensor(
            inshape).to(device) - 1 - 1e-5

    def forward(self, vert, vol):
        v = vert.clamp(min=self.min_bound, max=self.max_bound)
        vid = v[0].floor().long()
        u = v - vid
        
        # sample voxel values for interpolation
        vid = vid[...,None,None,None] + self.linear_grid
        value = vol[0,:,vid[:,0], vid[:,1], vid[:,2]]

        # 3D interpolation
        value = (1-u[:,:,0])[...,None,None] * value[:,:,0] +\
                u[:,:,0][...,None,None] * value[:,:,1]
        value = (1-u[:,:,1])[...,None] * value[:,:,0] +\
                u[:,:,1][...,None] * value[:,:,1]
        value = (1-u[:,:,2]) * value[:,:,0] +\
                u[:,:,2] * value[:,:,1]

        value = value.t()[None]
        return value
        
        

class CubicInterpolator(nn.Module):
    """
    Differentiable cubic volume-to-surface interpolation 
    by cubic convolution algorithm [1].
    
    [1] Keys, Robert. "Cubic convolution interpolation for
    digital image processing." IEEE transactions on acoustics,
    speech, and signal processing.
    
    Args:
    - inshape: input shape, [D1,D2,D3]
    - device: [cpu, cuda]
    
    Inputs:
    - vert: mesh vertices, (B,|V|,3)
    - vol: 3D volume, (B,C,D1,D2,D3)
    
    Returns:
    - value: interpolated values, (B,|V|,C)
    """
    
    def __init__(self, inshape=[112,224,160], device='cpu'):
        super(CubicInterpolator, self).__init__()
        cubic_grid = [torch.arange(-1,3) for s in range(3)]
        cubic_grid = torch.meshgrid(cubic_grid, indexing='ij')
        self.cubic_grid = torch.stack(
            cubic_grid).unsqueeze(0).to(device)
        self.min_bound = torch.Tensor(
            [1.0+1e-5, 1.0+1e-5, 1.0+1e-5]).to(device)
        self.max_bound = torch.Tensor(
            inshape).to(device) - 2 - 1e-5
        # fixed kernel for interpolation
        self.kernel = 0.5*torch.Tensor(
            [[0, 2, 0, 0], [-1,0,1,0],
             [2, -5, 4, -1], [-1,3,-3,1]]).to(device)

    def forward(self, vert, vol):
        v = vert.clamp(min=self.min_bound, max=self.max_bound)
        vid = v[0].floor().long()
        u = (v - vid)[...,None]
        # distance to voxels
        u = torch.cat(
            [torch.ones_like(u), u, u**2, u**3], dim=-1)

        # sample voxel values for interpolation
        vid = vid[...,None,None,None] + self.cubic_grid
        value = vol[0,:,vid[:,0], vid[:,1], vid[:,2]]
        
        # 3D interpolation
        value = ((u[:,:,0,:] @ self.kernel)[...,None,None] * value).sum(2)
        value = ((u[:,:,1,:] @ self.kernel)[...,None] * value).sum(2)
        value = ((u[:,:,2,:] @ self.kernel) * value).sum(2)
        value = value.t()[None]
        return value
    
    

class GaussianFilter(nn.Module):
    """
    Differentiable Gaussian Filter.
    
    Args:
    - C: number of channels
    - K: filter size
    - sigma: standard deviation of gaussian kernel
    - device: [cpu, cuda]
    
    Inputs:
    - x: input features, (B,C,D1,D2,D3)
    
    Returns:
    - x: smoothed features, (B,C,D1,D2,D3)
    """
    
    def __init__(self, C=3, K=3, sigma=0.5, device='cpu'):
        super(GaussianFilter, self).__init__()
        mesh_grids = torch.meshgrid(
            [torch.linspace(-(K-1)/2, (K-1)/2, K)] * 3, indexing='ij')
        kernel = 1.
        for grid in mesh_grids:
            kernel *= 1. / (sigma * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((grid / sigma) ** 2 / 2))
            kernel = kernel / torch.sum(kernel)

        kernel = kernel[None,None].repeat(C,1,1,1,1)
        self.kernel = kernel.to(device)
        self.K = K
        self.C = C
        
    def forward(self, x):
        x = F.conv3d(
            x, weight=self.kernel, padding=self.K//2, groups=self.C)
        return x

    

class MUNet(nn.Module):
    """
    Multiscale U-Net that predicts multiscale SVFs.
    
    Args:
    - C_in: input channels 
    - C_hid: hidden channels
    - K: kernel size
    
    Inputs: 
    - x: 3D volume, (B,C,D1,D2,D3) torch.Tensor

    Returns:
    - SVF1, SVF2, SVF3, SVF4: multiscale stationary velocity fields (SVFs),
    (B,3,D1,D2,D3) torch.Tensor
    """
    def __init__(self, C_in=1, C_hid=[8,16,32,32,32,32], K=3):
        super(MUNet, self).__init__()

        
        # convolutional encoder
        self.conv1 = nn.Conv3d(in_channels=C_in, out_channels=C_hid[0],
                               kernel_size=K, stride=1, padding=K//2)
        self.conv2 = nn.Conv3d(in_channels=C_hid[0], out_channels=C_hid[1],
                               kernel_size=K, stride=1, padding=K//2)
        self.conv3 = nn.Conv3d(in_channels=C_hid[1], out_channels=C_hid[2],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv4 = nn.Conv3d(in_channels=C_hid[2], out_channels=C_hid[3],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv5 = nn.Conv3d(in_channels=C_hid[3], out_channels=C_hid[4],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv6 = nn.Conv3d(in_channels=C_hid[4], out_channels=C_hid[5],
                               kernel_size=K, stride=1, padding=K//2)
        
        # convolutional decoder
        self.deconv5 = nn.Conv3d(in_channels=C_hid[5]+C_hid[4],
                                 out_channels=C_hid[4], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv4 = nn.Conv3d(in_channels=C_hid[4]+C_hid[3],
                                 out_channels=C_hid[3], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv3 = nn.Conv3d(in_channels=C_hid[3]+C_hid[2],
                                 out_channels=C_hid[2], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv2 = nn.Conv3d(in_channels=C_hid[2]+C_hid[1], 
                                 out_channels=C_hid[1], kernel_size=K,
                                 stride=1, padding=K//2)
        self.deconv1 = nn.Conv3d(in_channels=C_hid[1]+C_hid[0],
                                 out_channels=C_hid[0], kernel_size=K,
                                 stride=1, padding=K//2)
        
        # velocity fields prediction
        self.flow1 = nn.Conv3d(in_channels=C_hid[3], out_channels=3,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow2 = nn.Conv3d(in_channels=C_hid[2], out_channels=3,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow3 = nn.Conv3d(in_channels=C_hid[1], out_channels=3,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow4 = nn.Conv3d(in_channels=C_hid[0], out_channels=3,
                               kernel_size=K, stride=1, padding=K//2)
        
        nn.init.normal_(self.flow1.weight, 0, 1e-5)
        nn.init.constant_(self.flow1.bias, 0.0)
        nn.init.normal_(self.flow2.weight, 0, 1e-5)
        nn.init.constant_(self.flow2.bias, 0.0)
        nn.init.normal_(self.flow3.weight, 0, 1e-5)
        nn.init.constant_(self.flow3.bias, 0.0)
        nn.init.normal_(self.flow4.weight, 0, 1e-5)
        nn.init.constant_(self.flow4.bias, 0.0)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                              align_corners=False)
        
    def forward(self, x):
        # encode
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x5 = F.leaky_relu(self.conv5(x4), 0.2)
        x  = F.leaky_relu(self.conv6(x5), 0.2)

        # decode
        x = torch.cat([x, x5], dim=1)
        x = F.leaky_relu(self.deconv5(x), 0.2)

        x = self.up(x)
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        svf1 = self.up(self.up(self.flow1(x)))
        
        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        svf2 = self.up(self.flow2(x))
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        svf3 = self.flow3(x)

        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        svf4 = self.flow4(x)
        return svf1, svf2, svf3, svf4


        
class MeshDeform(nn.Module):
    """
    Mesh diffeomorphic deformation network for 
    cortical surface reconstruction.
    
    Args:
    - C_in: input channels 
    - C_hid: hidden channels
    - sigma: standard deviation of gaussian kernel
    - interpolation: type of interpolation, ['trilinear', 'tricubic']
    - device: [cpu, cuda]
    
    Inputs: 
    Inputs:
    - vert: mesh vertices, (B,|V|,3)
    - vol: 3D volume, (B,C,D1,D2,D3)
    - n_steps: number of scaling & squaring steps

    Returns:
    - vert: predicted mesh vertices, (B,|V|,3)
    """
    
    def __init__(self, C_in=1,
                 C_hid=[16,32,32,32,32],
                 inshape=[112,224,160],
                 sigma=0.5,
                 interpolation='trilinear',
                 device='cpu'):
        
        super(MeshDeform, self).__init__()
        self.munet = MUNet(C_in=C_in, C_hid=C_hid).to(device)
        self.scale = torch.Tensor(inshape).to(device)
        grid = [torch.arange(0, s) for s in inshape]
        grid = torch.stack(torch.meshgrid(grid, indexing='ij'))
        self.grid = grid[None].to(device)
        self.gaussian = GaussianFilter(sigma=sigma, device=device)
        if interpolation == 'trilinear':
            interpolator = LinearInterpolator
        elif interpolation == 'tricubic':
            interpolator = CubicInterpolator
        self.interp = interpolator(inshape=inshape, device=device)
        
    def forward(self, vert, vol, n_steps=7):
        # predict multiscale velocity fields
        svfs = self.munet(vol)
        
        # multiscale deformation
        for n in range(len(svfs)):
            # integrate to deformation field
            phi_n = self.integrate(svfs[n], n_steps=n_steps)
            phi_n = self.gaussian(phi_n)  # gaussian smooth
            # displace vertices
            deform = self.interp(vert, phi_n)
            vert = vert + deform
        return vert
    
    def integrate(self, svf, n_steps=7):
        flow = svf / (2 ** n_steps)
        for n in range(n_steps):
            flow = flow + self.transform(flow, flow)
        return flow #+ self.grid
    
    def transform(self, src, flow):
        # new coordinates
        coord = self.grid + flow

        # normalize location to [-1,1]
        for i in range(3):
            coord[:,i] = 2*coord[:,i,] / (self.scale[i] - 1) - 1
        coord = coord.permute(0,2,3,4,1).flip(-1)
        out = F.grid_sample(
            src, coord, align_corners=True, mode='bilinear')
        return out
