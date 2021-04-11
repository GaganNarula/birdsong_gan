import torch
from torch import nn
import numpy as np
        
class Resblock_down(nn.Module):
    def __init__(self, nz, W, H, nlayers = 3, ks = (3,3)):
        super(Resblock_down, self).__init__()
        self.nz = nz
        self.nlayers = nlayers
        self.convs = []
        self.lnormlist = []
        for n in range(nlayers):
            if n == 0:
                self.convs.append(nn.Conv2d(nz, nz, kernel_size = ks, stride = 1,
                                            padding = (ks[0]//2, ks[1]//2),
                                            bias = True))
            else:
                self.convs.append(nn.Conv2d(nz, nz, kernel_size = ks, stride = 1,
                                            padding = (ks[0]//2, ks[1]//2),
                                            bias = True))
            self.lnormlist.append(nn.LayerNorm([nz, H, W]))
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        for n in range(self.nlayers):
            if n == 0:
                z = self.convs[n](x)
            else:
                z = self.convs[n](z)
            z = self.lnormlist[n](z)
            z = self.relu(z)
        return z + x


class Resblock_up(nn.Module):
    def __init__(self, nz, W, H, nlayers = 3, ks = (3,3)):
        super(Resblock_up, self).__init__()
        self.nz = nz
        self.nlayers = nlayers
        self.convs = []
        self.lnormlist = []
        for n in range(nlayers):
            if n == 0:
                self.convs.append(nn.ConvTranspose2d(nz, nz, kernel_size = ks, stride = 1,
                                            padding = (ks[0]//2, ks[1]//2),
                                            bias = True))
            else:
                self.convs.append(nn.ConvTranspose2d(nz, nz, kernel_size = ks, stride = 1,
                                            padding = (ks[0]//2, ks[1]//2),
                                            bias = True))
            self.lnormlist.append(nn.LayerNorm([nz, H, W]))
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        for n in range(self.nlayers):
            if n == 0:
                z = self.convs[n](x)
            else:
                z = self.convs[n](z)
            z = self.lnormlist[n](z)
            z = self.relu(z)
        return z + x