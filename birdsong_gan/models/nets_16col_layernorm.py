import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


    
    
class _netG(nn.Module):
    def __init__(self, nz,ngf,nc):
        super(_netG, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.lnlist = nn.ModuleList([nn.LayerNorm([ngf * 8, 3, 3]),
                                     nn.LayerNorm([ngf * 4, 8, 4]),
                                     nn.LayerNorm([ngf * 2, 31, 5]),
                                     nn.LayerNorm([ngf, 63, 8]) ])
        self.convs = nn.ModuleList([nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(3,3), stride=1, padding=0, bias=False),
                                    # Hout = (Hin − 1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+outpad+1
                                    # size is H = 3, W = 3
                                    nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4, 4), stride=(3, 1),
                                                       padding=(1, 1), bias=False),
                                    # size is H = 8, W = 4
                                    nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(5, 4), stride=(4, 1),
                                                       padding=(1, 1), bias=False),
                                    # size is H = 31, W = 5
                                    nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0),
                                                       bias=False),
                                    # size is H = 63, W = 8
                                    nn.ConvTranspose2d(ngf, nc, kernel_size=(5, 4), stride=(2, 2), padding=(0,1),
                                                       bias=True)]
                                   # size is H = 129, W = (8-1)*2  -2 + 4
                                  )
        self.activation_last = nn.Softplus()
        self.activation = nn.ReLU(True)
        self.reconstruction = False

    def mode(self, reconstruction=False):
        self.reconstruction = reconstruction

    def forward(self, input):
        x = self.convs[0](input.view(input.size(0), self.nz, 1, 1))
        x = self.lnlist[0](x)
        for i in range(4):
            x = self.activation(x)
            x = self.convs[i + 1](x)
            if i < 3:
                x = self.lnlist[i + 1](x)
        x = self.activation_last(x)
        return x


class _netE(nn.Module):
    def __init__(self, nz,ngf,nc):
        super(_netE, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(nc, ngf, kernel_size=(5, 4), stride=(2, 2), padding=(0,1), bias=False),
            # H = ((129 -5)/2 + 1 = 63, W = (16 + 2 - 4)/2 + 1 = 8
            nn.LayerNorm([ngf, 63, 8]),
            nn.ReLU(True),
            
            nn.Conv2d(ngf, ngf * 2, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0), bias=False),
            # size H = (63 +2 -5)/2 +1 = 31, W = (8 - 4) + 1 = 5
            nn.LayerNorm([ngf * 2, 31, 5]),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(5, 4), stride=(4, 1), padding=(1, 1), bias=False),
            # size H = (31 +2 - 5)/4 + 1 = 8, W = (5 +2 -4) + 1 = 4
            nn.LayerNorm([ngf * 4, 8, 4]),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=(4, 4), stride=(3, 1), padding=(1, 1), bias=False),
            # size H = (8 +2 -4)/3 + 1 = 3, W = (4 +2 -4) + 1 = 3
            nn.LayerNorm([ngf * 8, 3, 3]),
            nn.ReLU(True),
            nn.Conv2d(ngf * 8, nz, kernel_size=(3,3), stride=1, padding=0, bias=False),
        )
        self.activation = nn.Sequential(
            nn.Linear(nz, nz * 2),
            nn.ReLU(True),
            nn.Linear(nz * 2 , nz)
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        return self.activation(output)

    
    
class _netD(nn.Module):
    def __init__(self, ndf,nc):
        super(_netD, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(nc, ndf, kernel_size=4, stride=(2,2), padding=(1,1), bias=False),
            # size H = (129 +2 -4)/2 + 1 = 64, W = (16 +2 -4)/2 + 1 = 8
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 3), stride=(2, 1), padding=(1, 0), bias=False),
            # size H = (64 +2 -4)/2 + 1 = 32, W = (8 -3) + 1 = 6
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4,3), stride=(2,1), padding=(2,0), bias=False),
            # size H = (32 +4 -4)/2 + 1 = 17, W = (6 -3) + 1 = 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(8,4), stride=(2,1), padding=1, bias=False),
            # H = (17 +2 -8)/2 + 1 = 6, W = (4 +2 -4) + 1 = 3
            nn.Conv2d(ndf * 8, 1, kernel_size=(6,3), stride=1, padding=0, bias=False),
            # H = 6-6 + 1 =1, W = (4 - 3) +1 =  
        ])
        self.lns = nn.ModuleList([nn.LayerNorm([ndf, 64, 8]),
                                  nn.LayerNorm([ndf * 2, 32, 6]),
                                  nn.LayerNorm([ndf * 4, 17, 4]),
                                  nn.LayerNorm([ndf * 8, 6, 3])])
        self.activations = nn.ModuleList([nn.LeakyReLU(0.2),nn.LeakyReLU(0.2),
                                    nn.LeakyReLU(0.2),nn.LeakyReLU(0.2)])

    def forward(self, x):
        for i in range(4):
            x = self.convs[i](x)
            x = self.lns[i](x)
            x = self.activations[i](x)
        x = self.convs[-1](x)
        x = nn.Sigmoid()(x)
        output = x.view(-1,1)
        return output

    

class InceptionNet(nn.Module):
    def __init__(self, ndf, nc):
        super(InceptionNet, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(nc, ndf, kernel_size=4, stride=(2,2), padding=(1,1), bias=False),
            # size H = (129 +2 -4)/2 + 1 = 64, W = (16 +2 -4)/2 + 1 = 8
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 3), stride=(2, 1), padding=(1, 0), bias=False),
            # size H = (64 +2 -4)/2 + 1 = 32, W = (8 -3) + 1 = 6
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4,3), stride=(2,1), padding=(2,0), bias=False),
            # size H = (32 +4 -4)/2 + 1 = 17, W = (6 -3) + 1 = 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(8,4), stride=(2,1), padding=1, bias=False),
            # H = (17 +2 -8)/2 + 1 = 6, W = (4 +2 -4) + 1 = 3
            nn.Conv2d(ndf * 8, 1, kernel_size=(6,3), stride=1, padding=0, bias=False),
            # H = 6-6 + 1 =1, W = (4 - 3) +1 =  
        ])
        self.lns = nn.ModuleList([nn.LayerNorm([ndf, 64, 8]),
                                  nn.LayerNorm([ndf * 2, 32, 6]),
                                  nn.LayerNorm([ndf * 4, 17, 4]),
                                  nn.LayerNorm([ndf * 8, 6, 3])])
        self.activations = nn.ModuleList([nn.LeakyReLU(0.2),nn.LeakyReLU(0.2),
                                    nn.LeakyReLU(0.2),nn.LeakyReLU(0.2)])

    def get_middle_layer(self, x):
        for i in range(4):
            x = self.convs[i](x)
            x = self.lns[i](x)
            x = self.activations[i](x)
        return x
    
    def fid_score(self, x_real, x_gen):
        x_real = self.get_middle_layer(x_real)
        # flatten its last dimensions
        x_real = x_real.view(-1, x_real.size(1)*x_real.size(2)*x_real.size(3))
        
        x_gen = self.get_middle_layer(x_real)
        # flatten its last dimensions
        x_gen = x_gen.view(-1, x_real.size(1)*x_real.size(2)*x_real.size(3))
        
        return FID(x_real, x_gen)
    
    def forward(self, x):
        x = self.get_middle_layer(x)
        x = self.convs[-1](x)
        x = nn.Sigmoid()(x)
        output = x.view(-1,1)
        return output
    
    
    
class GANLoss_orig(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss_orig, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    

    
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                real_tensor.requires_grad = False
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                fake_tensor.requires_grad=False
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)