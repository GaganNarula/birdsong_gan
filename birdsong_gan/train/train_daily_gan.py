import torch
import torch.nn as nn
from configs.cfg import *
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from data.dataset import bird_dataset
import itertools
from models.nets_16col_residual import _netD, _netE, _netG, InceptionNet, weights_init


# some cuda / cudnn settings for memory issues#
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.deterministic = True
cudnn.benchmark = True
#cudnn.allow_tf32 = True

def make_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    dirs = os.listdir(path)
    for d in dirs:
        if len(os.listdir(os.path.join(path, d)))<=3:
            try:
                os.rmdir(os.path.join(path, d))
            except:
                shutil.rmtree(os.path.join(path,d))
    path += str(datetime.now()).replace(':', '-')
    path = path.replace(' ','_')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,'losses'))
        os.makedirs(os.path.join(path, 'hist'))
    return path
    


# useful to flip labels randomly
def true_wp(prob, size, device):
    # generate a uniform random number
    p = torch.rand(size).to(device).float()
    # if prob = 0.9, most of the time, this will be True
    p = (p < prob).float()
    return p 


class Model:
    """Model container for the GAN.
    """

    def __init__(self, path_to_bird_hdf, opts):
        
        self.opts = opts

        self.dataset = bird_dataset(path_to_bird_hdf)

        self.nz = opts['self.nz']
        ngf = opts['ngf']
        ndf = opts['ndf']
        nc = opts['nc']
        logpt = opts['log_every']

        if opts['cuda']:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # custom weights initialization called on networks
        self.netG = _netG(self.nz, ngf, nc)
        self.netG.apply(weights_init)
        if self.opts['netGpath'] != '':
            self.netG.load_state_dict(torch.load(opts['netG']))

        self.netD1 = _netD(ndf,nc)
        self.netD1.apply(weights_init)
        if self.opts['netD1path'] != '':
            self.netD1.load_state_dict(torch.load(opts['netD1']))

        self.netD2 = _netD(ndf,nc)
        self.netD2.apply(weights_init)
        if self.opts['netD2path'] != '':
            self.netD2.load_state_dict(torch.load(opts['netD2']))

        # inception net
        self.netD3 = InceptionNet(ndf,nc)
        self.netD3.apply(weights_init)
        if self.opts['netD3path'] != '':
            self.netD3.load_state_dict(torch.load(opts['netD2']))

        self.netE = _netE(self.nz, ngf, nc)
        self.netE.apply(weights_init)
        if self.opts['netEpath'] != '':
            self.netE.load_state_dict(torch.load(opts['netE']))

        if self.opts['cuda']:
            self.netD1 = self.netD1.to(self.device)
            self.netD2 = self.netD2.to(self.device)
            self.netD3 = self.netD3.to(self.device)
            self.netG = self.netG.to(self.device)
            self.netE = self.netE.to(self.device)

        # setup optimizers
        self.optimizerD1 = optim.Adam(self.netD1.parameters(), lr = opts['lr'], 
                                betas = (opts['beta1'], 0.999))
        self.optimizerD2 = optim.Adam(self.netD2.parameters(), lr = opts['lr'], 
                                betas = (opts['beta1'], 0.999))
        self.optimizerD3 = optim.Adam(self.netD3.parameters(), lr = opts['lr'], 
                                betas = (opts['beta1'], 0.999))
        self.optimizerG = optim.Adam(itertools.chain(self.netG.parameters(),
                                            self.netE.parameters()), lr = opts['lr'],
                                            betas = (opts['beta1'], 0.999))
        # for mds loss
        if self.opts['mds_loss']:
            self.optimizerE = optim.Adam(self.netE.parameters(), lr = opts['lr'], 
                                        betas = (opts['beta1'], 0.999))

        # Define loss function
        self.criterion_gan = nn.BCELoss()
        if opts['distance_fun']=='L1':
            print('Using L1 loss')
            self.criterion_dist = nn.L1Loss()
        else:
            print('Using L2 loss')
            self.criterion_dist = nn.MSELoss()


        # downsample function for reconstruction error 
        self.downsample_pth = torch.nn.AvgPool2d(3, stride=4)

        
        # probability that a label is wrongly labelled
        self.d_prob = torch.FloatTensor([1. - opts['d_noise']]).to(self.device)

    def make_dataloader(self, day=0):

        TD, age = self.dataset.make_chunk_tensor_dataset(day, imageW=self.opts['imageW'],
                                                        shuffle_chunks=True)


    def train_one_day(self, traindataloader):
        
        #losses
        minibatchLossD1 = []
        minibatchLossG1_rec = []
        minibatchLossG1_gan = []
        minibatchLossD2 = []
        minibatchLossG2 = []
        minibatchLossD3 = []
        FID = []
    
        # noise variable
        noise = torch.FloatTensor(self.opts['batchSize'], self.nz, 1,1).to(self.device)
        
        # training loop
        for epoch in range(self.opts['niter']):

            for i, data in enumerate(traindataloader):
                
                data = data[0] # pytorch produces tuples by default
                data = data.view(data.size(0),1,data.size(1),data.size(2)).to(self.device)

                # map data X -> Z latent
                encoding = self.netE(data)
                # map Z -> Xhat
                reconstruction = self.netG(encoding)

                
                # map  Xhat -> class [0: Fake, 1: Real] (Make discriminator give it value 1)
                pred_rec_d1 = self.netD1(reconstruction.detach())
                # map X -> class (maximize D)
                pred_real_d1 = self.netD1(data)

                # For discriminator, the Pr(class=1|X) = 0.9, true_wp = label with that probability
                err_real_d1 = self.criterion_gan(pred_real_d1, true_wp(self.d_prob,pred_real_d1.size(),
                                                self.device))

                # For disc, probability this is a reconstruction, the Pr(class=1| Xhat) = 0.1 = d_noise
                err_fake_d1 = self.criterion_gan(pred_rec_d1, true_wp(1.-self.d_prob,pred_real_d1.size(),
                                                self.device))
                
                err_d1 = err_real_d1 + err_fake_d1

                self.optimizerD1.zero_grad()
                err_d1.backward()
                # minimize  -logD(X) and maximize -log(D(Xhat)) only w.r.t Discriminator params!
                self.optimizerD1.step()

                # map Xhat -> class
                pred_rec_d1 = self.netD1(reconstruction)
                labell = torch.FloatTensor(self.opts['batchSize'],1).fill_(1.).to(self.device) # true label
                errG_discrim = self.criterion_gan(pred_rec_d1, labell)
                errG_recon = (self.criterion_dist(reconstruction, data) + \
                              self.criterion_dist(self.downsample_pth(reconstruction),
                              self.downsample_pth(data))) * self.opts['lambdaa']

                err_g_d1 = errG_discrim + errG_recon

                self.optimizerG.zero_grad()
                err_g_d1.backward()
                # maximize log D(Xhat) or minimize -log D(Xhat) + MSE for encoder and generator
                self.optimizerG.step()

                # ------------- Diffusion step ---------------

                encoding = self.netE(data)
                reconstruction = self.netG(encoding)
                pred_rec_d2 = self.netD2(reconstruction.detach())
                err_real_d2 = self.criterion_gan(pred_rec_d2, true_wp(self.d_prob,pred_real_d1.size(),self.device))

                
                if self.opts['noise_dist'] == 'normal':
                    noise.normal_(0., self.opts['z_var'])
                else:
                    noise.uniform_(-self.opts['z_var'],self.opts['z_var'])

                # Make Discriminator D2 learn difference between 
                fake = self.netG(noise)
                pred_fake_d2 = self.netD2(fake.detach())
                err_fake_d2 = self.criterion_gan(pred_fake_d2, true_wp(1.-self.d_prob,pred_real_d1.size(),self.device))
                err_d2 = err_real_d2 + err_fake_d2

                self.optimizerD2.zero_grad()
                err_d2.backward()
                self.optimizerD2.step()


                #------ extra regularization for z------#
                if self.opts['z_reg']:
                    self.netG.zero_grad()
                    self.netE.zero_grad()
                    fake = self.netG(noise)
                    err_E = self.opts['zreg_weight'] * self.criterion_dist(self.netE(fake), noise.squeeze())
                    err_E.backward()
                    self.optimizerG.step()

                pred_fake_d2 = self.netD2(fake.detach())
                # true label
                labell = torch.FloatTensor(self.opts['batchSize'],1).fill_(1.).to(self.device) 
                err_g_d2 = self.criterion_gan(pred_fake_d2, labell)

                self.optimizerD2.zero_grad()
                self.optimizerG.zero_grad()
                err_g_d2.backward()
                self.optimizerG.step()


                ######  FID SCORE AND INCEPTION NET ######
                pred_real_d3 = self.netD3(data)
                pred_fake_d3 = self.netD3(fake.detach())
                err_real_d3 = self.criterion_gan(pred_real_d3, true_wp(self.d_prob, pred_real_d3.size(),self.device))
                err_fake_d3 = self.criterion_gan(pred_fake_d3, true_wp(1.-self.d_prob, pred_fake_d3.size(), self.device))
                inception_loss = err_real_d3 + err_fake_d3

                self.optimizerD3.zero_grad()
                inception_loss.backward()
                self.optimizerD3.step()

                # compute fid score
                with torch.no_grad():

                    if self.opts['noise_dist'] == 'normal':
                        noise.normal_(0., self.opts['z_var'])
                    else:
                        noise.uniform_(-self.opts['z_var'],self.opts['z_var'])

                    fake = self.netG(noise)
                    fid = self.netD3.fid_score(data.detach(), fake.detach())

                # SAVE LOSSES
                minibatchLossD1.append(float(err_d1.detach()))
                minibatchLossD2.append(float(err_d2.detach()))
                minibatchLossG1_gan.append(float(errG_discrim.detach()))
                minibatchLossG1_rec.append(float(errG_recon.detach()))
                minibatchLossG2.append(float(err_g_d2.detach()))
                minibatchLossD3.append(float(inception_loss.detach()))