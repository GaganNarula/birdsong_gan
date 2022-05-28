import torch
import torch.nn as nn
import numpy as np
from birdsong_gan.models.nets_16col_residual import _netD, _netE, _netG, InceptionNet, weights_init




class MDGANTrainer:
    
    def __init__(self, opts):
        
        self.opts = opts
        self.device = torch.device(opts["device"])
        self.nz = opts['nz']
        self.ngf = opts['ngf']
        self.ndf = opts['ndf']

        if opts['cuda']:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        if init_nets is None:
            self._init_networks(opts['netG'], opts['netE'], opts['netD1'], opts['netD2'],
                                 opts['netD3'])
        else:
            self._set_networks(*init_nets)

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
        
    
    def _init_networks(self, netGpath='', netEpath='', netD1path='', netD2path='', netD3path=''):

        # custom weights initialization called on networks
        self.netG = _netG(self.nz, self.ngf, 1)
        self.netG.apply(weights_init)
        if netGpath != '':
            self.netG.load_state_dict(torch.load(netGpath))

        self.netD1 = _netD(self.ndf, 1)
        self.netD1.apply(weights_init)
        if netD1path != '':
            self.netD1.load_state_dict(torch.load(netD1path))

        self.netD2 = _netD(self.ndf, 1)
        self.netD2.apply(weights_init)
        if netD2path != '':
            self.netD2.load_state_dict(torch.load(netD2path))

        # inception net
        self.netD3 = InceptionNet(self.ndf, 1)
        self.netD3.apply(weights_init)
        if netD3path != '':
            self.netD3.load_state_dict(torch.load(netD3path))

        self.netE = _netE(self.nz, self.ndf, 1)
        self.netE.apply(weights_init)
        if netEpath != '':
            self.netE.load_state_dict(torch.load(netEpath))

        if self.opts['cuda']:
            self.netD1 = self.netD1.to(self.device)
            self.netD2 = self.netD2.to(self.device)
            self.netD3 = self.netD3.to(self.device)
            self.netG = self.netG.to(self.device)
            self.netE = self.netE.to(self.device)

        # setup optimizers
        self.optimizerD1 = optim.Adam(self.netD1.parameters(), lr=self.opts['lr'], 
                                betas = (self.opts['beta1'], 0.999))
        self.optimizerD2 = optim.Adam(self.netD2.parameters(), lr=self.opts['lr'], 
                                betas = (self.opts['beta1'], 0.999))
        self.optimizerD3 = optim.Adam(self.netD3.parameters(), lr=self.opts['lr'], 
                                betas = (self.opts['beta1'], 0.999))
        self.optimizerG = optim.Adam(itertools.chain(self.netG.parameters(),
                                            self.netE.parameters()), lr=self.opts['lr'],
                                            betas = (self.opts['beta1'], 0.999))
        
        if self.opts['lr_schedule']:
            gamma=0.9 # multiply lr by this every time scheduler is called
            self.schedulerG = ExponentialLR(self.optimizerG,gamma,verbose=True)
            self.schedulerD1 = ExponentialLR(self.optimizerD1,gamma,verbose=True)
            self.schedulerD2 = ExponentialLR(self.optimizerD2,gamma,verbose=True)
            self.schedulerD3 = ExponentialLR(self.optimizerD3,gamma,verbose=True)

    def _scheduler_step(self):
        self.schedulerG.step()
        self.schedulerD1.step()
        self.schedulerD2.step()
        self.schedulerD3.step()
        
    def common_step(self, data, noise, netE, netG, netD1, netD2, netD3, stage="train"):
        
        # map data X -> Z latent
        encoding = netE(data)
        # map Z -> Xhat
        reconstruction = netG(encoding)
        
        # map  Xhat -> class [0: Fake, 1: Real] (Make discriminator give it value 1)
        pred_rec_d1 = netD1(reconstruction.detach())
        # map X -> class (maximize D)
        pred_real_d1 = netD1(data)
        
        ###### Make Discriminator D1 learn difference between Real and Reconstruction  ########
        # For discriminator, the Pr(class=1|X) = 0.9, true_wp = label with that probability
        err_real_d1 = self.criterion_gan(pred_real_d1, true_wp(self.d_prob,pred_real_d1.size(),
                                        self.device))

        # For disc, probability this is a reconstruction, the Pr(class=1| Xhat) = 0.1 = d_noise
        err_fake_d1 = self.criterion_gan(pred_rec_d1, true_wp(1.-self.d_prob,pred_real_d1.size(),
                                        self.device))

        err_d1 = err_real_d1 + err_fake_d1
        
        if stage == "train":
            self.optimizerD1.zero_grad()
            err_d1.backward()
            # minimize  -logD(X) and maximize -log(D(Xhat)) only w.r.t Discriminator params!
            self.optimizerD1.step()
        
        
        ######## TEACH ENCODER+GENERATOR FOOL Discriminator D1 while reducing autoencoder reconstruction loss ########## 
        # map Xhat -> class
        pred_rec_d1 = netD1(reconstruction)
        labell = torch.FloatTensor(self.opts['batchSize'],1).fill_(1.).to(self.device) # true label
        errG_discrim = self.criterion_gan(pred_rec_d1, labell)
        errG_recon = (self.criterion_dist(reconstruction, data) + \
                      self.criterion_dist(self.downsample_pth(reconstruction),
                      self.downsample_pth(data))) * self.opts['lambdaa']
        
        err_g_d1 = errG_discrim + errG_recon
        
        if stage == "train":
            self.optimizerG.zero_grad()
            err_g_d1.backward()
            # maximize log D(Xhat) or minimize -log D(Xhat) + MSE for encoder and generator
            self.optimizerG.step()
        
        
        # ------------- Diffusion step --------------- #

        encoding = netE(data)
        reconstruction = netG(encoding)
        pred_rec_d2 = netD2(reconstruction.detach())
        err_real_d2 = self.criterion_gan(pred_rec_d2, true_wp(self.d_prob, pred_real_d1.size(),self.device))


        if self.opts['noise_dist'] == 'normal':
            noise.normal_(0., self.opts['z_var'])
        else:
            noise.uniform_(-self.opts['z_var'],self.opts['z_var'])
            
        # Make Discriminator D2 learn difference between real and fake
        fake = netG(noise)
        pred_fake_d2 = netD2(fake.detach())
        err_fake_d2 = self.criterion_gan(pred_fake_d2, true_wp(1.-self.d_prob,pred_real_d1.size(),self.device))
        err_d2 = err_real_d2 + err_fake_d2

        self.optimizerD2.zero_grad()
        err_d2.backward()
        self.optimizerD2.step()
            
        #------ Cycle loss : regularization for latent space ------#
        fake = netG(noise)
        err_E = self.opts['zreg_weight'] * self.criterion_dist(netE(fake), noise.squeeze())
        
        if self.opts['z_reg']  and stage=="train":
            self.optimizerG.zero_grad()
            err_E.backward()
            self.optimizerG.step()
            
        
        # Make generator maximize the discriminator loss = log D(fake)
        pred_fake_d2 = netD2(fake.detach())
        # true label
        labell = torch.FloatTensor(self.opts['batchSize'],1).fill_(1.).to(self.device) 
        err_g_d2 = self.criterion_gan(pred_fake_d2, labell)

        self.optimizerD2.zero_grad()
        self.optimizerG.zero_grad()
        err_g_d2.backward()
        self.optimizerG.step()
            
        
        
        ######  FID SCORE AND INCEPTION NET ######
        pred_real_d3 = netD3(data)
        pred_fake_d3 = netD3(fake.detach())
        err_real_d3 = self.criterion_gan(pred_real_d3, true_wp(self.d_prob, pred_real_d3.size(),self.device))
        err_fake_d3 = self.criterion_gan(pred_fake_d3, true_wp(1.-self.d_prob, pred_fake_d3.size(), self.device))
        inception_loss = err_real_d3 + err_fake_d3

        self.optimizerD3.zero_grad()
        inception_loss.backward()
        self.optimizerD3.step()

        # compute fid score
        if self.opts['get_FID_score']:
            with torch.no_grad():#

                if self.opts['noise_dist'] == 'normal':
                    noise.normal_(0., self.opts['z_var'])
                else:
                    noise.uniform_(-self.opts['z_var'],self.opts['z_var'])
                fake = netG(noise)
                fid = netD3.fid_score(data.detach(), fake.detach())
        else:
            fid = -1.

                
        return float(err_d1.detach()), float(err_d2.detach()), float(err_g_d1.detach()), float(err_g_d2.detach()),
            float(err_E.detach()), float(inception_loss.detach()), fid
    
                
    def train(self, netE, netG, netD1, netD2, netD3, traindataloader, Xpca=None):
        
        # noise variable
        noise = torch.FloatTensor(self.opts['batchSize'], self.nz, 1,1).to(self.device)

        for i, data in enumerate(traindataloader):

            data = data[0] # pytorch produces tuples by default
            data = data.view(data.size(0),1,data.size(1),data.size(2)).to(self.device)
            
            err_d1, err_d2, err_g_d1, err_g_d2, err_E, inception, fid = self.common_step(data, noise,
                                                                                         netE, netG,
                                                                                         netD1, netD2,
                                                                                         netD3, stage="train")
            

            if self.opts['do_pca']:
                if len(Xpca) < self.opts['npca_samples'] and pca_model is None:
                    data = data.detach().cpu().numpy().squeeze()
                    for d in data:
                        Xpca.append(d)
        
        return netG, netE,  netD1, netD2, netD3, Xpca
    
    def evaluate(self, netG, netE, netD1, netD2, netD3, dataloader):
        
        
        with torch.no_grad():
            
            # noise variable
            noise = torch.FloatTensor(self.opts['batchSize'], self.nz, 1,1).to(self.device)
            for i, data in enumerate(traindataloader):

                data = data[0] # pytorch produces tuples by default
                data = data.view(data.size(0),1,data.size(1),data.size(2)).to(self.device)

                err_d1, err_d2, err_g_d1, err_g_d2, err_E, inception, fid = self.common_step(data, noise,
                                                                                             netE, netG,
                                                                                             netD1, netD2,
                                                                                             netD3, stage="test")
    
    
    def run(self, netG, netE, netD1, netD2, netD3, train_dataloader, val_dataloader):
        
        
        if self.opts['do_pca']:
            Xpca = [] # will store some chunks
            pca_model = None

        # training loop
        for epoch in range(self.opts['nepochs']):

            netG, netE,  netD1, netD2, netD3, Xpca = self.train(netG, netE, netD1, netD2, netD3, train_dataloader)
            
            self.evaluate(netG, netE, netD1, netD2, netD3, val_dataloader)