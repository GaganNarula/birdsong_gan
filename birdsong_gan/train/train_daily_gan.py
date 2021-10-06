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
from data.dataset import bird_dataset, transform
import itertools
from models.nets_16col_residual import _netD, _netE, _netG, InceptionNet, weights_init
from hmm.hmm import learnKmodels_getbest
from hmm.hmm_utils import munge_sequences
from utils.utils import overlap_encode, overlap_decode, gagan_save_spect, save_audio_sample, rescale_spectrogram


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

        TD, _ = self.dataset.make_chunk_tensor_dataset(day, imageW=self.opts['imageW'],
                                                        shuffle_chunks=True)
        return DataLoader(TD, batch_size= self.opts['batchSize'], sampler = None,
                                    shuffle=True, num_workers=int(self.opts['workers']),
                                    drop_last = True)

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
                    FID.append(fid)

                # SAVE LOSSES
                minibatchLossD1.append(float(err_d1.detach()))
                minibatchLossD2.append(float(err_d2.detach()))
                minibatchLossG1_gan.append(float(errG_discrim.detach()))
                minibatchLossG1_rec.append(float(errG_recon.detach()))
                minibatchLossG2.append(float(err_g_d2.detach()))
                minibatchLossD3.append(float(inception_loss.detach()))

                if i % self.opts['log_every'] == 0:
                    # LOG 
                    
                    print('[%d/%d][%d/%d] D1: %.2f D2: %.2f G1_gan: %.2f G1_rec: %.2f G2: %.2f D3: %.2f FID: %.3f MDS: %.2f'
                      % (epoch, self.opts['niter'], i, len(traindataloader),
                        np.mean(minibatchLossD1[-self.opts['log_every']:]),
                         np.mean(minibatchLossD2[-self.opts['log_every']:]),
                         np.mean(minibatchLossG1_gan[-self.opts['log_every']:]),
                         np.mean(minibatchLossG1_rec[-self.opts['log_every']:]),
                         np.mean(minibatchLossG2[-self.opts['log_every']:]),
                         np.mean(minibatchLossD3[-self.opts['log_every']:]),
                         np.mean(FID[-self.opts['log_every']:]),
                         minbE)
                      )
                
                    # sample and reconstruct
                    with torch.no_grad():
                        
                        if self.opts['noise_dist'] == 'normal':
                            noise.normal_(0., self.opts['z_var'])
                        else:
                            noise.uniform_(-self.opts['z_var'],self.opts['z_var'])
                        
                        # generate fakes
                        #fake = self.netG(noise)
                        #out_shape = [self.opts['imageH'], self.opts['imageW']]
                        #fake_spectrograms =[fake.data[k].cpu().numpy().reshape(out_shape) for k in range(8)]
                        #fake_spectrograms = np.concatenate(fake_spectrograms,axis=1)
                        #gagan_save_spect('%s/fake_samples_epoch_%03d_batchnumb_%d.png' 
                        #                    % (self.opts['outf'], epoch, i),rescale_spectrogram(fake_spectrograms))
                        #gagan_save_spect('%s/fake_samples_epoch_%03d_batchnumb_%d.eps' 
                        #                    % (self.opts['outf'], epoch, i),rescale_spectrogram(fake_spectrograms))


                        # randomly sample a file and save audio sample
                        sample = self.dataset.get_random_item()[0] # first element of list output 
                        
                        # audio
                        if self.opts['get_audio']:
                            try:
                                save_audio_sample(lc.istft(inverse_transform(transform(sample))), \
                                                    '%s/input_audio_epoch_%03d_batchnumb_%d.wav' % 
                                                    (self.opts['outf'], epoch, i), self.opts['sample_rate'])
                            except:
                                print('..audio buffer error, skipped audio file generation')

                        # save original spectrogram
                        gagan_save_spect('%s/input_spect_epoch_%03d_batchnumb_%d.eps'
                                            % (self.opts['outf'], epoch, i), 
                                            rescale_spectrogram(transform(sample)))
                        # save reconstruction
                        zvec = overlap_encode(sample, netE, transform_sample = False, imageW = self.opts['imageW'],
                                            noverlap = self.opts['noverlap'], cuda = self.opts['cuda'])
                        spect, audio = overlap_decode(zvec, netG, noverlap = self.opts['noverlap'], get_audio = self.opts['get_audio'], 
                                                    cuda = self.opts['cuda'])
                        # save reconstructed spectrogram
                        spect = rescale_spectrogram(spect)
                        gagan_save_spect('%s/rec_spect_epoch_%03d_batchnumb_%d.eps' % (self.opts['outf'], epoch, i), spect)
                        
                        if self.opts['get_audio']:
                            try:
                                save_audio_sample(audio,'%s/rec_audio_epoch_%03d_batchnumb_%d.wav' % 
                                                    (self.opts['outf'], epoch, i),self.opts['sample_rate'])
                            except:
                                print('..audio buffer error, skipped audio file generation')




    def train_hmm(self, day, hidden_size):
        
        # encode all spectrograms from that day
        X = self.dataset.get(day)
        self.Z = [overlap_encode(x, self.netE, transform_sample = False, imageW = self.opts['imageW'], 
                                 noverlap = self.opts['noverlap'], cuda = self.opts['cuda']) for x in X]
        # munge sequences
        if self.opts['munge']:
            self.Z = munge_sequences(self.Z, self.opts['munge_len'])

        # split into train and validation
        ids = np.random.permutation(len(self.Z))
        ntrain = int(self.opts['hmm_train_proportion'] * len(ids))
        ztrain = [self.Z[ids[i]] for i in range(ntrain)]
        ztest = [self.Z[ids[i]] for i in range(ntrain, len(ids))]
        ids_train = ids[:ntrain]
        ids_test = ids[ntrain:]

        # get lengths of sequences
        Ltrain = [z.shape[0] for z in ztrain]
        # train HMM 
        print('# learning hmm with %d states #'%(hidden_size))
        model = learnKmodels_getbest(ztrain, None, Ltrain, hidden_size, self.opts)

        