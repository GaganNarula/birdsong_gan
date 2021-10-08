from time import daylight
import torch
import torch.nn as nn
from configs.cfg import *
import os
from os.path import join
import joblib
import argparse
import shutil
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import librosa as lc
from datetime import datetime
from data.dataset import bird_dataset_single_hdf, transform, inverse_transform
import itertools
from models.nets_16col_residual import _netD, _netE, _netG, InceptionNet, weights_init
from hmm.hmm import learnKmodels_getbest
from hmm.hmm_utils import munge_sequences, full_entropy, create_output, number_of_active_states_viterbi
from utils.utils import overlap_encode, overlap_decode, gagan_save_spect, save_audio_sample, \
    rescale_spectrogram



gan_opts = {'datapath': '', 'outf': '', 'distance_fun': 'L1', 'subset_age_weights' : [0., 1.], 
            'workers': 6, 'batchSize': 128, 
            'imageH': 129, 'imageW': 16, 'noverlap':0, 'nz': 16,'nc': 1, 'ngf': 256,
            'ndf': 128,'nepochs': 10,
            'lr': 1e-5, 'lambdaa': 150, 'zreg_weight': 1, 'schedule_lr':False, 'd_noise': 0.1,
            'beta1': 0.5, 'cuda': True, 'ngpu': 1, 'nreslayers': 3, 'z_reg':False, 'mds_loss':False,
            'netG': '','netE': '','netD1':'','netD2':'','netD3':'', 'log_every': 300,
            'sample_rate': 16000.,'noise_dist': 'normal','z_var': 1.,'nfft': 256, 'get_audio': False,
            'manualSeed': [], 'do_pca': True, 'npca_samples': 1e6, 'npca_components': 256}

hmm_opts = {'hidden_state_size' : [5, 10, 15, 20, 30, 50, 75, 100], 'covariance_type' : 'spherical', 
           'fit_params' : 'stmc', 'transmat_prior' : 1., 'n_iter' : 300, 'tolerance' : 0.01,
            'covars_prior' : 1., 'init_params' : 'stmc',
            'train_proportion' : 0.7, 'nsamplesteps' : 128, 'nsamps': 10,
            'sample_var': 0., 'sample_invtemperature' : 1.,
            'munge' : False, 'munge_len' : 50,
            'n_restarts': 1, 'do_chaining': False,
            'min_seq_multiplier': 10, 'cuda' : True, 'hmm_random_state' : 0,
            'last_day': -1,
            'start_from' : 0,
            'get_audio': False
           }

# opts structure to use
opts = {**gan_opts, **hmm_opts}



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
        if len(os.listdir(join(path, d)))<=3:
            try:
                os.rmdir(join(path, d))
            except:
                shutil.rmtree(join(path,d))
    path += str(datetime.now()).replace(':', '-')
    path = path.replace(' ','_')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(join(path,'net_training_losses'))
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

        The container contains the model, a GAN training method, a 
        HMM training method and sampling + checkpointing methods.
    """

    def __init__(self, dataset, outpath, opts, init_nets=None):
        
        self.opts = opts

        # make a folder to save results
        self.outpath = outpath

        self.dataset = dataset

        self.nz = opts['self.nz']
        self.ngf = opts['ngf']
        self.ndf = opts['ndf']

        if opts['cuda']:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        if init_nets is None:
            self._init_networks(opts['netG'], opts['netE'], opts['netD1'], opts['netD2'],
                                opts['netD2'], opts['netD3'])
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

        self.X = None # data arrays (lists) for hmm learning
        self.Z = None # encoded latent vectors for hmm learning

    def _set_networks(self, netG, netE, netD1, netD2, netD3):
        self.netG = netG
        self.netE = netE
        self.netD1 = netD1
        self.D2 = netD2
        self.D3 = netD3

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
        # for mds loss
        if self.opts['mds_loss']:
            self.optimizerE = optim.Adam(self.netE.parameters(), lr=self.opts['lr'], 
                                        betas = (self.opts['beta1'], 0.999))


    def make_dataloader(self, day=0):

        TD, _ = self.dataset.make_chunk_tensor_dataset(day, imageW=self.opts['imageW'],
                                                        shuffle_chunks=True)
        return DataLoader(TD, batch_size= self.opts['batchSize'], sampler = None,
                                    shuffle=True, num_workers=int(self.opts['workers']),
                                    drop_last = True)

    def train_one_day(self, day, traindataloader):
        
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
        for epoch in range(self.opts['nepochs']):

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
                    
                    print('[%d/%d][%d/%d] D1: %.2f D2: %.2f G1_gan: %.2f G1_rec: %.2f G2: %.2f D3: %.2f FID: %.3f'
                      % (epoch, self.opts['niter'], i, len(traindataloader),
                        np.mean(minibatchLossD1[-self.opts['log_every']:]),
                         np.mean(minibatchLossD2[-self.opts['log_every']:]),
                         np.mean(minibatchLossG1_gan[-self.opts['log_every']:]),
                         np.mean(minibatchLossG1_rec[-self.opts['log_every']:]),
                         np.mean(minibatchLossG2[-self.opts['log_every']:]),
                         np.mean(minibatchLossD3[-self.opts['log_every']:]),
                         np.mean(FID[-self.opts['log_every']:]))
                      )

                    self.checkpoint(i, epoch)

                # END OF MINIBATCH

            # END OF EPOCH
            # document losses at end of epoch
            losspath = join(self.outpath, 'net_training_losses/')
            np.save(losspath+'D1',np.array(minibatchLossD1))
            np.save(losspath+'D2',np.array(minibatchLossD2))
            np.save(losspath+'G1rec',np.array(minibatchLossG1_rec))
            np.save(losspath+'G1gan',np.array(minibatchLossG1_gan))
            np.save(losspath+'G2',np.array(minibatchLossG2))
            np.save(losspath+'D3',np.array(minibatchLossD3))
            np.save(losspath+'FID',np.array(FID))
                    
                    
            # do checkpointing of models
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%d_day_%d.pth'%(self.outpath,epoch,
                                                                                day))
            torch.save(self.netD1.state_dict(), '%s/netD1_epoch_%d_day_%d.pth' % (self.outpath,epoch,
                                                                                day))
            torch.save(self.netD2.state_dict(), '%s/netD2_epoch_%d_day_%d.pth' % (self.outpath,epoch,
                                                                                day))
            torch.save(self.netD3.state_dict(), '%s/netD3_epoch_%d_day_%d.pth' % (self.outpath,epoch,
                                                                                day))
            torch.save(self.netE.state_dict(), '%s/netE_epoch_%d_day_%d.pth' % (self.outpath, epoch,
                                                                                day))
            

    def checkpoint(self, minibatch_idx, epoch) -> None:
        
        # noise variable
        noise = torch.FloatTensor(self.opts['batchSize'], self.nz, 1,1).to(self.device)

        # sample and reconstruct
        with torch.no_grad():
            
            if self.opts['noise_dist'] == 'normal':
                noise.normal_(0., self.opts['z_var'])
            else:
                noise.uniform_(-self.opts['z_var'],self.opts['z_var'])
        

            # randomly sample a file and save audio sample
            sample = self.dataset.get_random_item()[0] # first element of list output 
            
            # audio
            if self.opts['get_audio']:
                try:
                    save_audio_sample(lc.istft(inverse_transform(transform(sample))), \
                                        '%s/input_audio_epoch_%03d_batchnumb_%d.wav' % 
                                        (self.outpath, epoch, minibatch_idx), self.opts['sample_rate'])
                except:
                    print('..audio buffer error, skipped audio file generation')

            # save original spectrogram
            gagan_save_spect('%s/input_spect_epoch_%03d_batchnumb_%d.eps'
                                % (self.outpath, epoch, minibatch_idx), 
                                rescale_spectrogram(transform(sample)))
            # get reconstruction
            zvec = overlap_encode(sample, self.netE, transform_sample = False,
                                imageW = self.opts['imageW'],
                                noverlap = self.opts['noverlap'], cuda = self.opts['cuda'])

            spect, audio = overlap_decode(zvec, self.netG, noverlap = self.opts['noverlap'],
                                        get_audio = self.opts['get_audio'], 
                                        cuda = self.opts['cuda'])

            # save reconstructed spectrogram
            spect = rescale_spectrogram(spect)
            gagan_save_spect('%s/rec_spect_epoch_%03d_batchnumb_%d.eps' % (self.outpath,
                            epoch, minibatch_idx), spect)
            
            if self.opts['get_audio']:
                try:
                    save_audio_sample(audio,'%s/rec_audio_epoch_%03d_batchnumb_%d.wav' % 
                                        (self.outpath, epoch, minibatch_idx), self.opts['sample_rate'])
                except:
                    print('..audio buffer error, skipped audio file generation')


    def get_loglikelihood(self, model, data, lengths):
        return model.score(np.concatenate(data), lengths)


    def train_hmm(self, day, hidden_size):
        
        # encode all spectrograms from that day
        if self.X is None:
            self.X = self.dataset.get(day)
        if self.Z is None:
            self.Z = [overlap_encode(x, self.netE, transform_sample=False,
                                    imageW=self.opts['imageW'], 
                                    noverlap = self.opts['noverlap'],
                                    cuda=self.opts['cuda']) for x in self.X]
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
        self.hmm = learnKmodels_getbest(ztrain, None, Ltrain, hidden_size, self.opts)

        # produce samples
        # compute 2 step full entropy
        print('# computing model entropy #')
        Hsp, Htrans, Hgauss = full_entropy(self.hmm)
        print('..... Transition entropy = %.2f, Emission entropy = %.2f .....'%(Htrans, Hgauss))
        
        # compute test log likelihood
        Ltest = [z.shape[0] for z in ztest]
        test_scores = self.get_loglikelihood(self.hmm, ztest, Ltest)
        # compute train log likelihood
        train_scores = self.get_loglikelihood(self.hmm, ztrain, Ltrain)

        # create 10 samples
        # concatenate the sequences because otherwise they are usually shorter than batch_size
        outputfolder = join(self.outpath, 'hmm_hiddensize_'+str(hidden_size))
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)

        # choose some ztrain for saving
        inds_to_save = np.random.choice(len(ztrain), size=self.opts['nsamps'])
        ztosave = [ztrain[i] for i in inds_to_save]
        ztosave = np.concatenate(ztosave, axis=0)

        # save samples
        create_output(self.hmm, outputfolder, hidden_size, day, self.P, self.netG, [])
        # save real files
        create_output(self.hmm, outputfolder, hidden_size, day, self.P, self.netG, ztosave)
        print('# generated samples #')

        # get number of active states etc
        # how many active states were there ? 
        med_active, std_active = number_of_active_states_viterbi(self.hmm,
                                                                    np.concatenate(ztrain), 
                                                                    Ltrain)
        # save model
        joblib.dump({'train_score':train_scores, 'test_score': test_scores,
                        'med_active':med_active,
                        'ztrain':ztrain,'ztest':ztest, 'std_active':std_active,
                        'ids_train':ids_train,
                        'ids_test':ids_test, 'Lengths_train':Ltrain,'Lengths_test':Ltest, 
                        'Entropies':[Hsp,Htrans,Hgauss]}, 
                        join(outputfolder, 'data_and_scores_day_'+str(day)+'.pkl'))
        joblib.dump({'model':self.hmm}, join(outputfolder, 'model_day_'+str(day)+'.pkl'))
        


        



parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True, help='path to training dataset')
parser.add_argument('--birdname', type=str, )
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--train_residual', action = 'store_true')
parser.add_argument('--noise_dist', type=str, default = 'normal', help='noise distribution: {normal, uniform, t}')
parser.add_argument('--lambdaa', type = float, default = 100., help = 'weighting for recon loss')
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--ndf', type=int, default=256)
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train GAN ')
parser.add_argument('--lr', type=float, default = 0.00001, help='learning rate')
parser.add_argument('--z_reg', action="store_true", help='whether to regularize the posterior')
parser.add_argument('--zreg_weight', type = float, default = 1., help = 'weight for z regularization')
parser.add_argument('--z_var', type = float, default = 1., help = 'variance of latent prior')
parser.add_argument('--manualSeed', type=int, default=-1, help='random number generator seed')
parser.add_argument('--schedule_lr', action = 'store_true', help='change learning rate')
parser.add_argument('--log_every', type=int, default=300, help='make images and print loss every X batches')
parser.add_argument('--get_audio', action='store_true', help='write wav file from spectrogram')
parser.add_argument('--cuda', action='store_true', help='use gpu')
parser.add_argument('--netE',type = str, default = '', help='path to encoder network file')
parser.add_argument('--netG',type = str, default = '', help='path to generator network file')
parser.add_argument('--netD1',type = str, default = '', help='path to disc 1 network file')
parser.add_argument('--netD2',type = str, default = '', help='path to encoder network file')
parser.add_argument('--netD3',type = str, default = '', help='path to encoder network file')
parser.add_argument('--hidden_state_size', type = int, nargs = '+', default = [5, 10, 15, 20, 30, 50, 75, 100])
parser.add_argument('--covariance_type', type = str, default = 'spherical')
parser.add_argument('--covars_prior', type = float, default = 1., help ='diagnoal term weight on the prior covariance')
parser.add_argument('--fit_params', type = str, default = 'stmc', help = 'which parameters to fit, s = startprob, t = transmat, m = means, c = covariances')
parser.add_argument('--transmat_prior', type = float, default = 1., help = 'transition matrix prior concentration')
parser.add_argument('--n_iter', type = int, default = 400, help = 'number of EM iterations')
parser.add_argument('--tolerance', type = float, default = 0.01)
parser.add_argument('--get_audio', action = 'store_true', help = 'generate audio files as well')
parser.add_argument('--start_from', type = int, default = 0, help = 'start day of learning') 
parser.add_argument('--last_day', type = int, default = -1, help = 'last day of learning')
parser.add_argument('--min_seq_multiplier', type = int ,default = 10, help='the number of files should be at least hidden size x this factor')
parser.add_argument('--init_params', type = str, default = 'str', help='which variables to initialize, or to initialize with kmeans, enter kmeans')
parser.add_argument('--munge', action = 'store_true', help='whether to concate')
parser.add_argument('--munge_len', type = int, default = 50, help = 'minimum length of a sequence to which sequences be concatenated')




if __name__ == '__main__':

    
    args = parser.parse_args()
    args  = vars(args)
    for k,v in args.items():
        if k in opts.keys():
            opts[k] = v


    if opts['manualSeed']==-1:
        opts['manualSeed'] = random.randint(1, 10000) # fix seed

    # fix seed
    print("Random Seed: ", opts['manualSeed'])
    random.seed(opts['manualSeed'])
    torch.manual_seed(opts['manualSeed'])
    if opts['cuda']:
        torch.cuda.manual_seed(opts['manualSeed'])
    opts['hmm_random_state'] = opts['manualSeed']

    # make bird dataset
    dataset = bird_dataset_single_hdf(opts['datapath'], opts['birdname'])
    # how many days are there ? 
    ndays = dataset.ndays

    # make model
    model = Model(dataset, '', opts, None)
    # make a save folder (run e.g. 2021-09-10_15_12_31)
    opts['outf'] = make_output_folder(opts['outf'])

    for day in range(ndays):
        print('\n\n.... ### WORKING ON DAY %d for bird %s ### .....'%(day, opts['birdname']))

        # update output folder
        outpath = join(opts['outf'], 'day_' + str(day))
        os.makedirs(outpath, exist_ok=True)
        model.outpath = outpath

        # re-initialize networks
        if day > 0 and not opts['chain_networks']:
            model._init_networks()

        # make traindataloader
        traindataloader = model.make_dataloader(day)
        # train networks
        model.train_one_day(day, traindataloader)

        # then, train hmm
        for k in range(opts['hidden_state_size']):

            model.train_hmm(day, opts['hidden_state_size'][k])

        # clear the saved spectrogram and latent vectors arrays
        model.X = None 
        model.Z = None
        

    