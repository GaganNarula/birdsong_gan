import sys
import os
sys.path.append(os.pardir)
from configs.cfg import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import songbird_full_spectrogram_single_file
from models.recurrent_gan import RecurrentGAN
from collections import namedtuple
import argparse
import itertools
from utils.utils import rescale_spectrogram, gagan_save_spect
from datetime import datetime
import random
import joblib
import pdb
#import torch.backends.cudnn as cudnn
#cudnn.deterministic = True
#cudnn.benchmark = True




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
    

    
def mdgan_loss(x, x_hat, x_samp, z_hat, z, y_hat_real, y_hat_real2, y_hat_fake, y_hat_recon,
               y_hat_fake_G, y_hat_recon_G, costfunc, downsample_func, gan_loss, device, d_prob, 
               y_hat_fake_incep=None, y_hat_real3=None):
    """MDGAN loss + cycle loss
        Params
        ------
            x : input data
            x_hat : reconstruction of input data
            x_samp : sampled data from decoder/gen
            z_hat : reconstruction of latent
            z : true latent when sampling
            y_hat_real : predicted label for real input data for discriminator_FvR
            y_hat_real2 : predicted label for real input data for discriminator_ReVR
            y_hat_fake : predicted label on x_samp
            y_hat_recon : predicted label on reconstructed data
            y_hat_fake_G : predicted label on x_samp, for generator loss (torch issues)
            y_hat_recon_G : predicted label on x_hat, for autoencoder loss (torch issues)
            costfunc : L2 / L1 recon loss
            downsample_func : part of recon loss
            gan_loss : binary cross entropy with neg log 
            
        Returns
        -------
            loss1 : recon loss on input
            loss2 : recon loss on latent
            
    """
    # compute errors
    # reconstruction error
    loss0 = costfunc(x_hat, x) + costfunc(downsample_func(x_hat), downsample_func(x))
    # cycle loss
    loss1 = costfunc(z_hat, z)
    
    # gan loss for real data, discriminator needs to minimize this - E[log(D_fakevsreal(x)))]
    loss2 = gan_loss(y_hat_real, true_wp(d_prob, y_hat_real.size(), device))
    # discriminator 1 needs to minimize -E[ log(1 - D(G(z)))]
    loss3 = gan_loss(y_hat_fake, true_wp(1.-d_prob, y_hat_fake.size(),device))
    # gan loss for reconstructed data, discriminator 2 needs to minimize this -E[ log(1-D(G(E(x)))) ]
    loss4 = gan_loss(y_hat_recon, true_wp(1.-d_prob, y_hat_recon.size(),device))
    # discriminator 2 needs to minimize -E [ log(D_reconvsreal(x))]
    loss5 = gan_loss(y_hat_real2, true_wp(d_prob, y_hat_real2.size(), device))
    # for decoder/generator, minimize - E[ log(D(G(z)))]
    loss6 = gan_loss(y_hat_fake_G, true_wp(1., y_hat_fake.size(),device))
    # for autoencoder
    loss7 = gan_loss(y_hat_recon_G, true_wp(1., y_hat_recon.size(),device))
    
    if y_hat_fake_incep is not None:
        loss8 = gan_loss(y_hat_real3, true_wp(1., y_hat_real3.size(), device))
        loss8 += gan_loss(y_hat_fake_incep, true_wp(0., y_hat_fake_incep.size(), device))
    else:
        loss8 = None
        
    return loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8



def evaluate(model, testdataloader, costfunc, downsample_func, gan_loss, device, d_prob, opts):
    """Evaluate model on test dataset """
    test_loss_D_FvR = []
    test_loss_D_RevR = []
    test_loss_Autoenc_recon = []
    test_loss_G_gan = []
    
    with torch.no_grad():
        
        for i, x in enumerate(testdataloader):
            if opts['cuda']:
                x = x.cuda()
                
            # compute reconstruction
            x_hat = model(x)
            
            # sample some fake trajectories
            z = model.prior_sample(opts['batch_size'], opts['max_length']//opts['imageW'],
                                   to_tensor=True)
            x_samp = model.decode(z)
            
            # for cycle loss, encode back the sample
            z_hat = model.encode(x_samp)
            
            # discriminator 1 output for real
            y_hat_real = model.discriminate(x, model.disc_FvR)
            # discriminator 1 output for fake
            y_hat_fake = model.discriminate(x_samp, model.disc_FvR)
            
            # discriminator 2 output for reconstruction
            y_hat_recon = model.discriminate(x_hat, model.disc_RevR)
            # discriminator 2 output for real
            y_hat_real2 = model.discriminate(x, model.disc_RevR)
            
            loss = mdgan_loss(x, x_hat, x_samp, z_hat, z, y_hat_real, y_hat_real2, 
                               y_hat_fake, y_hat_recon, y_hat_fake, y_hat_recon,
                               costfunc, downsample_func, gan_loss, device, d_prob, 
                               y_hat_fake_incep=None, y_hat_real3=None)
            
            loss_disc_FvR = loss[2] + loss[3]
            loss_disc_RevR = loss[4] + loss[5]
            
            test_loss_D_FvR.append(loss_disc_FvR.item())
            test_loss_D_RevR.append(loss_disc_RevR.item())
            
            loss_recon = opts['lambda']*loss[0] + loss[1] 
            test_loss_Autoenc_recon.append(loss_recon.item())
            test_loss_G_gan.append(loss[6].item())
            
    return test_loss_D_FvR, test_loss_D_RevR, test_loss_Autoenc_recon, test_loss_G_gan



    
    
def train(model, traindataloader, testdataloader, opts):
    """Traning function. 
        Params
        ------
            model : 
            traindataloader : torch.utils.data.Dataloader map style
            testdataloader : dataloader for test set
            opts : options dict
                    - nepochs : int, how many epochs of training
    """
    
    if opts['cuda']:
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    # define losses
    if opts['recon_loss_type'] == 'L2':
        recon_func = nn.MSELoss(reduction='mean')
    elif opts['recon_loss_type'] == 'L1':
        recon_func = nn.L1Loss(reduction='mean')
    # downsample function for reconstruction error 
    downsample_func = torch.nn.AvgPool2d(3, stride=4)
    
    gan_loss = nn.BCEWithLogitsLoss(reduction='mean')
    # probability that a label is wrongly labelled
    d_prob = torch.FloatTensor([1. - opts['d_noise']]).to(device)
    
    optimizer_GE = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                            model.decoder.parameters()), 
                                      lr = opts['lr'], weight_decay = opts['l2'], 
                                     betas=(opts['beta1'], 0.999))
    optimizer_G = torch.optim.Adam(model.decoder.parameters(), 
                                      lr = opts['lr'], weight_decay = opts['l2'], 
                                     betas=(opts['beta1'], 0.999))
    optimizer_D1 = torch.optim.Adam(model.disc_FvR.parameters(), 
                                      lr = opts['lr'], weight_decay = opts['l2'], 
                                     betas=(opts['beta1'], 0.999))
    optimizer_D2 = torch.optim.Adam(model.disc_RevR.parameters(), 
                                      lr = opts['lr'], weight_decay = opts['l2'], 
                                     betas=(opts['beta1'], 0.999))
    optimizer_D3 = torch.optim.Adam(model.inception_net.parameters(), 
                                      lr = opts['lr'], weight_decay = opts['l2'], 
                                     betas=(opts['beta1'], 0.999))
    
    N = len(traindataloader)
        
    train_loss_D_FvR = []
    train_loss_D_RevR = []
    train_loss_inception = []
    train_loss_Autoenc_recon = []
    train_loss_G_gan = []
    train_fid = []
    
    for n in range(opts['nepochs']):

        for i, x in enumerate(traindataloader):
            # x is a tensor of shape (N, freq bins, timesteps)
            #with torch.autograd.set_detect_anomaly(True): 
            
            if opts['cuda']:
                x = x.cuda()
            
                
            ###### TRAIN DISCRIMINATOR 1 Fake vs Real ######
            # sample some fake trajectories
            z = model.prior_sample(opts['batch_size'], opts['max_length']//opts['imageW'],
                                   to_tensor=True)
            # generate the spcetrograms
            x_samp = model.decode(z)
            # discriminator 1 output for real
            y_hat_real_D1 = model.discriminate(x, model.disc_FvR)
            # discriminator 1 output for fake
            y_hat_fake_D1 = model.discriminate(x_samp, model.disc_FvR)
            # gan loss for real data, discriminator needs to minimize this - E[log(D_fakevsreal(x)))]
            loss2 = gan_loss(y_hat_real_D1, true_wp(d_prob, y_hat_real_D1.size(), device))
            # discriminator 1 needs to minimize -E[ log(1 - D(G(z)))]
            loss3 = gan_loss(y_hat_fake_D1, true_wp(1.-d_prob, y_hat_fake_D1.size(),device))
            # update Discriminator 1 (fake vs real)
            optimizer_D1.zero_grad()
            loss_disc_FvR = loss2 + loss3
            loss_disc_FvR.backward()
            optimizer_D1.step()


            ###### TRAIN DISCRIMINATOR 2 Reconstruction vs Real #####
            # gan loss for reconstructed data, discriminator 2 needs to minimize this -E[ log(1-D(G(E(x)))) ]
            # compute reconstruction
            x_hat = model(x)
            # discriminator 2 output for reconstruction
            y_hat_recon_D2 = model.discriminate(x_hat, model.disc_RevR)
            y_hat_real_D2 = model.discriminate(x, model.disc_RevR)
            loss4 = gan_loss(y_hat_recon_D2, true_wp(1.-d_prob, y_hat_recon_D2.size(),device))
            # discriminator 2 needs to minimize -E [ log(D_reconvsreal(x))]
            loss5 = gan_loss(y_hat_real_D2, true_wp(d_prob, y_hat_real_D2.size(), device))
            # update Discriminator 2 (reconstruction vs real)
            optimizer_D2.zero_grad()
            loss_disc_RevR = loss4 + loss5
            loss_disc_RevR.backward()
            optimizer_D2.step()


            ###### TRAIN GENERATOR ONLY #####
            # sample latent trajectory
            z = model.prior_sample(opts['batch_size'], opts['max_length']//opts['imageW'],
                                   to_tensor=True)
            # generate the spcetrograms
            x_samp = model.decode(z)
            # discriminator 1 output for samples
            y_hat_fake_G = model.discriminate(x_samp, model.disc_FvR)
            optimizer_G.zero_grad()
            loss6 = gan_loss(y_hat_fake_G, true_wp(1., y_hat_fake_G.size(),device))
            # sampling gan loss just for decoder/generator
            loss6.backward()
            optimizer_G.step()


            ###### TRAIN GENERATOR AND ENCODER TOGETHER #####
            z = model.prior_sample(opts['batch_size'], opts['max_length']//opts['imageW'],
                                   to_tensor=True)
            # generate the spectrograms
            x_samp = model.decode(z)
            # for cycle loss, encode back the sample
            z_hat = model.encode(x_samp)
            # for input reconstruction loss, compute reconstruction
            x_hat = model(x)
            # reconstruction error
            loss0 = recon_func(x_hat, x) + recon_func(downsample_func(x_hat), downsample_func(x))
            # cycle loss
            loss1 = recon_func(z_hat, z)
            # discriminator 2 output for reconstruction
            y_hat_recon_G = model.discriminate(x_hat, model.disc_RevR)
            optimizer_GE.zero_grad()
            loss7 = gan_loss(y_hat_recon_G, true_wp(1., y_hat_recon_G.size(),device))
            loss_autoencoder = opts['lambda']*loss0 + loss1 + loss7
            loss_autoencoder.backward()
            optimizer_GE.step()



            ##### TRAIN INCEPTION NET #####
            # discriminator 3 output for fake
            z = model.prior_sample(opts['batch_size'], opts['max_length']//opts['imageW'],
                                   to_tensor=True)
            # generate the spectrograms
            x_samp = model.decode(z)
            y_hat_fake_incep = model.discriminate(x_samp, model.inception_net)
            # discriminator 3 output for real
            y_hat_real_D3 = model.discriminate(x, model.inception_net)
            # update discriminator 3
            optimizer_D3.zero_grad()
            loss8 = gan_loss(y_hat_real_D3, true_wp(1., y_hat_real_D3.size(), device))
            loss8 += gan_loss(y_hat_fake_incep, true_wp(0., y_hat_fake_incep.size(), device))
            # inception net
            loss8.backward()
            optimizer_D3.step()
            
                
            # record loss values
            train_loss_D_FvR.append(loss_disc_FvR.item())
            train_loss_D_RevR.append(loss_disc_RevR.item())
            train_loss_inception.append(loss8.item())

            loss_recon = opts['lambda']*loss0 + loss1
            train_loss_Autoenc_recon.append(loss_recon.item())
            train_loss_G_gan.append(loss6.item())
            
            
            # frechet inception distance
            if opts['compute_fid']:
                train_fid.append(model.frechet_inception_distance(x, x_hat, model.inception_net))
            else:
                train_fid.append(-1.)
        
            if i%opts['log_every'] == 0:
                print("..... Epoch %d, minibatch [%d/%d], D_FvR=%.2f, D_RevR=%.2f, Auto_enc=%.2f, G_gan=%.2f, incep=%.2f, fid=%.2f ....."%(n,i,N,
                                                                                                                    train_loss_D_FvR[-1],
                                                                                                                    train_loss_D_RevR[-1],
                                                                                                                    train_loss_Autoenc_recon[-1],
                                                                                                                    train_loss_G_gan[-1],
                                                                                                             train_loss_inception[-1],
                                                                                                                    train_fid[-1]))

                # save spectrograms (only first)
                gagan_save_spect('%s/input_spect_epoch_%03d_batchnumb_%d.eps'
                                         % (opts['outf'], n, i), 
                                         rescale_spectrogram(x.detach().cpu().numpy()[0]))
                
                gagan_save_spect('%s/recon_spect_epoch_%03d_batchnumb_%d.eps'
                                         % (opts['outf'], n, i), 
                                         rescale_spectrogram(x_hat.detach().cpu().numpy()[0]))
                
                
                
        # model checkpoint
        torch.save(model.state_dict(), '%s/rec_gan_epoch_%d.pth' % (opts['outf'], n))
        
        # evaluate on test / validation set
        val_loss = evaluate(model, testdataloader, recon_func, downsample_func, gan_loss, device, d_prob, opts)
        print("..... Epoch %d, D_FvR=%.2f, D_RevR=%.2f, Auto_enc=%.2f, G_gan=%.2f ....."%(n, np.mean(val_loss[0]),
                                                                                            np.mean(val_loss[1]),
                                                                                          np.mean(val_loss[2]),
                                                                                          np.mean(val_loss[3])) )
        
    train_loss = (train_loss_D_FvR, train_loss_D_RevR, train_loss_Autoenc_recon, train_loss_G_gan)
    
    return model, train_loss, val_loss



parser = argparse.ArgumentParser()
parser.add_argument('--training_path', required=True, help='path to training dataset id list')
parser.add_argument('--test_path', required=True, help='path to test dataset id list')
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--path2hdf', default=EXT_PATH, help='path to folder containing bird hdf files')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, default=256, help='maximum spectrogram length for rnn training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='momentume term 1 for Adam')
parser.add_argument('--l2', type=float, default=0.0, help='weight_decay')
parser.add_argument('--lambda', type=float, default=100., help='weight on input recon error')
parser.add_argument('--recon_loss_type', type=str, default='L1', help='reconstruction loss type')
parser.add_argument('--rnn_input_dim', type=int, default=50, help='input dimensionality to RNNs')
parser.add_argument('--nz', type=int, default=16, help ='latent space dimensionality')
parser.add_argument('--nrnn', type=int, default=200, help='number of units in recurrent network')
parser.add_argument('--nlin', type=int, default=200, help='number of units in dense mlp')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers in rnn')
parser.add_argument('--bidirectional', action='store_true', help='whether to run rnn forwards and backwards')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate on rnns, only used if more than 2 layers')
parser.add_argument('--leak', type=float, default=0.1, help='leak on leaky relu')
parser.add_argument('--d_noise', type=float, default=0.1, help='noise on labels for discriminator')
parser.add_argument('--ngf', type=int, default=64, help='filter multiplier constant')
parser.add_argument('--compute_fid',action='store_true',help='whether to compute FID scores')
parser.add_argument('--imageW', type=int, default=16, help='chunk length for spectrogram chunking')
parser.add_argument('--hmm_components', type=int, default=20)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--spectral_norm_decoder', action='store_true')
parser.add_argument('--spectral_norm_discriminator', action='store_true')
parser.add_argument('--manual_seed', type=int, default=-1)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--log_every', type=int, default=500, help='log every')



def main():
    
    args = parser.parse_args()
    opts = vars(args)
    
    outf = make_output_folder(args.outf)
    opts['outf'] = outf
    
    if opts['manual_seed'] ==-1:
        opts['manual_seed'] = random.randint(1, 10000) # fix seed
    
    # save opts
    joblib.dump(opts,os.path.join(opts['outf'],'opts.pkl'))
    
    # fix seed
    print("Random Seed: ", opts['manual_seed'])
    random.seed(opts['manual_seed'])
    torch.manual_seed(opts['manual_seed'])
    if opts['cuda']:
        torch.cuda.manual_seed(opts['manual_seed'])
    
    
    # initialize the dataset and dataloader objects
    train_dataset = songbird_full_spectrogram_single_file(args.training_path, args.path2hdf, 
                                                          opts['max_length'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=opts['batch_size'], sampler = None,
                                             shuffle=True, num_workers=opts['workers'],
                                                drop_last = True)
    
    test_dataset = songbird_full_spectrogram_single_file(args.test_path, args.path2hdf,
                                                           opts['max_length'])
    
    test_dataloader = DataLoader(test_dataset, batch_size= opts['batch_size'],
                                             shuffle=True, num_workers=opts['workers'],
                                                    drop_last = True)
    # initiate model
    model = RecurrentGAN(rnn_input_dim=opts['rnn_input_dim'], nz=opts['nz'], nrnn=opts['nrnn'],
                         nlin=opts['nlin'], nlayers=opts['nlayers'], bidirectional=opts['bidirectional'], 
                         dropout=opts['dropout'], leak=opts['leak'], ngf=opts['ngf'], imageW=opts['imageW'],
                         hmm_components=opts['hmm_components'], cuda=opts['cuda'],
                         spectral_norm_decoder=opts['spectral_norm_decoder'],
                         spectral_norm_discriminator=opts['spectral_norm_discriminator'])
    
    # train model
    model, train_loss, test_loss = train(model, train_dataloader, test_dataloader, opts)
    
    joblib({'model': model, 'train_loss': train_loss, 'test_loss':test_loss, 
           }, os.path.join(opts['outf'], 'model_and_losses.pkl'))
    
main()
    
    