r"""Compute the FID score of a model. A model consists of a combination of an HMM and GAN generater/decoder.
    
    One Inception network is learned from some subset of data.
    
    The Inception
"""


import os
from os.path import join
import numpy as np
import json
import torch 
import torch.nn as nn
import h5py
import gc
import joblib
from glob import glob
from random import shuffle
import pdb
import pandas as pd

from utils.utils import load_netG, segment_image, load_InceptionNet
from models.nets_16col_residual import FID
from hmm.hmm_utils import generate_samples
from data.dataset import bird_dataset_single_hdf
from torch.utils.data import TensorDataset, DataLoader
import argparse



class HDFDataset(torch.utils.data.Dataset):
    
    def __init__(self, path_to_hdf):
        self.filepath = path_to_hdf
        self.file = None
            
        self._len = 0
        
    def _create(self):
        self.file = h5py.File(self.filepath, 'w')
        self.group = self.file.create_group('data')
        
    def _open(self):
        self.file = h5py.File(self.filepath, 'r')
        # set length, divide by two because there are pairs
        # data and labels, minus 2 for daily_lengths and which_days
        self._len = (len(self.file['data'].keys()) // 2) - 2 
        
    def _add_data(self, data, data_name):
        self.group.create_dataset(data_name, data=data)
        self._len += 1
        
    def __getitem__(self, index):
        x = np.array(self.file['/data/' + str(index) + '_x'])
        y = np.array(self.file['/data/' + str(index) + '_y'])
            
        return torch.from_numpy(x).to(torch.float32), \
                torch.from_numpy(y).to(torch.float32)
    
    def __len__(self):
        return self._len
        
    def close(self):
        self.file.close()
    
    
    
def load_hmm(hmm_path):
    hmm = joblib.load(hmm_path)
    return hmm['model']



@torch.no_grad()
def compute_fid_scores(model, data, samples, batch_size=128, max_length=16, cuda_device='cuda:0'):
    """Computes the Frechet Inception Distance by taking the feature vector of the second last layer
        of the InceptionNet model and computing the frechet distance between a set of min_samps real
        and min_samps fake data.
        
        Parameters
        ----------
            model : InceptionNet model, already learned
            data : list of np.arrays, real data (log transformed spectrograms)
            sample_seqs : list of np.arrays, same shape as real data, but are samples from the model.
            batch_size : int, minimum number of samples in a FID computation, default = 400
            max_length : int, pad or cut spectrograms to this length
            cuda_device : str, either 'cpu' or 'cuda:0' (default) or other gpu ids (e.g. 'cuda:1')
            
        Returns
        -------
            FID_scores : np.array
    """
    FID_scores = []
    device = torch.device(cuda_device)
    
    x_real = []
    x_fake = []
    for n in range(len(data)):
         
        if len(x_real) < batch_size:
            # add up
            x_real.append(data[n])
            x_fake.append(samples[n])
            continue
            
        x_r = torch.from_numpy(x_real).to(torch.float32).cuda()
        x_f = torch.from_numpy(x_fake).to(torch.float32).cuda()
        
        x_real, x_fake = [], []
        
        score = model.fid_score(x_r, x_f)

        FID_scores.append(score)

    return np.array(FID_scores).mean()



def make_chunks(x, chunk_len=16):
    """Create L chunks from an spectrogram of size (H,W) 
        Generally, H = 129  (freq), W = spectrogram frames (time)
        
        Returns
        -------
            chunks : list of numpy array of shape (1, H, chunk_len)
    """
    chunks = [x[:,i*chunk_len : (i+1)*chunk_len].reshape(1,x.shape[0],chunk_len) \
                for i in range(x.shape[1]//chunk_len)]
    return chunks
    
    
    
def load_data_hdf(path_bird_hdf, bird):
    """Load one day's data. Tutor cane be loaded by passing day = 'tutor'.
    """
    return bird_dataset_single_hdf(path_bird_hdf, bird)
   
    
    
def make_hdf_for_classifier(dataset, netG, end_day, args):
    
    # make a temporary hdf file 
    hdf_dataset = HDFDataset(args.hdf_file_path)
    hdf_dataset._create()
    
    daily_lengths = []
    which_days = [] # on which days does this hmm type exist
    idx = 0
    for day in range(args.start_day, end_day):
        
        # load hmm model
        modelpath = join(args.birdpath, 'day_' + str(day),
                         'hmm_hiddensize_'+str(args.hidden_state_size),
                        'model_day_'+str(day)+'.pkl')
        
        if not os.path.exists(modelpath):
            continue
            
        which_days.append(day)
        
        hmm = load_hmm(modelpath)
        
        # load data
        X = dataset.get(day, nsamps=-1)
        
        # create spectrogram chunks 
        data = []
        for x in X:
            data += make_chunks(x)
        Yreal = [1 for _ in range(len(X))] # labels
        
        daily_lengths.append(len(X))
        X = None
        
        # generate sample sequences
        print('..... day %d , generating samples .....'%(day))
        
        samples = generate_samples(netG, hmm, nsamples=len(X), invtemp=1.,
                         timesteps=[x.shape[1]//args.chunk_len for x in X],
                         cuda=True)
        
        sample_seqs = []
        for seq in samples:
            sample_seqs += make_chunks(seq)
        Ysample = [0 for _ in range(len(samples))]
        samples = None
        
        all_data = data + sample_seqs
        all_labels = Yreal + Ysample
        
        # add to hdf 
        for x,y in zip(all_data, all_labels):
            hdf_dataset._add_data(x, str(idx) + '_x')
            hdf_dataset._add_data(y, str(idx) + '_y')
            idx += 1
            
        data = None
        sample_seqs = None
    
    hdf_dataset._add_data(np.array(daily_lengths), 'daily_lengths')
    hdf_dataset._add_data(np.array(which_days), 'which_days')
    
    hdf_dataset.close()
    return hdf_dataset, daily_lengths, which_days
        



class InceptionNet(nn.Module):
    """A discriminator network from which fid_score can be computed
        Inputs are assumed to be spectrogram chunks.
    """
    def __init__(self, ndf, nc=1, lr=1e-4, l2=1e-4, log_every=100, cuda_device='cuda:0'):
        super(InceptionNet, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(nc, ndf, kernel_size=4, stride=(2,2), padding=(1,1), bias=False),
            # size H = (129 +2 -4)/2 + 1 = 64, W = (16 +2 -4)/2 + 1 = 8
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 3), stride=(2, 1), padding=(1, 0), bias=False),
            # size H = (64 +2 -4)/2 + 1 = 32, W = (8 -3) + 1 = 6
            
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=(4,3), stride=(2,1), padding=(2,0), bias=False),
            # size H = (32 +4 -4)/2 + 1 = 17, W = (6 -3) + 1 = 4
            
            nn.Conv2d(ndf * 2, ndf, kernel_size=(8,4), stride=(2,1), padding=1, bias=False),
            # H = (17 +2 -8)/2 + 1 = 6, W = (4 +2 -4) + 1 = 3
            
            nn.Conv2d(ndf, 1, kernel_size=(6,3), stride=1, padding=0, bias=False),
            # H = 6-6 + 1 =1, W = (4 - 3) +1 = 1
        ])
        self.lns = nn.ModuleList([nn.LayerNorm([ndf, 64, 8]),
                                  nn.LayerNorm([ndf * 2, 32, 6]),
                                  nn.LayerNorm([ndf * 2, 17, 4]),
                                  nn.LayerNorm([ndf, 6, 3])])
        self.activations = nn.ModuleList([nn.LeakyReLU(0.2),nn.LeakyReLU(0.2),
                                    nn.LeakyReLU(0.2),nn.LeakyReLU(0.2)])
        self.out_act =nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2, betas = (0.5, 0.9))
        self.costfunc = nn.BCELoss()
        self.log_every = log_every
        self.device = torch.device(cuda_device)
        
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
        
        x_gen = self.get_middle_layer(x_gen)
        # flatten its last dimensions
        x_gen = x_gen.view(-1, x_gen.size(1)*x_gen.size(2)*x_gen.size(3))
        
        return FID(x_real, x_gen)
    
    def forward(self, x):
        x = self.get_middle_layer(x)
        x = self.convs[-1](x)
        x = self.out_act(x.view(-1,1))
        return x
        
    def fit(self, traindataloader):
        
        N = len(traindataloader)
        for i, batch in enumerate(traindataloader):
            
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            yhat = self.forward(x)
            
            loss = self.costfunc(yhat, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % self.log_every == 0:
                print('..... batch=%d/%d loss = %.3f .....'%(i,N,float(loss.detach())))
                
                
                
    
parser = argparse.ArgumentParser()
parser.add_argument('--birdname', type=str, required=True)
parser.add_argument('--birdpath', type=str, required=True)
parser.add_argument('--path_to_bird_hdf', type=str, required=True)
parser.add_argument('--hdf_file_path', type=str, default=None)
parser.add_argument('--inception_model_path', type=str, default=None)
parser.add_argument('--netGfilepath', type=str, required=True)
parser.add_argument('--hidden_state_size', type=int)
parser.add_argument('--start_day', type=int, default=0, help='first day to compute')
parser.add_argument('--end_day', type=int, default=-1, help='last day to compute')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--fid_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--nepochs', type=int, default=10, help='num training epochs')
parser.add_argument('--chunk_len', type=int, default=16, help='chunk length of spectrogram chunks')
parser.add_argument('--ndf', type=int, default=8, help='num embedding filters')
parser.add_argument('--nlin', type=int, default=50, help='num dense units')
parser.add_argument('--leak', type=float, default=0.1, help='leak on leaky relu functions')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--log_every', type=int, default=100, help='log every so many')
parser.add_argument('--l2', type=float, default=1e-4)
parser.add_argument('--cuda_device', type=str, default='cuda:0')



def main():
    
    args = parser.parse_args()
    
    dataset = load_data_hdf(args.path_to_bird_hdf, args.birdname)
    
    days = glob(join(args.birdpath, 'day_*'))
    ndays = len(days)
    end_day = ndays if args.end_day==-1 else args.end_day
    assert end_day <= ndays, 'end day has to be <= ndays'
    
    # model opts
    with open(join(args.birdpath, 'opts.json'), 'rb') as f:
        model_opts = json.load(f)
    
      
    if not os.path.exists(args.hdf_file_path):
        # load the generator
        print('..... loading generator .....')
        netGfilepath = join(args.birdpath, args.netGfilepath)
        netG = load_netG(netGfilepath, model_opts['nz'], model_opts['ngf'], model_opts['nc'],
                             True, resnet=True)
    
        print('..... making a new dataset of real and fakes samples .....')
        dataset, daily_lengths, which_days = make_hdf_for_classifier(dataset, netG, end_day, args)
        
        # for read, open
        dataset._open()
    
    else:
        print('..... loading premade dataset of real and fake samples .....')
        dataset = HDFDataset(args.hdf_file_path)
        dataset._open()
        
        daily_lengths = np.array(dataset.file['/data/daily_lengths'])
        which_days = np.array(dataset.file['/data/which_days'])
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # make model 
    if not os.path.exists(args.inception_model_path):
        print('..... training model .....')
        model = InceptionNet(args.ndf, 1,
                              args.lr, args.l2,
                              args.log_every,
                              args.cuda_device)

        model.to(torch.device(args.cuda_device))

        # train
        for n in range(args.nepochs):
            model.fit(dataloader)

        model.eval()
        
        # checkpoint model
        torch.save(model.state_dict(), args.inception_model_path)
        
    else:
        print('..... loading model .....')
        model = InceptionNet(args.ndf, 1,
                              args.lr, args.l2,
                              args.log_every,
                              args.cuda_device)
        
        model.load_state_dict(torch.load(args.inception_model_path))
        model.to(torch.device(args.cuda_device))
        model.eval()
        
    del dataloader 
    gc.collect()
    
    # compute scores
    
    ndays = len(which_days)
    FID_scores = np.zeros(ndays)
    
    k = 0
    for j in range(ndays):
        
        print('..... computing scores for day %d .....'%(which_days[j]))
        
        real = torch.stack([dataset[k + l][0] for l in range(daily_lengths[j])])
        
        samples = torch.stack([dataset[k + daily_lengths[j] + l][0] for l in range(daily_lengths[j])])
        
        FID_scores[j] = compute_fid_scores(model, real,
                                           samples,
                                           args.fid_batch_size,
                                           args.max_length,
                                           args.cuda_device)
        k += 2 * daily_lengths[j]
        print(f'..... FID score = {FID_scores[j]} .....')
        
    dataset.close()
    
    # save the scores
    savepath = join(args.birdpath, 'hidden_state_size_'+ str(args.hidden_state_size) +'_FID_scores.csv') 
    
    df = pd.DataFrame(data={'day': which_days, 'FID': FID_scores})
    
    df.to_csv(savepath, index=False)
    
    
    
if __name__ == '__main__':
    main()