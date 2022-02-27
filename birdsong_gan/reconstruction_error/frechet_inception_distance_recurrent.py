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
def compute_fid_scores(model, data, sample_seqs, batch_size=128, max_length=16,
                       cuda_device='cuda:0'):
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
    
    notdone = True
    idx = 0
    n = 0
    while notdone:
        
        x_real = data[idx : idx + batch_size]
        x_fake = sample_seqs[idx : idx + batch_size]
        
        x_real, x_fake = x_real.to(device), x_fake.to(device)
        
        score = model.fid_score(x_real, x_fake)
        FID_scores.append(score)
        
        idx += batch_size
        n += 1
        
        if idx > len(data):
            notdone = False
        
    return np.array(FID_scores).mean()



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
        Yreal = [1 for _ in range(len(X))] # labels
        
        daily_lengths.append(len(X))
        
        
        # generate sample sequences
        print('..... day %d , generating samples .....'%(day))
        
        sample_seqs = generate_samples(netG, hmm, nsamples=len(X), invtemp=1.,
                         timesteps=[x.shape[1]//args.chunk_len for x in X],
                         cuda=True)
        Ysample = [0 for _ in range(len(sample_seqs))]
        
        
        all_data = X + sample_seqs
        all_labels = Yreal + Ysample
        
        # add to hdf 
        for x,y in zip(all_data, all_labels):
            x = pad_to_maxlength(x, args.max_length)
            hdf_dataset._add_data(x, str(idx) + '_x')
            hdf_dataset._add_data(y, str(idx) + '_y')
            idx += 1
            
        X = None
        sample_seqs = None
    
    hdf_dataset._add_data(np.array(daily_lengths), 'daily_lengths')
    hdf_dataset._add_data(np.array(which_days), 'which_days')
    
    hdf_dataset.close()
    return hdf_dataset, daily_lengths, which_days
        
    
    
    
def pad_to_maxlength(x, max_length=100):
    """Pad a sequence to maximum length"""
    if x.shape[1] >= max_length:
        return x[:, :max_length]
    
    # else pad right
    x = np.concatenate([x, np.zeros((x.shape[0], max_length-x.shape[1]))],axis=1)
    return x


    
class InceptionNetRecurrent(nn.Module):
    """A discriminator network from which fid_score can be computed
        Inputs are assumed to be full spectrograms.
    """
    def __init__(self, indims=129, nrnn=100, nrnnlayers=1, nembed=12, nlin=50, leak=0.1,
                 dropout=0.0, lr=1e-4, l2=1e-4, bidirectional=False, log_every=100,
                 cuda_device='cuda:0'):
        super(InceptionNetRecurrent, self).__init__()
        
        self.encoder_rnn = nn.GRU(indims, nrnn, nrnnlayers, bidirectional=bidirectional,
                          dropout=dropout, batch_first=True)
        
        bid = 2 if bidirectional else 1
        
        self.mlp_in = nn.Sequential(
            nn.Linear(nrnn*bid, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nembed)
        )
        
        self.classifier = nn.Linear(nembed, 1)
        
        self.len_in = None
        
        self.out_act =nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2, betas = (0.5, 0.9))
        self.costfunc = nn.BCELoss()
        self.log_every = log_every
        self.device = torch.device(cuda_device)
        
    def encode(self, x):
        # x is a spectrogram of shape (N,H,L)
        # H = feature dims
        # L = length
        x = x.permute(0,2,1)
        
        x,_ = self.encoder_rnn(x)
        # take last time step value
        x = x[:,-1,:]
        return self.mlp_in(x) # z
    
    def fid_score(self, x_real, x_gen):
        x_real = self.encode(x_real)
        
        x_gen = self.encode(x_gen)
        
        return FID(x_real, x_gen)
    
    def forward(self, x):
        x = self.encode(x)
        return self.out_act(self.classifier(x))
        
    def fit(self, traindataloader):
        
        N = len(traindataloader)
        for i, batch in enumerate(traindataloader):
            
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            yhat = self.forward(x)
            
            loss = self.costfunc(yhat.squeeze(), y.squeeze())
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
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
parser.add_argument('--max_length', type=int, default=100, help='time length to pad (or cut) spectrogram to')
parser.add_argument('--chunk_len', type=int, default=16, help='chunk length of spectrogram chunks')
parser.add_argument('--nrnn', type=int, default=100, help='number of rnn units')
parser.add_argument('--nrnnlayers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--nembed', type=int, default=12, help='num embedding dims')
parser.add_argument('--in_dims',type=int, default=129, help='number of fft bins in spectrogram')
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
        model = InceptionNetRecurrent(args.in_dims,
                                      args.nrnn,
                                      args.nrnnlayers,
                                      args.nembed,
                                      args.nlin, args.leak,
                                      args.dropout,
                                      args.lr, args.l2,
                                      True,
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
        model = InceptionNetRecurrent(args.in_dims,
                                      args.nrnn,
                                      args.nrnnlayers,
                                      args.nembed,
                                      args.nlin, args.leak,
                                      args.dropout,
                                      args.lr, args.l2,
                                      True,
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