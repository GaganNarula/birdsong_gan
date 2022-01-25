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

from utils.utils import load_netG, segment_image, load_InceptionNet
from hmm.hmm_utils import generate_samples
from data.dataset import bird_dataset_single_hdf
from torch.utils.data import TensorDataset, DataLoader
import argparse



class HDFDataset(torch.utils.data.Dataset):
    
    def __init__(self, path_to_hdf):
        self.filepath = path_to_hdf
        self._create()
            
        self._len = 0
        
    def _create(self):
        self.file = h5py.File(self.filepath, 'w')
    
    def _open(self):
        self.file = h5py.File(self.filepath, 'r')
        
    def _add_data(self, data, data_name):
        self.file.create_dataset(data_name, data, compression='lzf')
        self._len += 1
        
    def __getitem__(self, index):
        x = np.array(self.file[str(index) + '_x'])
        y = float(self.file[str(index) + '_y'])
            
        return torch.from_numpy(x).to(float32), torch.Tensor([y]).to(float32)
    
    def __len__(self):
        return self._len
        
    def close(self):
        self.file.close()
    
    
    
def load_hmm(hmm_path):
    hmm = joblib.load(hmm_path)
    return hmm['model']



@torch.no_grad()
def compute_fid_scores(model, data, sample_seqs, batch_size=128, max_length=16,
                       cuda_device='cuda:0', verbose=False, log_every=10):
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
            verbose : bool, default = False, does not print messages
            
        Returns
        -------
            FID_scores : np.array
    """
    FID_scores = []
    device = torch.device(cuda_device)
    
    x_real = []
    x_fake = []

    for n in range(len(data)):
        x_r = data[n]
        x_f = sample_seqs[n]

        if len(x_real) < batch_size:

            if len(x_real)==0:
                x_real = x_r.reshape(1, x_r.shape[0], x_r.shape[1])
                x_fake = x_f.reshape(1, x_f.shape[0], x_f.shape[1])
                
            else:
                x_real = np.concatenate([x_real, x_r])
                x_fake = np.concatenate([x_fake, x_f])
            continue

        x_r, x_f = x_r.to(device), x_f.to(device)
        
        x_real, x_fake = [], []

        FID_scores.append(score)

        if verbose and n % log_every == 0:
            print('..... done with %.3f percent .....'%(n*100 / len(X)))
            print(f'..... FID: {score} .....')
        
    return np.array(FID_scores).mean()



def load_data_hdf(path_bird_hdf, bird):
    """Load one day's data. Tutor cane be loaded by passing day = 'tutor'.
    """
    return bird_dataset_single_hdf(path_bird_hdf, bird)
   
    
    
def make_hdf_for_classifier(dataset, netG, end_day, args):
    
    # make a temporary hdf file 
    hdf_dataset = HDFDataset(join(args.birdpath, 'fid_computation.hdf'))
    
    daily_lengths = []
    idx = 0
    for day in range(args.start_day, end_day):
        
        # load data
        X = dataset.get(day, nsamps=-1)
        Yreal = [1 for _ in range(len(X))] # labels
        
        daily_lengths.append(len(X))
        
        # load hmm model
        modelpath = join(args.birdpath, 'day_' + str(day),
                         'hmm_hiddensize_'+str(args.hidden_state_size),
                        'model_day_'+str(day)+'.pkl')
        
        hmm = load_hmm(modelpath)
        
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
            pdb.set_trace()
            hdf_dataset._add_data(x, str(idx) + '_x')
            hdf_dataset._add_data(y, str(idx) + '_y')
            idx += 1
            
        X = None
        sample_seqs = None
    
    hdf_dataset.close()
    return hdf_dataset, daily_lengths
        
    
    
    
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
                 dropout=0.0, lr=1e-4, l2=1e-4, bidirectional=False, log_every=100):
        super(InceptionNetRecurrent, self).__init__()
        
        self.encoder_rnn = nn.GRU(indims, nrnn, nrnnlayers, bidirectional=False,
                          dropout=dropout, batch_first=True)
        self.decoder_rnn = nn.GRU(nembed, nrnn, nrnnlayers, bidirectional=False,
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
    
    def encode(self, x):
        # x is a spectrogram of shape (N,H,L)
        # H = feature dims
        # L = length
        x = x.permute(0,2,1)
        
        self.len_in = x.shape[1]
        
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
            yhat = self.forward(x)
            
            loss = self.costfunc(yhat, y)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if i % self.log_every == 0:
                print('..... batch=%d/%d loss = %.3f .....'%(i,N,float(loss.detach())))
                
    
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--birdname', type=str, required=True)
parser.add_argument('--birdpath', type=str, required=True)
parser.add_argument('--path_to_bird_hdf', type=str, required=True)
parser.add_argument('--netGfilepath', type=str, required=True)
parser.add_argument('--hidden_state_size', type=int)
parser.add_argument('--start_day', type=int, default=0, help='first day to compute')
parser.add_argument('--end_day', type=int, default=-1, help='last day to compute')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--fid_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--npeochs', type=int, default=30, help='num training epochs')
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
    
    # load the generator
    print('..... loading generator .....')
    netGfilepath = join(args.birdpath, args.netGfilepath)
    netG = load_netG(netGfilepath, model_opts['nz'], model_opts['ngf'], model_opts['nc'],
                         True, resnet=True)
    
    dataset, daily_lengths = make_hdf_for_classifier(dataset, netG, end_day, args)
    # for read, open
    dataset._open()
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # make model 
    print('..... training model .....')
    model = InceptionNetRecurrent(args.in_dims,
                                  args.nrnn,
                                  args.nrnnlayers,
                                  args.nembed,
                                  args.nlin, args.leak,
                                  args.dropout,
                                  args.lr, args.l2,
                                  args.bidirectional,
                                  args.log_every)
    
    model.to(torch.device(args.cuda_device))
    
    # train
    for n in range(args.nepochs):
        model.fit(dataloader)
    
    model.eval()
    
    del dataloader 
    gc.collect()
    
    # compute scores
    print('..... computing scores .....')
    FID_scores = np.zeros(ndays)
    
    k = 0
    for j in range(ndays):
        
        real, _ = dataset[k : k + daily_lengths[j]]
        
        samples, _ = dataset[k + daily_lengths[j] : 2*(k + daily_lengths[j])]
        
        FID_scores[j] = compute_fid_scores(model, real,
                                           samples,
                                           args.fid_batch_size,
                                           args.max_length,
                                           args.cuda_device,
                                           verbose=True, 
                                           log_every=args.log_every)
        k += 2 * daily_lengths[j]
        
    dataset.close()
    # save the scores
    savepath = join(args.birdpath, 'hidden_state_size_'+ args.hidden_state_size +'_FID_scores.csv') 
    
    df = pd.DataFrame(data={'day': day_index, 'FID': FID_scores})
    
    df.to_csv(savepath, index=False)
    
    
    
if __name__ == '__main__':
    main()