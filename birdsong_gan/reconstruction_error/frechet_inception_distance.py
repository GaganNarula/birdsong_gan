r"""Compute the FID score of a model. A model consists of a combination of an HMM and GAN generater/decoder.
    
    One Inception network is learned from some subset of data.
    
    The Inception
"""


import os
import numpy as np
import json
import torch 
import torch.nn as nn
import h5py
import gc
import joblib
from glob import glob

from utils.utils import load_netG, segment_image, load_InceptionNet
from hmm.hmm_utils import generate_samples
from models.nets_16col_residual import InceptionNet, FID
from data.dataset import bird_dataset_single_hdf
from torch.utils.data import TensorDataset, DataLoader
import argparse



def make_tensor_dataloader(data, sample_sequences, batch_size=32, chunk_len=16):
    """Creates dataloader to train network
        Parameters
        ----------
            data : list of 2D np.array. Transformed spectrograms sequences
            sample_sequences : list of 2D np.array
            batch_size : int, batch size for dataloader
            chunk_len : int, how long is a spectrogram chunk
        Returns
        ----------
            torch.utils.data.DataLoader
        
    """
    real = []
    fake = []
    
    for x,s in zip(data, sample_seqs):

        real.append(np.stack(segment_image(x, width = chunk_len)))
        fake.append(np.stack(segment_image(s, width = chunk_len)))

    real = np.concatenate(real)
    fake = np.concatenate(fake)

    x = np.concatenate([real, fake], axis=0)

    x = torch.from_numpy(x).view(-1, 1, real.shape[1], real.shape[2]).to(torch.float32)
    
    y = np.concatenate([np.ones((real.shape[0], 1)), np.zeros((fake.shape[0],1))])
    y = torch.from_numpy(y).to(torch.float32)
    
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    
def train_model(model, dataloader, nepochs=5):
    
    for n in range(nepochs):
        model.fit(dataloader)
        
    return model


    
    
def load_hmm(hmm_path):
    hmm = joblib.load(hmm_path)
    return hmm['model']


@torch.no_grad()
def compute_fid_scores(model, data, sample_seqs, min_samps=400, chunk_len=16,
                       cuda_device='cuda:0', verbose=False, log_every=10):
    """Computes the Frechet Inception Distance by taking the feature vector of the second last layer
        of the InceptionNet model and computing the frechet distance between a set of min_samps real
        and min_samps fake data.
        
        Parameters
        ----------
            model : InceptionNet model, already learned
            data : list of np.arrays, real data (log transformed spectrograms)
            sample_seqs : list of np.arrays, same shape as real data, but are samples from the model.
            min_samps : int, minimum number of samples in a FID computation, default = 400
            chunk_len : int, chunk length for spectrograms, default = 16
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

    for n in range(len(X)):

        x_r = np.stack(segment_image(X[n], width = 16))
        x_f = np.stack(segment_image(sample_seqs[n], width = 16))


        if len(x_real) < min_samps:

            if len(x_real)==0:
                x_real = x_r * 1.
                x_fake = x_f * 1.
            else:
                x_real = np.concatenate([x_real, x_r])
                x_fake = np.concatenate([x_fake, x_f])

            continue

        x_r = torch.from_numpy(x_real).view(-1, 1, x_real.shape[1], x_real.shape[2]).to(torch.float32)
        x_f = torch.from_numpy(x_fake).view(-1, 1, x_fake.shape[1], x_fake.shape[2]).to(torch.float32)
        
        x_r, x_f = x_r.to(device), x_f.to(device)
        
        x_real, x_fake = [], []

        score = model.fid_score(x_r, x_f)

        FID_scores.append(score)

        if verbose and n % log_every == 0:
            print('..... done with %.3f percent .....'%(n*100 / len(X)))
            print(f'..... FID: {score} .....')
        
    return np.array(FID_scores)



def get_fid_score(data, sample_seqs, 
                  model=None,
                  batch_size=32, 
                  min_samps=400
                  lr=5e-5,
                  num_filters=8,
                  log_every = 100,
                  nepochs=5,
                  weight_decay = 0.0,
                  cuda_device='cuda:0'):
    """Computes FID scores for real 'data' and sample sequences
    """
    dataloader = make_tensor_dataloader(data, sample_sequences, batch_size=batch_size,
                                        chunk_len=chunk_len)
    cuda_device = torch.device(cuda_device)

    if models is None:
        # build one
        model = InceptionNet(ndf=num_filters, nc=1, lr=lr, l2=weight_decay, log_every=log_every)
        model = model.to(cuda_device)
    
    # train the network (may use previously initialized one)
    model = train_model(model, dataloader, nepochs)
    
    # compute the scores, return scores and model
    return compute_fid_scores(model, data, samples_seqs, min_samps, chunk_len, cuda_device), model




def load_data_hdf(path_bird_hdf, bird):
    """Load one day's data. Tutor cane be loaded by passing day = 'tutor'.
    """
    return bird_dataset_single_hdf(path_bird_hdf, bird)
    
    

parser = argparse.ArgumentParser()
parser.add_argument('--birdname', type=str, required=True)
parser.add_argument('--birdpath', type=str, required=True)
parser.add_argument('--path_to_bird_hdf', type=str, required=True)
parser.add_argument('--netGfilepath', type=str, required=True)
parser.add_argument('--hidden_state_size', type=int)
parser.add_argument('--start_day', type=int, default=0)
parser.add_argument('--end_day', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--npeochs', type=int, default=30)
parser.add_argument('--chunk_len', type=int, default=16)


def main():
    
    args = parser.parse_args()
    
    dataset = load_data_hdf(args.path_to_bird_hdf, args.birdname)
    
    days = glob(join(args.birdpath, 'day_*'))
    ndays = len(days)
    end_day = ndays if args.end_day==-1 else args.end_day
    assert end_day <= nday, 'end day has to be <= ndays'
    
    # model opts
    with open(join(args.birdpath, 'opts.json'), 'rb') as f:
        model_opts = json.load(f)
    
    # load the generator
    netGfilepath = join(args.birdpath, args.netGfilepath)
    netG = load_netG(netGfilepath, model_opts['nz'], model_opts['ngf'], model_opts['nc'],
                        args.cuda, resnet=True)
    
    daily_data = []
    daily_sample_seqs = []
        
    for day in range(args.start_day, args.end_day):
        
        # load data
        X = dataset.get(day, nsamps=-1)
        
        daily_data += X
        
        # load hmm model
        modelpath = join(args.birdpath, 'day_' + str(day),
                         'hmm_hiddensize_'+str(args.hidden_state_size),
                        'model_day_'+str(day)+'.pkl')
        
        hmm = load_hmm(modelpath)
        
        # generate sample sequences
        samples_seqs = generate_samples(netG, hmm, nsamples=len(X), invtemp=1.,
                         timesteps=[x.shape[1] for x in X],
                         cuda=True)
        
        daily_sample_seqs += samples_seqs
        
        
    dataloader = make_tensor_dataloader(data, sample_sequences, batch_size=batch_size,
                                        chunk_len=chunk_len)
    
    