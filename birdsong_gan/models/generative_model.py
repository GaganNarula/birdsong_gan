import torch
import numpy as np
from models.nets_16col_residual import _netG, _netE
from hmmlearn.hmm import GaussianHMM
from utils.utils import load_netG, load_netE, overlap_encode, overlap_decode
from hmm.hmm_utils import tempered_sampling
import joblib
import pdb
from typing import List


@torch.no_grad()
def overlap_decode(z, netG, noverlap = 0, get_audio = False, cuda = True):
    """Overlap decode. For a given numpy array Z of shape (timesteps , latent_dim)
        the output spectrogram (and optionally also audio) is created. 
        
        Parameters
        -------
            Z : numpy.ndarray, (timesteps , latent_dim)
            netG : generator neural network
            noverlap  : int, default = 0, how much overlap (in spectrogram frames) between 
                        consecutive spectrogram chunks
            get_audio : bool, default = False, to generate audio using Griffin Lim
            cuda : bool, default = True, if True pushes computation on gpu
        Returns
        -------
            X : numpy.ndarray, (nfft bins , chunks)
            X_audio : numpy array, reconstructed audio
    """
    X = [] # to store output chunks
    X_audio = None # for audio representation
    
    # in case only one chunk
    if z.ndim==1:
        z = np.expand_dims(z, axis=0) # add a timestep dimension
        
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).float()
        
    if cuda:
        z = z.cuda()
    
    for i in range(z.shape[0]):

        # reshape
        x = netG(z[i].view(1,z.size(-1))).cpu().numpy().squeeze()
        
        # take out any overlap slices
        # first slice is always fully accepted
        if i > 0:
            x = x[:, noverlap:]
            
        X.append(x)

    X = np.concatenate(X, axis=1) # concatenate in time dimension, axis=1
    
    if get_audio:
        X_audio = inverse_transform(X, N=500)
        X_audio = lc.istft(X_audio)
        
    return X, X_audio



def generate_samples(netG, hmm, nsamples=1, invtemp=1., timesteps=[], cuda=True):
    """Generate samples from trained netG and hmm"""
    seqs = [tempered_sampling(hmm, invtemp, timesteps=timesteps[i], 
                                sample_obs=True, start_state_max=True, 
                                 sample_var = 0.)[0] for i in range(nsamples)]
    # decode with netG
    seqs_out = [overlap_decode(seq, netG,  noverlap=0,
                                          cuda = cuda,
                               get_audio=False)[0] \
                for seq in seqs]
    
    return seqs_out



class GaussHMMGAN:
    
    netG: _netG # decoder / generator network
    netE: _netE
    hmm : GaussianHMM
    # num latent dims
    nz : int 
    # num hidden states
    K : int
    chunk_len : int = 16 # chunk length of spectrogram
        
    def __init__(self, netGpath, netEpath, hmmpath, ngf=128, chunk_len=16, cuda_device='cuda:0'):
        
        self.hmm = joblib.load(hmmpath)
        self.hmm = self.hmm['model']
        
        self.nz = self.hmm.n_features
        self.K = self.hmm.n_components
        self.chunk_len = chunk_len
        self.cuda_device = torch.device(cuda_device)
        
        self.netG = load_netG(netGpath, nz=self.nz, ngf=ngf, resnet=True, cuda=False)
        self.netE = load_netE(netEpath, nz=self.nz, ngf=ngf, resnet=True, cuda=False)
        self._to_cuda()
        
    def _to_cuda(self):
        self.netG = self.netG.to(self.cuda_device)
        self.netE = self.netE.to(self.cuda_device)
        
    def encode(self, x, transform_sample=False, cuda=True):
        
        # returns numpy array
        return overlap_encode(x, self.netE,
                              transform_sample,
                              imageW=self.chunk_len,
                              noverlap=0, cuda=cuda)
    
    def decode(self, z, noverlap=0, cuda=True):
           
        return overlap_decode(z, self.netG, noverlap, get_audio=False, cuda=cuda)
    
    def log_likelihood(self, x):
        z = self.encode(x)
        return self.log_likelihood_latent(z)
    
    def log_likelihood_latent(self, z):
        return self.hmm.score_samples(z)
    
    def reconstruct(self, x):
        z = self.encode(x)
        return self.decode(z)[0]
    
    def sample(self, nsamples=1, invtemp=1., timesteps=[], cuda=True):
        
        return generate_samples(self.netG, self.hmm, nsamples, invtemp, timesteps, cuda)
    
    def sample_latent(self, nsamples=1, invtemp=1., timesteps=[]):
        seqs = [tempered_sampling(self.hmm, invtemp, timesteps=timesteps[i], 
                                sample_obs=True, start_state_max=True, 
                                 sample_var = 0.)[0] for i in range(nsamples)]
        return seqs
    
    def generate_audio(self, sample_spectrograms: List[np.ndarray]):
        
        X_audio = [None for _ in range(len(sample_spectrograms))]
        
        for s in sample_spectrograms:
            # griffin lim
            x_audio = inverse_transform(s, N=500) 
            X_audio.append(lc.istft(x_audio))
            
        return X_audio
    
        