import torch
import numpy as np
from birdsong_gan.models.nets_16col_residual import _netG, _netE
from hmmlearn.hmm import GaussianHMM
from birdsong_gan.utils.utils import overlap_encode, overlap_decode, inverse_transform
from birdsong_gan.hmm.hmm_utils import tempered_sampling
import librosa as lc
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




def load_netG(netG_file_path, nz = 16, ngf = 128, nc = 1, cuda = False, resnet = False):
    """Load the generator network
    
        Params
        ------
            netG_file_path : str, location of decoder/generator network file (torch state_dict)
            nz : int, number of latent dimensions
            ngf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netG : decoder network
    """
    if resnet:
        from birdsong_gan.models.nets_16col_residual import _netG
    else:
        from birdsong_gan.models.nets_16col_layernorm import _netG
        
    netG = _netG(nz, ngf, nc)
    netG.load_state_dict(torch.load(netG_file_path))

    if cuda:
        netG = netG.cuda()
    return netG




def load_netE(netE_file_path, nz = 16, ngf = 128, nc = 1, cuda = False, resnet = False):
    """Load the encoder network
    
        Params
        ------
            netE_file_path : str, location of encoder network file (torch state_dict)
            nz : int, number of latent dimensions
            ngf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netE : encoder network
    """
    if resnet:
        from birdsong_gan.models.nets_16col_residual import _netE
    else:
        from birdsong_gan.models.nets_16col_layernorm import _netE
        
    netE = _netE(nz, ngf, nc)
    netE.load_state_dict(torch.load(netE_file_path))

    if cuda:
        netE = netE.cuda()
    return netE



def load_netD(netD_file_path, ndf = 128, nc = 1, cuda = False, resnet = False):
    """Load the discriminator network
    
        Params
        ------
            netD_file_path : str, location of encoder network file (torch state_dict)
            ndf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netE : encoder network
    """
    if resnet:
        from birdsong_gan.models.nets_16col_residual import _netD
    else:
        from birdsong_gan.models.nets_16col_layernorm import _netD
        
    netD = _netD(ndf, nc)
    netD.load_state_dict(torch.load(netD_file_path))

    if cuda:
        netD = netD.cuda()
    return netD



def load_InceptionNet(inception_net_file_path, ndf = 128, nc = 1, cuda = False, resnet = False):
    """Load the inception discriminator network
    
        Params
        ------
            inception_net_file_path : str, location of encoder network file (torch state_dict)
            ndf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netE : encoder network
    """
    if resnet:
        from birdsong_gan.models.nets_16col_residual import InceptionNet
    else:
        from birdsong_gan.models.nets_16col_layernorm import InceptionNet
        
    netI = InceptionNet(ndf, nc)
    netI.load_state_dict(torch.load(inception_net_file_path))

    if cuda:
        netI = netI.cuda()
    return netI




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
        self.netG.eval()
        self.netE.eval()
        
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
        
        X_audio = []
        
        for s in sample_spectrograms:
            # griffin lim
            x_audio = inverse_transform(s, N=500) 
            X_audio.append(lc.istft(x_audio))
            
        return X_audio
    
        