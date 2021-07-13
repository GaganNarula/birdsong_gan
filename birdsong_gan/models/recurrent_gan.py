"""Recurrent GAN is using recurrent networks as encoders, generators (or decoders) and discrimnators.


    The ideas is as follows 
    
    ::math
    
    X_n = (x_1, x_2, ... x_{T_n})
    
    There is some dependence on time as well. 
    
    Approximate posterior Q(z_t | X_n) = func(rnn(conv(x_t)))
"""
import torch
import torch.nn as nn
import numpy as np
from hmmlearn.hmm import GaussianHMM
from torch.nn.utils import spectral_norm
import pdb


def make_mlp(n_in, nhidden, nout, leak=0.1):
    """Single hidden layer feedforward net"""
    m = nn.Sequential(
            nn.Linear(n_in, nhidden),
            nn.LayerNorm(nhidden),
            nn.LeakyReLU(leak),
            
            nn.Linear(nhidden, nhidden),
            nn.LayerNorm(nhidden),
            nn.LeakyReLU(leak),
            
            nn.Linear(nhidden, nout),
        )
    return m


def make_downsampling_cnn(nrnn_input=50, ngf=128, leak=0.1, spec_norm=False):
    """This architecture is specific to spectrogram chunk length of 16 """
    if not spectral_norm:
        m = nn.Sequential(
              spectral_norm(nn.Conv2d(1, ngf, kernel_size=(5, 4), stride=(2, 2), padding=(0,1), bias=False)),
              # H = ((129 -5)/2 + 1 = 63, W = (16 + 2 - 4)/2 + 1 = 8
              nn.LayerNorm([ngf, 63, 8]),
              nn.LeakyReLU(leak),
              spectral_norm(nn.Conv2d(ngf, ngf * 2, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0), bias=False)),
              # size H = (63 +2 -5)/2 +1 = 31, W = (8 - 4) + 1 = 5
              nn.LayerNorm([ngf*2, 31, 5]),
              nn.LeakyReLU(leak),
              # down sample
              spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=(5, 4), stride=(4, 1), padding=(1, 1), bias=False)),
              # size H = (31 +2 - 5)/4 + 1 = 8, W = (5 +2 -4) + 1 = 4
              nn.LayerNorm([ngf*4, 8, 4]),
              nn.LeakyReLU(leak),
              spectral_norm(nn.Conv2d(ngf*4, ngf*8, kernel_size=(4, 4), stride=(3, 1), padding=(1, 1), bias=False)),
              # size H = (8 +2 -4)/3 + 1 = 3, W = (4 +2 -4) + 1 = 3
              nn.LayerNorm([ngf*8, 3, 3]),
              nn.LeakyReLU(leak),
              spectral_norm(nn.Conv2d(ngf * 8, nrnn_input, kernel_size=(3,3), stride=1, padding=0, bias=False)),
              # shape is (N, nrnn_input, 1, 1)
        )
    else:
        m = nn.Sequential(
              nn.Conv2d(1, ngf, kernel_size=(5, 4), stride=(2, 2), padding=(0,1), bias=False),
              # H = ((129 -5)/2 + 1 = 63, W = (16 + 2 - 4)/2 + 1 = 8
              nn.LayerNorm([ngf, 63, 8]),
              nn.LeakyReLU(leak),
              nn.Conv2d(ngf, ngf * 2, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0), bias=False),
              # size H = (63 +2 -5)/2 +1 = 31, W = (8 - 4) + 1 = 5
              nn.LayerNorm([ngf*2, 31, 5]),
              nn.LeakyReLU(leak),
              # down sample
              nn.Conv2d(ngf*2, ngf * 4, kernel_size=(5, 4), stride=(4, 1), padding=(1, 1), bias=False),
              # size H = (31 +2 - 5)/4 + 1 = 8, W = (5 +2 -4) + 1 = 4
              nn.LayerNorm([ngf*4, 8, 4]),
              nn.LeakyReLU(leak),
              nn.Conv2d(ngf*4, ngf*8, kernel_size=(4, 4), stride=(3, 1), padding=(1, 1), bias=False),
              # size H = (8 +2 -4)/3 + 1 = 3, W = (4 +2 -4) + 1 = 3
              nn.LayerNorm([ngf*8, 3, 3]),
              nn.LeakyReLU(leak),
              nn.Conv2d(ngf * 8, nrnn_input, kernel_size=(3,3), stride=1, padding=0, bias=False),
              # shape is (N, nrnn_input, 1, 1)
        )
    return m


def make_upsampling_cnn(nz, ngf, spec_norm=False):
    """ This architecture is specific to spectrogram chunk length of 16 """
    if not spec_norm:
        m = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf*8, kernel_size=(3,3), stride=1, padding=0, bias=False),
                # Hout = (Hin − 1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+outpad+1
                # size is H = 3, W = 3
                nn.LayerNorm([ngf*8, 3, 3]),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf*8, ngf * 4, kernel_size=(4, 4), stride=(3, 1),
                                   padding=(1, 1), bias=False),
                # size is H = 8, W = 4
                nn.LayerNorm([ngf*4, 8, 4]),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=(5, 4), stride=(4, 1),
                                   padding=(1, 1), bias=False),
                nn.LayerNorm([ngf*2, 31, 5]),
                nn.ReLU(True),
                # size is H = 31, W = 5
                nn.ConvTranspose2d(ngf*2, ngf, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0),
                                   bias=False),
                nn.LayerNorm([ngf, 63, 8]),
                nn.ReLU(True),
                # size is H = 63, W = 8
                nn.ConvTranspose2d(ngf, 1, kernel_size=(5, 4), stride=(2, 2), padding=(0,1),
                                   bias=True),
               # size is H = 129, W = (8-1)*2  -2 + 4 = 16
                nn.Softplus()
            )
    else:
        m = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(3,3), stride=1, padding=0, bias=False)),
                # Hout = (Hin − 1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+outpad+1
                # size is H = 3, W = 3
                nn.LayerNorm([ngf*8, 3, 3]),
                nn.ReLU(True),
                spectral_norm(nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=(4, 4), stride=(3, 1),
                                   padding=(1, 1), bias=False)),
                # size is H = 8, W = 4
                nn.LayerNorm([ngf*4, 8, 4]),
                nn.ReLU(True),
                spectral_norm(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=(5, 4), stride=(4, 1),
                                   padding=(1, 1), bias=False)),
                nn.LayerNorm([ngf*2, 31, 5]),
                nn.ReLU(True),
                # size is H = 31, W = 5
                spectral_norm(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0),
                                   bias=False)),
                nn.LayerNorm([ngf, 63, 8]),
                nn.ReLU(True),
                # size is H = 63, W = 8
                spectral_norm(nn.ConvTranspose2d(ngf, 1, kernel_size=(5, 4), stride=(2, 2), padding=(0,1),
                                   bias=True)),
               # size is H = 129, W = (8-1)*2  -2 + 4 = 16
                nn.Softplus()
            )
        
    return m



def makeLRtransmat(self, K):
    """Creates a Left-right transition matrix for a markov chain"""
    transmat = np.zeros((K, K))
    for k in range(K-1):
        transmat[k,k] = 0.5
        transmat[k,k+1] = 0.5
    transmat[-1,-1] = 1.
    return transmat


    
class RecurrentGAN(nn.Module):
    """This is a GAN model that uses a recurrent network in the encoder and discriminator.
    
    """
    def __init__(self, rnn_input_dim=50, nz=16, nrnn=200, nlin=200, nlayers=1, bidirectional=False, 
                 dropout=0.1, leak=0.1, ngf=64, imageW=16, hmm_components=20, 
                cuda=True, spectral_norm_decoder=False, spectral_norm_discriminator=False):
        super(RecurrentGAN, self).__init__()
        
        self.imageW = imageW
        self.cuda = cuda
        self.nz = nz
        bid = 2 if bidirectional else 1
        
        # define encoder
        self.encoder = nn.ModuleList([make_downsampling_cnn(rnn_input_dim, ngf, spec_norm=False),
                        nn.GRU(rnn_input_dim, nrnn, nlayers, bidirectional=bidirectional, 
                               dropout=dropout, batch_first=True),
                        make_mlp(nrnn*bid, nlin, nz, leak)
                       ])
        # generator
        self.decoder = make_upsampling_cnn(nz, ngf, spectral_norm_decoder)
        
        # discriminator fake vs real
        self.disc_FvR = nn.ModuleList([make_downsampling_cnn(rnn_input_dim, ngf, spectral_norm_discriminator), 
                         nn.GRU(rnn_input_dim, nrnn, nlayers, bidirectional=bidirectional, 
                                dropout=dropout, batch_first=True),
                         make_mlp(nrnn*bid*nlayers, nlin, 1, leak)
                        ])
        # discriminator reconstruction vs real
        self.disc_RevR = nn.ModuleList([make_downsampling_cnn(rnn_input_dim, ngf, spectral_norm_discriminator), 
                         nn.GRU(rnn_input_dim, nrnn, nlayers, bidirectional=bidirectional, 
                                dropout=dropout, batch_first=True),
                         make_mlp(nrnn*bid*nlayers, nlin, 1, leak)
                         ])
        # this network is used to compute Frechet Inception Distance
        self.inception_net = nn.ModuleList([make_downsampling_cnn(rnn_input_dim, ngf, spectral_norm_discriminator), 
                         nn.GRU(rnn_input_dim, nrnn, nlayers, bidirectional=bidirectional, 
                                dropout=dropout, batch_first=True),
                         make_mlp(nrnn*bid*nlayers, nlin, 1, leak)
                         ])
        
        # sampling via hmm
        self._init_hmm(hmm_components)
        self.sigmoid = nn.Sigmoid()
        
        if cuda:
            self._models_to_gpu()
            
    def _models_to_gpu(self):
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        self.disc_FvR = self.disc_FvR.cuda()
        self.disc_RevR = self.disc_RevR.cuda()
        self.inception_net = self.inception_net.cuda()
        
    def _init_hmm(self, n_components, transmat_prior=1.):
        self.hmm = GaussianHMM(n_components, covariance_type='diag',
                                       transmat_prior=transmat_prior, 
                                        init_params = 'mc')
        #intialize randomly
        self.hmm.transmat_ = np.random.dirichlet(transmat_prior*np.ones(n_components),
                                                 size = n_components)
        # or initialize as Left right transmat
        #self.hmm.transmat_ = makeLRtransmat(n_components)
        self.hmm.startprob_ = np.random.dirichlet(transmat_prior * np.ones(n_components))
        
        fake_init_data = np.random.multivariate_normal(mean= np.zeros(self.nz),
                                                       cov=np.eye(self.nz), 
                                                       size = 1000)
        self.hmm._init(fake_init_data)
        
    def _chunk_and_convolve(self, x, model):
        """Split spectrogram into a series of chunks, convolve
            each and stack into shape (N, L, C)
        """
        h = []
        i = 0
        notdone=True
        while notdone:
            s = x[:,:,i:i+self.imageW].view(-1,1,x.shape[1],self.imageW)
            s = model(s).squeeze()
            # s has shape (N, rnn_in)
            h.append(s)
            i += self.imageW
            if i + self.imageW > x.shape[-1]:
                notdone=False
    
        return torch.stack(h, dim=1)
    
    def discriminate(self, x, model):
        """Convolutions -> downsampling -> RNN -> label
        """
        # fake vs real
        x = self._chunk_and_convolve(x, model[0])
        # x has shape (N, L, rnn_in)
        # run encoder, get hidden state 
        _, h = model[1](x)
        h = h.permute(1,0,2)
        # map hidden to single 
        h = h.view(x.size(0), h.size(1)*h.size(2))

        return model[2](h)
            
    def encode(self, x):
        """Encode to latent variable
            x has shape (N, imageH, imageW)
        """
        x = self._chunk_and_convolve(x, self.encoder[0])
        # run rnn   
        o,_ = self.encoder[1](x)
        # map each step to latent
        z = []
        for t in range(o.size(1)):
            z.append(self.encoder[2](o[:,t,:]))
            
        return torch.stack(z,dim=1)
        
    def decode(self, z):
        """ Here, z is a trajectory in latent space
            Decodes chunks for every z
            Shape of z = (N, L, nz)
        """
        x_hat = []
        N = z.size(0) # batch size
        for t in range(z.size(1)):
            z_in = z[:,t,:].view(N, z.size(2), 1, 1)
            x_hat.append(self.decoder(z_in).squeeze())
            
        return torch.cat(x_hat,dim=-1)
        
    def forward(self, x):
        """forward call is a reconstruction"""
        return self.decode(self.encode(x))
    
    def prior_sample(self, nsamples=1, T=100, to_tensor=False):
        """Gaussian HMM samples"""
        z = np.stack([self.hmm.sample(T)[0] for _ in range(nsamples)])
        if to_tensor:
            z = torch.from_numpy(z).float()
            if self.cuda:
                z = z.cuda()
        return z
        
    
    
    
    
        