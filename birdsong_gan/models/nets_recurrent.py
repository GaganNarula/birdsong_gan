import torch
import torch.nn as nn
from collections import namedtuple


class RecurrentNetv1(nn.Module):
    """A simple recurrent network using spectrogram columns, predicting spectrograms columns 
        K steps ahead. Output is mean and covariance of a multivariate gaussian describing
        the spectrogram column. For simplicity, the covariance is fixed to diagnoal only.
    """
    def __init__(self, Hin=129, nz=129, nrnn=200, nlin=200, nlayers=2, bidirectional=False, 
                 dropout=0.1, leak=0.1):
        super(RecurrentNetv1, self).__init__()
        self.rnn = nn.GRU(Hin, nrnn, nlayers, bidirectional=bidirectional, dropout=dropout,
                         batch_first=True)
        bid=1
        if bidirectional:
            bid=2
            
        self.mlp = nn.ModuleList([
            nn.Linear(nrnn*nlayers*bid, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nz),
            nn.LayerNorm(nz),
            nn.LeakyReLU(leak),
        ])
        
        self.sp = nn.Softplus()
        self.mu_map = nn.Linear(nz, nz)
        self.log_sigma_map = nn.Linear(nz, nz)
        
    def forward(self, x):
        # x has shape (N, L, Hin)
        o, _ = self.rnn(x)
        # map each
        for t in range(o.size(1)):
            h = self.mlp(o[:,t,:])
            mu = self.mu_map(h)
            logvar = self.log_sigma_map(h)
            # sample output
            
        return torch.cat(z)
    
    def sample(self, mu, logvar):
        
        
    def split_input(self, x, ksteps_ahead)
        """splits input tensor on time axis as input and target"""
        return x[:,:-ksteps_ahead,:], x[:,ksteps_ahead:,:]

    
    
    
def train(model, dataloader, opts):
    """Traning function. 
        Params
        ------
            model : 
            dataloader : torch.utils.data.Dataloader map style
            opts : options (dict or namedtuple)
                    - nepochs : int, how many epochs of training
    """
    if isinstance(opts,dict):
        # convert to named tuple
        opts = namedtuple('Options',opts)

    for n in range(opts.nepochs):

        for i, (x,age) in enumerate(dataloader):
            # make sure the shape is (N,L,H)
            x = x.permute(0,2,1)

            # input is past values, predicted are future
            # split the tensor on the time axis to get target
            # and input 
            x, y = model.split_input(x)

            yhat = model(x)
            
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                