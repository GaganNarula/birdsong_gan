import torch
import torch.nn as nn
import pdb

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
        bid = 2 if bidirectional else 1
        
        self.mlp = nn.Sequential(
            nn.Linear(nrnn*bid, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nz),
            nn.LayerNorm(nz),
            nn.LeakyReLU(leak),
        )
        
        self.sp = nn.Softplus()
        self.mu_map = nn.Linear(nz, nz)
        self.log_sigma_map = nn.Linear(nz, nz)
        
    def forward(self, x):
        # x has shape (N, L, Hin)
        o, _ = self.rnn(x)
        # map each
        z = []
        for t in range(o.size(1)):
            h = self.mlp(o[:,t,:])
            mu = self.mu_map(h)
            logvar = self.log_sigma_map(h)
            # sample output
            z.append(self.sample(mu, logvar))
        
        return torch.stack(z,dim=1)
    
    def sample(self, mu, logvar):
        noise = torch.randn_like(mu)
        x = []
        for i in range(logvar.shape[0]):
            V = torch.diag(logvar[i].exp())
            x.append(mu[i] + torch.matmul(V, noise[i]))
        x = torch.stack(x)
        
        return x
    
    def split_input(self, x, ksteps_ahead):
        """splits input tensor on time axis as input and target"""
        return x[:,:-ksteps_ahead,:], x[:,ksteps_ahead:,:]

    
    
    

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                