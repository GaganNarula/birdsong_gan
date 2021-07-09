"""Recurrent GAN is using recurrent networks as encoders, generators (or decoders) and discrimnators.


    The ideas is as follows 
    
    ::math
    
    X_n = (x_1, x_2, ... x_{T_n})
    
    There is some dependence on time as well. 
    
"""

def make_mlp(n_in, nhidden, nout, leak=0.1):
    m = nn.Sequential(
            nn.Linear(n_in, nhidden),
            nn.LayerNorm(nhidden),
            nn.LeakyReLU(leak),
            
            nn.Linear(nhidden, nhidden),
            nn.LayerNorm(nhidden),
            nn.LeakyReLU(leak),
            
            nn.Linear(nhidden, n_out),
            nn.LayerNorm(n_out)
        )
    return m


def make_downsampling_cnn(n_in, nfilter, kernel_size, padding):
    
    
class RecurrentGAN(nn.Module):
    """A simple recurrent network using spectrogram columns, predicting spectrograms columns 
        K steps ahead. Output is mean and covariance of a multivariate gaussian describing
        the spectrogram column. For simplicity, the covariance is fixed to diagnoal only.
    """
    def __init__(self, Hin=129, nz=129, nrnn=200, nlin=200, nlayers=2, bidirectional=False, 
                 dropout=0.1, leak=0.1):
        super(RecurrentGAN, self).__init__()
        
        self.encoder_rnn = nn.GRU(Hin, nrnn, nlayers, bidirectional=bidirectional, dropout=dropout,
                         batch_first=True)
        
        self.decoder_rnn = nn.GRU(nz, nrnn, nlayers, bidirectional=bidirectional, dropout=dropout,
                         batch_first=True)
        
        bid = 2 if bidirectional else 1
        
        self.encoder_mlp = make_mlp(nrnn*bid, nlin, nz, leak)
        
        self.decoder_mlp = make_mlp(n)
        
        self.lrelu = nn.LeakyReLU(leak)
        self.sp = nn.Softplus()
        
    def encode(self, x):
        """Encode latent variable """'
        o,_ = self.encoder_rnn(x)
        # map each step to latent
        z = []
        for t in range(o.size(1)):
            z.append(self.encoder_mlp(o[:,t,:]))
            
        return torch.stack(z,dim=1)
        
    def decode(self, z):
        """ Here, z is a trajectory in latent space"""
        o,_ = self.decoder_rnn(z)
        # map each step to latent
        x = []
        for t in range(o.size(1)):
            x.append(self.decoder_mlp(o[:,t,:]))
            
        return torch.stack(x,dim=1)
        
    def forward(self, x):
        """forward call is a reconstruction"""
        return self.decode(self.encode(x))
    
    def sample(self, mu, logvar):
        noise = torch.randn_like(mu)
        x = []
        for i in range(logvar.shape[0]):
            V = torch.diag(logvar[i].exp())
            x.append(mu[i] + torch.matmul(V, noise[i]))
        x = torch.stack(x)
        
        return x
    
    def prior_sample(self, T=100):
        """Multidimensional Brownian motion"""
        return
        
    def split_input(self, x, ksteps_ahead):
        """splits input tensor on time axis as input and target"""
        return x[:,:-ksteps_ahead,:], x[:,ksteps_ahead:,:]
    
    
    
    
def train(model, dataloader, opts):
    
        