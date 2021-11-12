r"""Module contains code to judge the quality of generative models. 

References
----------
..[1] Alaa et al. 2021, How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative models
        arxiv.org/pdf/2102.08921.pdf
"""


import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors



class EmbeddingNet(nn.Module):
    """Embed real data and generated data into a lower dimensional embedding space.
    """
    def __init__(self, indims=129, nembed=3, nlin=50, leak=0.1,
                 nrnn=50, nrnnlayers=3, bidirectional=True, dropout=0.0):
        
        super(EmbeddingNet, self).__init__()
        
        self.rnn = nn.GRU(indims, nrnn, nrnnlayers, bidirectional=bidirectional,
                          dropout=dropout, batch_first=True)
        
        bid = 2 if bidirectional else 1
        
        self.mlp = nn.Sequential(
            nn.Linear(nrnn*bid, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nembed),
            nn.LayerNorm(nembed),
        )
        
    def forward(self, x):
        # x is a spectrogram of shape (N,H,L)
        # H = feature dims
        # L = length
        x,_ = self.rnn(x)
        return self.mlp(x)
    
    
def make_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x
    
    
def one_class_SVM_loss(x, center, radius, nu):
    r"""Compute One class SVM loss for data points x.
    ..math:
        L = \sum(l_i)
        l_i = r^2 + (1/nu) * max(0, ||x_i - center||^2 - r^2)
    """
    # for each sample compute L2 distance to center
    
    n = embedded_x.shape[0] # num samples
    
    center = torch.tile(center, (n,1))
    
    l1 = nn.MSELoss(embedded_x, center)
    l2 = nn.functional.relu(l1 - radius**2)
    
    return radius**2 + (1/nu)*l2


pdist = nn.PairwiseDistance(p=2)


def compute_quantile_radius(x, center, alpha=0.95):
    """Given a set of data points in embedded / data space and a center
        returns: 
    """ 
    # pairwise distances between each point x and center 
    D = pdist(x, center)
    
    return torch.quantile(D, alpha), D
    
    
def precision_classifier(x, center, alpha):
    # make radius
    r_alpha = compute_quantile_radius(x, center, alpha)
    return pdist(x, center) <= r_alpha


def knn(x, K=10):
    """Make a sklearn.neighbors object for datapoints in tensor x
        x has shape (N, D), N = num points
    """
    x = make_numpy(x)
    
    return NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(x)
    
    
    
def recall_classifier(real_x, fake_x_pts, nearest_neighbors, K=10, beta=0.95):
    """
    1. Make a Ball(center_g, radius_beta). To do so, use the fake_x_pts 
        to get the quantile radius, eliminate fake pts outside the quantile radius.
    2. Find the nearest neighbor distance of real_x to all other reals
    3. Find the nearest fake to real_x, lying in the ball B(c_g, r_beta)
    4. Return : Is distance between nearest_fake and real_x <= NND ? 
    
    """
    # make center the average fake data
    c_g = fake_x_pts.mean(dim=0)
    # get ball B radius
    r_beta = compute_quantile_radius(fake_x_pts, c_g, beta)
    # which pts are in the ball ? 
    in_ball = pdist(fake_x_pts, c_g) <= r_beta
    
    # find the K nearest neighbors of real_x in all other reals
    rx = make_numpy(real_x)
    d, _ = nearest_neighbors.kneighbors(rx, K)
    NND = d[-1]
    
    # find nearest pt to real_x in ball B(c_g, r_beta)
    D = pdist(fake_x_pts[in_ball], real_x)
    nearest_fake = torch.argmin(D)
    
    # is nearest fake in Ball(real_x, NND) ? 
    return pdist(nearest_fale, real_x) <= NND
    
    

def authenticity(fake_x, real_x_pts):
    pass
        
        
