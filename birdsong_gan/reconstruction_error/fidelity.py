r"""Module contains code to judge the quality of generative models. 

References
----------
..[1] Alaa et al. 2021, How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative models
        arxiv.org/pdf/2102.08921.pdf
"""
import sys
import os
sys.path.append(os.pardir)

import warnings
warnings.filterwarnings("ignore")

from os.path import join
import joblib
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from hmm.hmm_utils import tempered_sampling
from utils.utils import load_netG, overlap_decode
from data.dataset import bird_dataset_single_hdf
import json
import pdb



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
            
            nn.Linear(nlin, nembed)
        )
        
    def forward(self, x):
        # x is a spectrogram of shape (N,H,L)
        # H = feature dims
        # L = length
        x = x.permute(0,2,1)
        x,_ = self.rnn(x)
        # take last time step value
        x = x[:,-1,:]
        return self.mlp(x)
    
    
class AutoencoderNet(nn.Module):
    """Embed real data and generated data into a lower dimensional embedding space.
        Use autoencoding loss for embedding
    """
    def __init__(self, indims=129, nembed=3, nlin=50, leak=0.1,
                 nrnn=50, nrnnlayers=3, bidirectional=True, dropout=0.0):
        
        super(AutoencoderNet, self).__init__()
        
        self.encoder_rnn = nn.GRU(indims, nrnn, nrnnlayers, bidirectional=bidirectional,
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
        
        self.mlp_out = nn.Sequential(
            nn.Linear(nrnn, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, nlin),
            nn.LayerNorm(nlin),
            nn.LeakyReLU(leak),
            
            nn.Linear(nlin, indims)
        )
        
        self.len_in = None
        
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
    
    def decode(self, z):
        # tile z 
        z = z.view(-1,1,z.shape[-1])
        z = torch.tile(z, (1, self.len_in, 1))
        
        xhat, _ = self.decoder_rnn(z)
        xhat = self.mlp_out(xhat).permute(0,2,1)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
    
    
    
    
def make_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x
    
    

def one_class_SVM_loss(embedded_x, center, radius, nu):
    r"""Compute One class SVM loss for data points x.
    ..math:
        L = \sum(l_i)
        l_i = r^2 + (1/nu) * max(0, ||x_i - center||^2 - r^2)
    """
    # for each sample compute L2 distance to center
    n = embedded_x.shape[0] # num samples
    l1 = torch.pow(embedded_x - center, 2).sum(dim=-1)
    l2 = nn.functional.relu(l1 - radius**2)
    
    return (radius**2 + (1/nu)*l2).sum()


pdist = nn.functional.pairwise_distance


def compute_quantile_radius(x, center, alpha=0.95):
    """Given a set of data points in embedded / data space and a center
        returns: 
    """ 
    # pairwise distances between each point x and center 
    D = pdist(x, center)
    
    return torch.quantile(D, alpha)
    
    
def precision_classifier(fake_x, center, r_alpha):
    """Precision classifier tests if the fake sample 'fake_x'
        outside of the alpha ball around reals B(center_r, r_alpha)
    """
    pred = pdist(fake_x.view(1,-1), center.view(1,-1)) <= r_alpha
    return float(pred.detach())



def alpha_precision_curve(fake_pts, real_pts, opts, alphas=np.linspace(0.1, 1.,50)):
    """Summary Precision values for a range of alphas
    """
    N = len(real_pts) # num points
    
    # radius of the real Ball 
    center_r = torch.zeros(opts['nembed'])
    if opts['cuda']:
        center_r = center_r.cuda()
    
    precision_curve = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        
        r_alpha = compute_quantile_radius(real_pts, center_r, alpha)
        precision_curve[i] = np.mean([precision_classifier(fake_pts[i], center_r, r_alpha) for i in range(N)])
    
    return precision_curve
    
    

def beta_recall_curve(fake_pts, real_pts, nearest_neighbors, opts, betas=np.linspace(0.1, 1.,50)):
    """Summary Recall values for a range of beta values
    """
    N = len(real_pts) # num points
    
    c_g = fake_pts.mean(dim=0)
    
    recall_curve = np.zeros(len(betas))
    
    for i, beta in enumerate(betas):
        # get ball B radius for ball of generated samples
        r_beta = compute_quantile_radius(fake_pts, c_g, beta)

        recall_curve[i] = np.mean([recall_classifier(y, fake_pts, nearest_neighbors, r_beta,
                                                     opts['kneighbors']) for y in real_pts])
        
    return recall_curve



def knn(x, K=10):
    """Make a sklearn.neighbors object for datapoints in tensor x
        x has shape (N, D), N = num points
    """
    x = make_numpy(x)
    
    return NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(x)
    
    
    
def recall_classifier(real_x, fake_x_pts, nearest_neighbors, r_beta=None, K=10, beta=0.95):
    """
    1. Make a Ball(center_g, radius_beta). To do so, use the fake_x_pts 
        to get the quantile radius, eliminate fake pts outside the quantile radius.
    2. Find the nearest neighbor distance of real_x to all other reals
    3. Find the nearest fake to real_x, lying in the ball B(c_g, r_beta)
    4. Return : Is distance between nearest_fake and real_x <= NND ? 
    
    """
    # make center the average fake data
    c_g = fake_x_pts.mean(dim=0).view(1,-1)
    
    if r_beta is None:
        # get ball B radius
        r_beta = compute_quantile_radius(fake_x_pts, c_g, beta)
        
    # which pts are in the ball ? 
    in_ball = pdist(fake_x_pts, c_g) <= r_beta
    
    # find the K nearest neighbors of real_x in all other reals
    rx = make_numpy(real_x)
    dists, _ = nearest_neighbors.kneighbors(rx.reshape(1,-1), K)
    NND = dists.squeeze()[-1] # Kth nearest neighbor
    
    # find nearest pt to real_x in ball B(c_g, r_beta)
    D = pdist(fake_x_pts[in_ball], real_x)
    nearest_fake = torch.argmin(D)
    
    # is nearest fake in Ball(real_x, NND) ? 
    pred = pdist(fake_x_pts[nearest_fake].view(1,-1), real_x.view(1,-1)) <= NND
    return float(pred.detach())
    

def authenticity_classifier(fake_x, real_x_pts):
    """
    """
    # find distance between fake pt and all reals
    D = pdist(fake_x, real_x_pts)
    # find nearest real to fake
    i = torch.argmin(D) # i* in paper
    d_g_j = torch.min(D) # d_{g,j} in paper
    
    real_x_pts_wo = torch.cat([real_x_pts[:i],real_x_pts[i+1:]],dim=0)
    
    d_r_i = pdist(real_x_pts[i], real_x_pts_wo).min() # smallest distance to rest
    
    # output of authenticity classifier is A_j=1 (authentic) if the 
    # distance d_r_i < d_g_j
    pred = d_r_i < d_g_j
    return float(pred.detach())
        

    
def density_and_coverage(fake_x, real_x_pts, K=10):
    """From Naeem et al 2020.
        Density is defined as (1/kM)*(1/n) \sum_j \sum_i (Is fake_j  in ball B(x_i, NND_k(x_i))
    """
    nearest_neighbors = knn(real_x_pts, K)
    
    D = []
    C = np.zeros(len(real_x_pts))
    M = fake_x.shape[0]
    for i in range(real_x_pts.shape[1]):
        
        for j in range(M):
            
            # is Y_j in Ball around X_i ?
            # what is the Ball around X_i
            rx = make_numpy(real_x_pts[i])
            dists, _ = nearest_neighbors.kneighbors(rx.reshape(1,-1), n_neighbors=K,
                                                         return_distance=True)
            # find the Kth neighbor distance
            NND = dists.squeeze()[-1]
            
            pred = pdist(fake_x[j].view(1,-1), real_x_pts[i].view(1,-1)) <= NND
            D.append(float(pred.detach()))
            
            # for coverage
            if D[-1] and C[i] == 0.:
                C[i] = 1.
            
    D = np.sum(D)
    return (1/K)*(1/M)*D, C.mean()
            
            
    
def learn_embedding(data, opts):
    
    print('..... making tensor dataset and dataloader .....')
    traindataset = make_dataset(data, opts['max_length'])
    traindataloader = DataLoader(traindataset, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
    
    print('..... training embedding .....')
    embedder = EmbeddingNet(data[0].shape[0], opts['nembed'], nrnn=opts['nrnn'], nrnnlayers=opts['nrnnlayers'],
                       dropout=opts['dropout'], nlin=opts['nlin'])
    if opts['cuda']:
        embedder = embedder.cuda()

    return train_embedding(embedder, traindataloader, opts)



def learn_autoencoding(data, opts):
    
    print('..... making tensor dataset and dataloader .....')
    traindataset = make_dataset(data, opts['max_length'])
    traindataloader = DataLoader(traindataset, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
    
    print('..... training embedding .....')
    embedder = AutoencoderNet(data[0].shape[0], opts['nembed'], nrnn=opts['nrnn'], nrnnlayers=opts['nrnnlayers'],
                       dropout=opts['dropout'], nlin=opts['nlin'])
    if opts['cuda']:
        embedder = embedder.cuda()

    return train_autoencoder(embedder, traindataloader, opts)



def embed_list_of_sequences(embedder, seqs, cuda=True):
    
    Xtilde = []
    for x in seqs:
        
        x = torch.from_numpy(x).to(torch.float32)
        
        if cuda:
            x = x.cuda()
            
        z = embedder(x.view(1, x.shape[0], x.shape[1]))
        
        if isinstance(z, torch.Tensor):  
            Xtilde.append(z)
        else:
            # tuple output from autoencoder
            Xtilde.append(z[1])
            
    return torch.cat(Xtilde, dim=0)
    
    

def evaluate_generative_model(netG, hmm, data, opts, embedder = None):
    """Evaluates fake data generated by the trained model in relation to 
        real data ("data") that the model is trained for.
        
        For real data sample / fake data sample, a tuple of (precision, recall, authenticity)
        are obtained.  
    """
    # number of samples = number of real data sequences
    N = len(data)
    
    if opts['nsamples'] is None or opts['nsamples'] < N:
        nsamps = N
    else:
        nsamps = opts['nsamples']
        
    # generate fake sample sequences
    with torch.no_grad():
        
        print('..... generating fake samples .....')
        timesteps = [seq.shape[1]//16 for seq in data]
        sample_seqs = generate_samples(netG, hmm, nsamps, 1., timesteps, opts['cuda'])

        # embed all
        print('..... embedding fakes and real samples .....')
        fake_pts = embed_list_of_sequences(embedder, sample_seqs, cuda=opts['cuda'])
        real_pts = embed_list_of_sequences(embedder, data, cuda=opts['cuda'])
        
        # nearest neighbor
        print('..... learning nearest neighbors for all reals .....')
        nearest_neighbors = knn(real_pts, opts['kneighbors'])

        # radius of the real Ball 
        center_r = torch.zeros(opts['nembed'])
        if opts['cuda']:
            center_r = center_r.cuda()

        r_alpha = compute_quantile_radius(real_pts, center_r, opts['alpha'])
        print('..... alpha radius for real ball is %.4f .....'%(float(r_alpha)))

        # sample wise metrics
        # precision
        print('..... computing precision for all fake samples .....')
        P = np.array([precision_classifier(fake_pts[i], center_r, r_alpha) for i in range(N)]).mean()

        # recall
        # make center of fake Ball the average fake data
        c_g = fake_pts.mean(dim=0)
        # get ball B radius for ball of generated samples
        r_beta = compute_quantile_radius(fake_pts, c_g, opts['beta'])
        print('..... beta radius for fake ball is %.4f .....'%(r_beta))

        print('..... computing recall for all fake samples .....')
        R = np.array([recall_classifier(y, fake_pts, nearest_neighbors, r_beta, opts['kneighbors'],
                          beta=opts['beta']) for y in real_pts]).mean()

        # authenticity 
        print('..... computing authenticity for all fake samples .....')
        A = np.array([authenticity_classifier(y, real_pts) for y in fake_pts]).mean()

        # density D and coverage C (Naeem et al 2020)
        D, C = density_and_coverage(fake_pts, real_pts, opts['kneighbors'])

    return P, R, A, D, C



def train_autoencoder(model, traindataloader, opts):
    """Traning function for autencoding encoder-decoder network
        Params
        ------
            model : 
            dataloader : torch.utils.data.Dataloader map style
            opts : options dict
    """
    
    if opts['cuda']:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr = opts['lr'], weight_decay = opts['l2'], 
                                 betas=(opts['beta1'], 0.999))
    N = len(traindataloader)
    
    costfunc = nn.MSELoss()
    
    for n in range(opts['nepochs']):

        for i, x in enumerate(traindataloader):
            x = x[0]
            
            optimizer.zero_grad()
            model.zero_grad()
            
            # make sure the shape is (N,L,H)
            x = x.to(dev)
             
            # encode and decode
            xhat, z = model(x)
            
            # reduce MSE and L2 norm of z
            loss1 = costfunc(xhat, x)
            loss2 = torch.norm(z, p = 2)
            loss = loss1 + loss2
            loss.backward()
            
            optimizer.step()
            
            #train_loss.append()
            recon_loss = float(loss1.detach())
            norm_loss = float(loss2.detach())
            
            if i%opts['log_every'] == 0:
                print("..... epoch %d, minibatch [%d/%d], recon_loss = %.3f, norm_loss = %.3f ....."%(n,i, N,
                                                                                                      recon_loss,
                                                                                                      norm_loss))
            
    # model checkpoint
    if opts['checkpoint_models']:
        torch.save(model.state_dict(), '%s/autoencoder_day_%d.pth' % (opts['savepath'], opts['dayidx']))
        
    return model



    
def train_embedding(model, traindataloader, opts):
    """Traning function for embedding network
        Params
        ------
            model : 
            dataloader : torch.utils.data.Dataloader map style
            opts : options dict
    """
    if opts['cuda']:
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr = opts['lr'], weight_decay = opts['l2'], 
                                 betas=(opts['beta1'], 0.999))
    
    
    center_r = torch.zeros(opts['batch_size'], opts['nembed'])
    center_r.requires_grad = False
    center_r = center_r.to(dev)
    
    radius = torch.ones(1)
    radius.requires_grad = True
    radius = radius.to(dev)
    
    
    N = len(traindataloader)
    
    for n in range(opts['nepochs']):

        for i, x in enumerate(traindataloader):
            x = x[0]
            
            optimizer.zero_grad()
            model.zero_grad()
            radius.retain_grad()
            
            # make sure the shape is (N,L,H)
            x = x.to(dev)
             
            # embed
            xtilde = model(x)
            
            # compute error
            loss = one_class_SVM_loss(xtilde, center_r, radius, opts['nu'])
            loss.backward()
            
            optimizer.step()
            
            radius -= opts['lr']*radius.grad
            
            if radius < 0.:
                radius = radius + 1e-4
                
            # zero out the radius buffer
            radius.grad = torch.zeros_like(radius)
            
            #train_loss.append()
            fl = float(loss.detach())
            
            if i%opts['log_every'] == 0:
                print("..... epoch %d, minibatch [%d/%d], training loss = %.3f, radius = %.3f ....."%(n,i,N,fl, radius.detach()))
            
    # model checkpoint
    if opts['checkpoint_models']:
        torch.save(model.state_dict(), '%s/embedding_net_day_%d.pth' % (opts['savepath'], opts['dayidx']))
        
    return model



def generate_samples(netG, hmm, nsamples=1, invtemp=1., timesteps=[], cuda=True):
    """Generate samples from trained netG and hmm"""
    seqs = [tempered_sampling(hmm, invtemp, timesteps=timesteps[i], 
                                sample_obs=True, start_state_max=True, 
                                 sample_var = 0.)[0] for i in range(nsamples)]
    # decode with netG
    seqs_out = [None for _ in range(len(seqs))]
    for i in range(len(seqs)):
        seqs_out[i] = overlap_decode(seqs[i], netG,  noverlap = 0,
                                          cuda = cuda, get_audio = False)[0]
    
    return seqs_out



def pad_to_maxlength(x, max_length=100):
    """Pad a sequence to maximum length"""
    if x.shape[1] >= max_length:
        return x[:, :max_length]
    
    # else pad right
    x = np.concatenate([x, np.zeros((x.shape[0], max_length-x.shape[1]))],axis=1)
    return x



def make_dataset(sequences, max_length=100):
    X = []
    for seq in sequences:
        X.append(pad_to_maxlength(seq, max_length))
    X = np.stack(X)
    
    return TensorDataset(torch.from_numpy(X).to(torch.float32))
    


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--birdname', type=str, required=True)
parser.add_argument('--modelpath',type=str, required=True)
parser.add_argument('--datapath', type=str, required=True, help='path to training dataset')
parser.add_argument('--netGfilepath', type=str, default='netG_epoch_40_day_all.pth')
parser.add_argument('--daily_gan', action='store_true', help='whether this a daily gan model')
parser.add_argument('--learn_embedding', action='store_true', help='learns one class SVM type embedder')
parser.add_argument('--savepath', type=str, default='fidelity_results')
parser.add_argument('--nembed', type=int, default=3, help='embedding dimensions')
parser.add_argument('--nsamples', type=int, default=1, help='number of fake sequences to generate')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--checkpoint_models', action='store_true')
parser.add_argument('--log_every', type=int, default=500)
parser.add_argument('--nepochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--l2', type=float, default=0.0)
parser.add_argument('--nrnn', type=int, default=100)
parser.add_argument('--nrnnlayers', type=int, default=1)
parser.add_argument('--nlin', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--kneighbors', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.95)
parser.add_argument('--beta', type=float, default=0.95)
parser.add_argument('--nu', type=float, default=1.)






def main():
    
    args = parser.parse_args()
    
    opts = vars(args)
    
    # which days were trained for this bird and model type
    day_folders = sorted(glob(join(args.modelpath, 'day_*')))
    
    # make bird dataset
    dataset = bird_dataset_single_hdf(opts['datapath'], opts['birdname'])
    
    # make output folder
    if not os.path.exists(join(opts['modelpath'], opts['savepath'])):
        os.makedirs(join(opts['modelpath'], opts['savepath']))
        
    # model opts
    with open(join(args.modelpath, 'opts.json'), 'rb') as f:
        model_opts = json.load(f)
        
    # to store all results
    metrics_dict = dict(day_id=[], hmm_nstates=[], avg_precision=[], avg_recall=[], avg_authenticity=[], 
                       density=[], coverage=[])
    
    netG = None
    if not opts["daily_gan"]:
        netGfilepath = join(args.modelpath, args.netGfilepath)
        netG = load_netG(netGfilepath, model_opts['nz'], model_opts['ngf'], model_opts['nc'],
                            args.cuda, resnet=True)
    
        
    for day in day_folders:
        
        # find the day index from the name
        dayname = os.path.split(day)[-1]
        dayidx = int(dayname.split('_')[-1])
        opts['dayidx'] = dayidx
        
        print('\n\n..... working on day %s .....'%(dayname))           
        
        # get real data for this day 
        X = dataset.get(dayidx, nsamps=-1)
        print('..... total %d sequences on this day .....'%(len(X)))
        
        # first train embedding on real data
        if opts['learn_embedding']:
            print('## training embedding ##')
            embedder = learn_embedding(X, opts)
            embedder.eval()
            print('## finished training embedding ##')
        else:
            print('## training autoencoder ##')
            embedder = learn_autoencoding(X, opts)
            embedder.eval()
            print('## finished training autoencoder ##')
            
        # find which netG to use
        if netG is None:
            netGfilepath = join(day, args.netGfilepath) + '_day_' + str(dayidx) + '.pth'
            netG = load_netG(netGfilepath, model_opts['nz'], model_opts['ngf'], model_opts['nc'],
                            args.cuda, resnet=True)
        
        hmm_models = sorted(glob(join(day, 'hmm_*')))
        
        for k in range(len(hmm_models)):
            
            # load hmm model
            hmm = joblib.load(join(day, hmm_models[k], 'model_day_'+str(dayidx)+'.pkl'))
            hmm = hmm['model']
            nstates = hmm.transmat_.shape[0]
            print('\n ## model with nstates = %d ##'%(nstates))
            
            metrics = evaluate_generative_model(netG, hmm, X, opts, embedder)
            
            print(f"..... day {dayname}, avg precision {metrics[0]} .....")
            print(f"..... day {dayname}, avg recall {metrics[1]} .....")
            print(f"..... day {dayname}, avg authenticity {metrics[2]} .....")
            print(f"..... day {dayname}, density {metrics[3]} .....")
            print(f"..... day {dayname}, coverage {metrics[4]} .....")
            
            metrics_dict['day_id'].append(dayidx)
            metrics_dict['hmm_nstates'].append(nstates)
            metrics_dict['avg_precision'].append(metrics[0])
            metrics_dict['avg_recall'].append(metrics[1])
            metrics_dict['avg_authenticity'].append(metrics[2])
            metrics_dict['density'].append(metrics[3])
            metrics_dict['coverage'].append(metrics[4])
            
            
    # save metrics as csv file
    metrics_dict = pd.DataFrame(metrics_dict)
    metrics_dict.to_csv(join(opts['savepath'], 'fidelity_metrics.csv'), index=False)
            
            
            
            
if __name__ == '__main__':
    main()