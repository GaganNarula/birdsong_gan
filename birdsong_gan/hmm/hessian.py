import numpy as np
from time import time
from scipy.stats import multivariate_normal as mvn
from scipy.stats import entropy
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as Tmvn
import joblib
import pdb
import hmmlearn
from hmmlearn.hmm import GaussianHMM
from glob import glob
import argparse


def gauss_entropy(model,k):
    # get kth covariance
    cov = np.squeeze(model.covars_[k])
    D = cov.shape[-1]
    H = (D/2)*np.log(2*np.pi*np.exp(1.))
    H += 0.5*np.log(np.linalg.det(cov))
    return H


def prepare_params(params):
    mu = torch.from_numpy(params['means']).float()
    mu.requires_grad = True

    cov = torch.from_numpy(params['covars']).float()
    cov.requires_grad = True

    A = torch.from_numpy(params['transmat']).float()
    A.requires_grad = True

    pi = torch.from_numpy(params['start_prob']).float()
    pi.requires_grad = True
    return mu, cov, A, pi


def prepare_params_no_redundant(params):
    mu = torch.from_numpy(params['means']).float()
    mu.requires_grad = True

    cov = torch.from_numpy(params['covars']).float()
    cov.requires_grad = True
    
    PI, T = regularize_pi_A(params['start_prob'],params['transmat'])
    B = 1 - T[:,:-1].sum(axis=1)
    B = torch.from_numpy(B).float()
    B.requires_grad = False
    
    A = torch.from_numpy(T[:,:-1]).float()
    A.requires_grad = True
    
    lastpi = 1 - PI[:-1].sum()
    pi = torch.from_numpy(PI[:-1]).float()
    pi.requires_grad = True
    return mu, cov, A, pi, B, lastpi



############## LOG LIKELIHOOD ##############



def logLLcalculation_scaled2(x, mu, varss, A, pi, T = 1):
    K = mu.shape[0] # num states
    D = mu.shape[1] # num dims
    alpha = []
    C = []
    for k in range(K):
        P = Tmvn(mu[k],varss[k]*torch.eye(D)).log_prob(x[0]).exp()
        alpha.append((pi[k] * P).view(1))
    alpha = torch.cat(alpha)
    c = torch.sum(alpha)
    alphahat = alpha / c
    C.append(c)
    # step 2 to T (python step 1 to T-1)
    for t in range(1,T):
        arr = []
        for k in range(K):
            term2 = torch.dot(A[:,k], alphahat)
            P = Tmvn(mu[k],varss[k]*torch.eye(D)).log_prob(x[t]).exp()

            if torch.isinf(term2):
                pdb.set_trace()
                
            arr.append((P * term2).view(1))
        alpha = torch.cat(arr)
        c = torch.sum(alpha)
        alphahat = alpha / c
        C.append(c)
    C = torch.stack(C)
    return C.log().sum() # log likelihood



def logLLcalculation_scaled_noredudantA(x, mu, varss, A, pi, B, T = 1):
    ''' A is not of full shape = [1:n, 1:n-1] and pi is not of full shape '''
    K = mu.shape[0] # num states
    D = mu.shape[1] # num dims
    
    lastpi = 1 - pi.detach().sum()
    alpha = []
    C = []
    for k in range(K):
        P = Tmvn(mu[k],varss[k]*torch.eye(D)).log_prob(x[0]).exp()
        if k < K-1:
            alpha.append((pi[k] * P).view(1))
        else:
            alpha.append((lastpi * P).view(1))
    alpha = torch.cat(alpha)
    c = torch.sum(alpha)
    alphahat = alpha / c
    C.append(c)
    
    # step 2 to T (python step 1 to T-1)
    for t in range(1,T):
        arr = []
        for k in range(K):
            if k < K-1:
                term2 = torch.dot(A[:,k], alphahat)
            else:
                term2 = torch.dot(B, alphahat)
            P = Tmvn(mu[k],varss[k]*torch.eye(D)).log_prob(x[t]).exp()

            if torch.isinf(term2):
                pdb.set_trace()
                
            arr.append((P * term2).view(1))
        alpha = torch.cat(arr)
        c = torch.sum(alpha)
        alphahat = alpha / c
        C.append(c)
    C = torch.stack(C)
    return C.log().sum() # log likelihood



def regularize_pi_A(pi, A, dt = 0.0001):
    # assuming pi and A are numpy arrays
    zero_inds = np.where(pi < dt)[0]
    mass_to_take = len(zero_inds)*dt
    nonzero_prop_inds = np.where(pi >= dt)[0]
    nonzero_props = pi[nonzero_prop_inds] / np.sum(pi[nonzero_prop_inds])
    pi[zero_inds] = dt
    pi[nonzero_prop_inds] = pi[nonzero_prop_inds] - nonzero_props*mass_to_take
    
    # for A
    for k in range(A.shape[0]):
        zero_inds = np.where(A[k,:] < dt)[0]
        mass_to_take = len(zero_inds)*dt
        nonzero_prop_inds = np.where(A[k,:] >= dt)[0]
        nonzero_props = A[k,nonzero_prop_inds] / np.sum(A[k,nonzero_prop_inds])
        A[k,zero_inds] = dt
        A[k,nonzero_prop_inds] = A[k,nonzero_prop_inds] - nonzero_props*mass_to_take
    return pi, A






def create_hess_mat_sphericalvar_symm_noredundant(H, ndim = 16, nstates = 5):
    # now the assumption is the covariance matrices are spherical
    totparams = ndim*nstates + nstates + (nstates*(nstates-1)) + (nstates-1)
    Hess = np.zeros((totparams,totparams))
    row = 0
    row_start = 0
    col1 = 0
    
    ##### NOW COV #####
    # covs vs means
    Hcov = H[1]
    for k in range(nstates):
        x = Hcov[0][k].flatten() # vectorized matrix
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
    col1 = col2*1
    row = row_start*1
    
    # covs vs covs
    for k in range(nstates):
        x = Hcov[1][k] # already a vector now 
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
    col1 = col2*1
    row = row_start*1
    
    # covs vs A
    for k in range(nstates):
        x = Hcov[2][k].flatten() # vectorized matrix
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
    col1 = col2*1
    row = row_start*1
    
    # covs vs pi
    for k in range(nstates):
        x = Hcov[3][k].flatten() # vectorized matrix
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1

    col1 = 0
    row = row_start*1
    row += nstates
    row_start = row*1
    
    # first do means
    
    Hmu = H[0]
    # means vs means
    for k in range(nstates):
        for j in range(ndim):
            x = Hmu[0][k,j].flatten() # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    col1 = col2*1
    row = row_start*1
    
    # means vs covs
    for k in range(nstates):
        for j in range(ndim):
            x = Hmu[1][k,j] # now just a vector, no need to flatten
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    col1 = col2*1
    row = row_start*1
    
    # means vs A
    for k in range(nstates):
        for j in range(ndim):
            x = Hmu[2][k,j].flatten() # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    col1 = col2*1
    row = row_start*1
    
    # means vs pi
    for k in range(nstates):
        for j in range(ndim):
            x = Hmu[3][k,j].flatten() # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    row = row_start*1
    row += ndim*nstates
    row_start = row*1
    col1 = 0
    
    
    #### NOW DO A ####
    HA = H[2]
    # A vs means
    for k in range(nstates):
        for j in range(nstates-1):
            x = HA[0][k,j].flatten() # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    col1 = col2*1
    row = row_start*1
    
    # A vs covs
    for k in range(nstates):
        for j in range(nstates-1):
            x = HA[1][k,j] # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    col1 = col2*1
    row = row_start*1
    
    # A vs A
    for k in range(nstates):
        for j in range(nstates-1):
            x = HA[2][k,j].flatten() # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    col1 = col2*1
    row = row_start*1
    
    # A vs pi
    for k in range(nstates):
        for j in range(nstates-1):
            x = HA[3][k,j].flatten() # vectorized matrix
            col2 = col1 + len(x)
            Hess[row, col1:col2] = x.detach().numpy()
            row += 1
    
    col1 = 0
    row = row_start*1
    row += nstates*(nstates-1)
    row_start = row*1
    
    #### NOW DO pi ####
    Hpi = H[3]
    # pi vs means
    for k in range(nstates-1):
        x = Hpi[0][k].flatten() # vectorized matrix
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
    col1 = col2*1
    row = row_start*1
    
    # pi vs covs
    for k in range(nstates-1):
        x = Hpi[1][k].flatten() # vectorized matrix
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
    col1 = col2*1
    row = row_start*1
    
    # pi vs A
    for k in range(nstates-1):
        x = Hpi[2][k].flatten() # vectorized matrix
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
    col1 = col2*1
    row = row_start*1
    
    # pi vs pi
    for k in range(nstates-1):
        x = Hpi[3][k] # already vector
        col2 = col1 + len(x)
        Hess[row, col1:col2] = x.detach().numpy()
        row += 1
        
    # take only upper triangle
    HH = np.zeros_like(Hess)
    HH = np.triu(Hess)
    Hess = HH + np.triu(Hess,k=1).T
    return Hess



def compute_hessian(mu, varss, A, pi, B, data):
    
    data = torch.from_numpy(data).float()
    data.requires_grad = False
    H = torch.autograd.functional.hessian(lambda x1,x2,x3,x4: logLLcalculation_scaled_noredudantA(data,x1,x2,x3,x4,B,T=data.shape[0]), 
                                      (mu,varss,A,pi), create_graph=False, strict=True)
    
    return create_hess_mat_sphericalvar_symm_noredundant(H, data.shape[-1], mu.shape[0]) 



def full_entropy(model):
    ''' calculate full entropy of a step 2 sequence '''
    # this means you need to add first the entropy of start
    # prob
    print('...... computing Entropy ......')
    Hsp = entropy(model.startprob_, base = 2)
    # for transition matrix, need a nested for loop
    # this is step 1
    Htrans = 0.
    for k in range(model.n_components):
        p_z1 = model.startprob_[k]
        Htr = 0.
        for j in range(model.n_components):
            if model.transmat_[k,j]>0.:
                Htr += model.transmat_[k,j] * np.log(model.transmat_[k,j]) 
            else:
                Htr += 0.
        Htrans += -1 * p_z1 * Htr
    # convert to bits
    Htrans *= np.log2(np.exp(1.))
    
    # spectral entropy
    # first get the 
    Hgauss = 0.
    for k in range(model.n_components):
        p_z1 = model.startprob_[k]
        Hg = 0.
        for j in range(model.n_components):
            Hg += model.transmat_[k,j] * gauss_entropy(model,j)
        Hgauss += p_z1 * Hg # minus sign is absorbed into gauss_entropy function
    return  Hsp, Htrans, Hgauss




parser = argparse.ArgumentParser()
parser.add_argument('--birddatapath', type=str, required=True, help='path to bird hmms')
parser.add_argument('--birdname', type = str, required=True)
parser.add_argument('--cuda', action = 'store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    # find all folders with the word day in the name
    all_folders = glob(args.birddatapath + '/*day*')
    
    
    results = {}
    for folder in all_folders:
        # load 
        flpath = glob(folder + '/' + '*model_data*')[0]
        # get day name
        fname = folder.split('/')[-1]
        print('\n ...... processing folder %s ......'%(fname))
        day = fname.split('_')[1]
         
        nstates = fname.split('_')[-1]
        
        D = joblib.load(flpath)
        
        model = D['model']
        zt = D['ztest']
        LL = D['test_score']
        
        
        params = {}
        params['means'] = model.means_
        params['covars'] = np.squeeze(model.covars_)
        params['transmat'] = model.transmat_
        params['start_prob'] = model.startprob_
    
        mu, cov, A, pi, B, _ = prepare_params_no_redundant(params)
        varss = cov[:,0,0]
        
        if args.cuda:
            mu = mu.cuda()
            varss = varss.cuda()
            A  = A.cuda()
            pi = pi.cuda()
            B = B.cuda()
            
        ndim = mu.shape[-1]
        nstates = mu.shape[0]
        totparams = ndim*nstates + nstates + (nstates*(nstates-1)) + (nstates-1)
        
        # compute entropy for this model
        Hsp, Htrans, Hgauss = full_entropy(model)
        Htot = Hsp+Htrans+Hgauss
        # ztest is a list
        print('...... Entropy = %.3f......'%(Htot))
        # get hessian
        print('...... computing Hessian over %d sequences ......'%(len(zt)))
        Htotal = np.zeros((totparams, totparams))
        for i, z in enumerate(zt):
            H = compute_hessian(mu, varss, A, pi, B, z)
            if np.sum(np.isnan(H))>0.:
                continue
            Htotal += H 
            if i%50 == 0:
                print('# done with %d / %d seqs # '%(i,len(zt)))
                
        
        Hdet = np.linalg.det(Htotal)
        if Hdet == 0.:
            pdb.set_trace()
            
        BIC = -2*LL + np.log(Hdet)
        
        print('...... BIC = %.5f......'%(BIC))
        results['day_'+day+'_nstates_'+str(nstates)] = [BIC, Htot, Hsp, Htrans, Hgauss]
    
    joblib.dump(results, BASEPATH + args.birdname + '/bic_entropy.pkl')