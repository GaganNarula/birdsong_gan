import numpy as np
from hmmlearn.hmm import GaussianHMM
from .hmm_utils import tempered_sampling
import argparse
import joblib
from joblib import Parallel, delayed
from time import time
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
from os.path import join
from glob import glob



def get_samples(model, nsteps = 100, nsamps = 10000, sample_var=None, beta=1.):
    
    # variance of sampling
    if sample_var is None:
        sample_variance = 0
    else:
        sample_variance = sample_var
        
    samples = Parallel(n_jobs=-2)(delayed(tempered_sampling)(model, beta=beta,
                                                             timesteps=nsteps,
                                                             sample_obs=True,
                                                             start_state_max=True,
                                                            sample_var=sample_variance) for i in range(nsamps))
    samples = [s[0] for s in samples]
    return samples



def get_scores(model, samples):
    s = Parallel(n_jobs=-2)(delayed(model.score)(s) for s in samples)
    return np.mean(s)


def get_normalized_scores(model, seqs, normalize_by_length=True):
    LogL = 0.
    for n in range(len(lengths)):
        ll = model.score(seq[n]) 
        if normalize_by_length:
            ll /= lengths[n]
        LogL += ll
    return LogL / n


def get_divergence(P, Q, samples_P=None, samples_Q=None, nsteps=100,
                  beta=1., sample_var=None):
    """Compute the Montecarlo estimate of KL divergence between two
        hmms.
        
        Parameters
        ----------
        P : type hmmlearn.hmm.GaussianHMM, hmm model 1 
        
        Q : type hmmlearn.hmm.GaussianHMM, hmm model 2
        
        samples_P : list of sample sequences from P. If None (default), samples are 
                    generated from the model P.
        
        samples_Q : list of sample sequences from Q. If None (default), samples are 
                    generated from the model Q.
                    
        nsteps : int, number of timesteps in a sample sequence
        
        beta : float, inverse temperature of sampling 
        
        sample_var : float, if 0. or None, the model.covar_ parameter is used, else, 
                            if > 0., a spherical gaussian of this variance is used.
        
        Returns
        -------
        
        DKL_P_Q : float, D_KL(P|Q) = <logP - logQ>_Psamples (average over samples from P)
        
        DKL_Q_P : float, D_KL(Q|P) = <logQ - logP>_Qsamples. (average over samples from Q)
        
        DKL_symm: float, Symmetric Jeffrey's divergence: 0.5*(DKL(P|Q) + DKL(Q|P))     
        
    """
    if samples_P is None:
        samples_P = get_samples(P, nsteps, nsamps,
                                random_state, sample_var
                                beta)
        
    if samples_Q is None:
        samples_Q = get_samples(Q, nsteps, nsamps,
                                random_state, sample_var
                                beta)
        
    logP_P = get_normalized_scores(P, samples_P)
    logQ_P = get_normalized_scores(Q, samples_P)
    logQ_Q = get_normalized_scores(Q, samples_Q)
    logP_Q = get_normalized_scores(P, samples_Q)
    
    DKL_P_Q = logP_P - logQ_P
    DKL_Q_P = logQ_Q - logP_Q
    DKL_symm = 0.5*(DKL_P_Q + DKL_Q_P)
    
    return DKL_P_Q, DKL_Q_P, DKL_symm

    
    
def load_hmm_model(model_outer_path, hidden_state_size):
    modelpaths = glob(join(model_outer_path, 'hmm_hiddensize*'))
    modelpath = [path for path in modelpaths if str(hidden_state_size) in path][0]
    model = joblib.load(modelpath)
    return model['model']


def load_sequence_data(data_path):
    data_pkl_path = glob(join(data_path, 'data*'))[0]
    data = joblib.load(data_pkl_path)
    ztrain = data['ztrain']
    ztest = data['ztest']
    entropies = data['Entropies']
    return ztrain + ztest, entropies



def compute_divergence_curve(opts):
    
    # how many models are there?
    days = glob(join(opts['birdpath'], 'day_*'))
    ndays = len(days)
    
    if ndays == 0:
        return
    
    days = sorted(days)
    
    # which one is tutor model? Last one by default
    
    if opts['tutor_hmm_model'] is None:
        tutorday = days[-1]
        
    K = len(opts['hmm_state_sizes'])
    
    KLD_tut_pup = np.zeros((ndays, K))
    KLD_pup_tut = np.zeros((ndays, K))
    JFD = np.zeros((ndays, K))
    
    # extra, for analysis
    total_duration = np.zeros(ndays)
    # 
    model_entropy = np.zeros((ndays, K, 3))
        
    for k in range(K):
        
        # get the tutor model
        tutormodel = load_hmm_model(tutorday, k)
        
        # get several sample trajectories from tutormodel
        if opts['divergence_from_samples']:
            tutsamples = get_samples(tutmodel, opts['nsteps'],
                                     opts['nsamps'],
                                     sample_var=opts['sample_variance'],
                                     beta=opts['sample_invtemp'])
            
        else:
            tutsamples, _ = load_sequence_data(tutorday)

        # cycle over days
        for d in range(ndays):
            
            print(' ..... evaluating day %d , hmm state size %d .....'%(d, k))
            
            pupmodel = load_hmm_model(days[d], k)
            
            if opts['divergence_from_samples']:
                pupsamples = get_samples(tutmodel, opts['nsteps'],
                                     opts['nsamps'],
                                     sample_var=opts['sample_variance'],
                                     beta=opts['sample_invtemp'])
                
                entropies = full_entropy(pupmodel)
                
            else:
                pupsamples, entropies = load_sequence_data(days[d])
            
            # record singing duration (in counts of frames)
            total_duration[d] = np.sum([x.shape[0] for x in pupsamples])
            
            # record entropies
            model_entropy[d,k,:] = entropies
            
            KLD_tut_pup[d,k], KLD_pup_tut[d,k], JFD[d,k] = get_divergence(P, Q,
                                                                          samples_P=tutsamples,
                                                                          samples_Q=pupsamples,
                                                                          nsteps=opts['nsteps'],
                                                                          beta=opts['sample_invtemp'],
                                                                          sample_var=opts['sample_variance'])
                   
            
            print('...... KLD_tut_pup: %.4f  , KLD_pup_tut: %.4f, JSD: %.4f'%(KLD_tut_pup[d,k],
                                                                              KLD_pup_tut[d,k],
                                                                              JFD[d,k]))
            
    return KLD_tut_pup, KLD_pup_tut, JFD, model_entropies, total_duration




parser = argparse.ArgumentParser()
parser.add_argument('--birdpath',type=str,required=True)
parser.add_argument('--savepath', type=str, required=True)
parser.add_argument('--divergence_from_samples', action='store_true',help='compute divergence from sampling from the model')
parser.add_argument('--nsteps', type = int, default = 100)
parser.add_argument('--nsamps', type = int, default = 10000)
parser.add_argument('--sample_variance', typ=int, default=0)
parser.add_argument('--sample_invtemp', type=float, default=1.)
parser.add_argument('--hidden_state_size', type = int, nargs = '+', default = [5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 100])


def main():
    
    args = parser.parse_args()
    args = vars(args)
    
    KLD_tut_pup, KLD_pup_tut, JFD, model_entropies, total_duration = compute_divergence_curve(args)
    
    joblib.dump({'KLD_tut_pup':KLD_tut_pup, 'KLD_pup_tut':KLD_pup_tut,
                'JFD':JFD, 'model_entropies': model_entropies, 'total_duration': total_duration},
                
                args['savepath'] + 'KLD_JSD_divergences.pkl')

    
    
if __name__ == '__main__':
    main()
    

        