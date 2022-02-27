import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmm_utils import tempered_sampling, full_entropy_1step
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



def get_samples(model,
                nsteps: int=100,
                nsamps: int=10000,
                sample_var: float=None,
                beta: float=1.):
    
    # variance of sampling
    if sample_var is None:
        sample_variance = 0
    else:
        sample_variance = sample_var
        
    samples = Parallel(n_jobs=-2)(delayed(tempered_sampling)(model,
                                                             beta=beta,
                                                             timesteps=nsteps,
                                                             sample_obs=True,
                                                             start_state_max=True,
                                                            sample_var=sample_variance)[0] \
                                  for i in range(nsamps))

    return samples



def get_scores(model, samples):
    s = Parallel(n_jobs=-2)(delayed(model.score)(s) for s in samples)
    return np.mean(s)


def get_normalized_scores(model, seqs, normalize_by_length=True):
    LogL = 0.
    for n in range(len(seqs)):
        
        ll = model.score(seqs[n]) 
        
        if normalize_by_length:
            ll /= seqs[n].shape[0]
            
        LogL += ll
    return LogL / len(seqs)


def get_pointwise_metrics(modelP, modelQ, samplesP, samplesQ):
    """Computes sample-wise or 'pointwise' KL terms
    """
    n_p = len(samplesP)
    n_q = len(samplesQ)
    
    PlogP = np.zeros(n_p)
    PlogQ = np.zeros(n_p)
    QlogP = np.zeros(n_q)
    QlogQ = np.zeros(n_q)
    
    KL_P_Q = np.zeros(n_p)
    KL_Q_P = np.zeros(n_q)
    KL_P_M = np.zeros(n_p)
    KL_Q_M = np.zeros(n_q)
    
    for n in range(n_p):

        l = samplesP[n].shape[0]
        PlogP[n] = modelP.score(samplesP[n]) / l
        P_x_from_P = np.exp(PlogP[n])
        
        PlogQ[n] = modelQ.score(samplesP[n]) / l
        Q_x_from_P = np.exp(PlogQ[n])
        
        KL_P_Q[n] = PlogP[n] - PlogQ[n]
        
        KL_P_M[n] = PlogP[n] - np.log((P_x_from_P + Q_x_from_P)/2)
    
    for n in range(n_q):
        
        l = samplesQ[n].shape[0]
        QlogP[n] = modelP.score(samplesQ[n]) / l
        P_x_from_Q = np.exp(QlogP[n])
        
        QlogQ[n] = modelQ.score(samplesQ[n]) / l
        Q_x_from_Q = np.exp(QlogQ[n])
        
        KL_Q_P[n] = QlogQ[n] - QlogP[n]
        KL_Q_M[n] = QlogQ[n] - np.log((P_x_from_Q + Q_x_from_Q)/2)
        
    return KL_P_Q, KL_Q_P, KL_P_M, KL_Q_M, PlogP, QlogQ, PlogQ, QlogP



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
        
        Jeffreys: float, Symmetric Jeffrey's divergence: 0.5*(DKL(P|Q) + DKL(Q|P))     
        
        logP_P : float, neg entropy of model P
        logQ_P : float, cross entropy with samples from model P
        logP_Q : float, cross entropy with samples from model Q
        logQ_Q : float, neg entropy of model Q
        
    """
    if samples_P is None:
        samples_P = get_samples(P, nsteps, nsamps,
                                random_state, sample_var,
                                beta)
        
    if samples_Q is None:
        samples_Q = get_samples(Q, nsteps, nsamps,
                                random_state, sample_var,
                                beta)
        
    KL_P_Q, KL_Q_P, KL_P_M, KL_Q_M, PlogP, QlogQ, PlogQ, QlogP = get_pointwise_metrics(P, Q, samples_P, samples_Q)
    
    KL_P_Q = KL_P_Q.mean()
    KL_Q_P = KL_Q_P.mean()
    Jeffreys = 0.5*(KL_P_Q + KL_Q_P)
    JensonShannon = 0.5*(KL_P_M.mean() + KL_Q_M.mean())
    
    return KL_P_Q, KL_Q_P, Jeffreys, JensonShannon, PlogP.mean(), PlogQ.mean(), QlogQ.mean(), QlogP.mean()

    
    
def load_hmm_model(model_outer_path: str, hidden_state_size: str):
    modelpath = join(model_outer_path, 'hmm_hiddensize_' + hidden_state_size)
    modelpath = glob(join(modelpath, 'model_*'))[0]
    model = joblib.load(modelpath)
    return model['model']


def load_sequence_data(data_path: str):
    data_pkl_path = glob(join(data_path, 'data_and_scores*'))[0]
    data = joblib.load(data_pkl_path)
    ztrain = data['ztrain']
    ztest = data['ztest']
    return ztrain + ztest



def compute_divergence_curve(opts):
    
    # how many models are there?
    days = glob(join(opts['birdpath'], 'day_*'))
    
    ndays = len(days)
    # sort them by number 
    day_ids = [int(d.split('_')[-1]) for d in days]
    day_ids = np.argsort(day_ids)
    days = [days[i] for i in day_ids]

    if ndays == 0:
        return
    
    # which one is tutor model? Last one by default
    
    if opts['tutor_hmm_model'] is None:
        tutorday = days[-1]
        
    
    # variables
    KLD_tut_pup = [None for _ in range(ndays)]
    KLD_pup_tut = [None for _ in range(ndays)]
    logLscores = [None for _ in range(ndays)]
    JFD = [None for _ in range(ndays)]
    JSD = [None for _ in range(ndays)]
    
    # extra, for analysis
    total_duration = np.zeros(ndays)
    model_entropies = [None for _ in range(ndays)]
        
    # cycle over days
    for d in range(ndays):
        
        # find the different hidden state sizes
        modelpaths = glob(join(days[d], 'hmm_hiddensize*'))
        
        hidden_state_sizes = [p.split('_')[-1] for p in modelpaths]
        nstates = len(hidden_state_sizes)
        
        KLD_tut_pup[d] = {}
        KLD_pup_tut[d] = {}
        JFD[d] = {}
        JSD[d] = {}
        model_entropies[d] = {}
        logLscores[d] = {}
        
        
        for k in hidden_state_sizes:
        
            print('\n..... evaluating day %d , hmm state size %s .....'%(d, k))
            
            # get the tutor model
            tutormodel = load_hmm_model(tutorday, k)  
        
            # get the pupil model
            pupmodel = load_hmm_model(days[d], k)
            
            
            if not opts['divergence_from_samples']:
                pupsamples = load_sequence_data(join(days[d], 'hmm_hiddensize_' + k))
                
                tutsamples = load_sequence_data(join(tutorday,  'hmm_hiddensize_' + k))
                                                           
            else:
                # sample pupil and tutor models
                
                pupsamples = get_samples(tutormodel, opts['nsteps'],
                                         opts['nsamps'],
                                         sample_var=opts['sample_variance'],
                                         beta=opts['sample_invtemp'])
                
                tutsamples = get_samples(tutormodel, opts['nsteps'],
                                         opts['nsamps'],
                                         sample_var=opts['sample_variance'],
                                         beta=opts['sample_invtemp'])
                
            
            entropies = full_entropy_1step(pupmodel)
            # record entropies
            model_entropies[d][k] = entropies
            
            # record singing duration (in counts of frames)
            total_duration[d] = np.sum([x.shape[0] for x in pupsamples])
            
            KL_P_Q, KL_Q_P, jeffreys, jenson, PlogP, PlogQ, QlogQ, QlogP = get_divergence(tutormodel, pupmodel,
                                                      samples_P=tutsamples,
                                                      samples_Q=pupsamples,
                                                      nsteps=opts['nsteps'],
                                                      beta=opts['sample_invtemp'],
                                                      sample_var=opts['sample_variance'])
                   
            KLD_tut_pup[d][k] = KL_Q_P
            KLD_pup_tut[d][k] = KL_P_Q
            JFD[d][k] = jeffreys
            JSD[d][k] = jenson
            
            logLscores[d][k] = [PlogP, PlogQ, QlogQ, QlogP]
            
            print('..... KLD_tut_pup: %.4f  , KLD_pup_tut: %.4f, JFD: %.4f, JSD: %.4f, PlogQ: %.4f, QlogP: %.4f .....'%(KL_P_Q,
                                                                              KL_Q_P, jeffreys,
                                                                              jenson, PlogQ, QlogP))
            
            
    return KLD_tut_pup, KLD_pup_tut, JFD, JSD, logLscores, model_entropies, total_duration




parser = argparse.ArgumentParser()
parser.add_argument('--birdpath',type=str,required=True)
parser.add_argument('--savepath', type=str, required=True)
parser.add_argument('--divergence_from_samples', action='store_true', help='compute divergence from sampling from the model')
parser.add_argument('--nsteps', type=int, default = 100, help='number of timesteps per sample sequence')
parser.add_argument('--nsamps', type=int, default = 10000, help='number of sequences to sample')
parser.add_argument('--sample_variance', type=int, default=0)
parser.add_argument('--sample_invtemp', type=float, default=1.)


def main():
    
    args = parser.parse_args()
    args = vars(args)
    
    os.makedirs(args['savepath'], exist_ok=True)
    
    # FIX 
    args['tutor_hmm_model'] = None
    
    KLD_tut_pup, KLD_pup_tut, JFD, JSD, logLscores, model_entropies, total_duration = compute_divergence_curve(args)
    
    joblib.dump({'KLD_tut_pup':KLD_tut_pup,
                 'KLD_pup_tut':KLD_pup_tut,
                 'JFD':JFD,
                 'JSD': JSD,
                 'logLscores': logLscores,
                 'model_entropies': model_entropies,
                 'total_duration': total_duration},
                
                join(args['savepath'], 'KLD_JSD_divergences.pkl'))

    
    
if __name__ == '__main__':
    main()
    

        