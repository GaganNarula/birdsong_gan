import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmm_utils import hmm_num_free_params, number_of_active_states_viterbi, tempered_sampling
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
    return ztrain, ztest



def get_unnormalized_scores(model, seqs):
    """Normalized loglikelihood score"""
    LogL = [model.score(seq) for seq in seqs]
    return sum(LogL)



def number_of_active_states_sampling(model, steps):
    # sample several sequences and extract states
    seqs = [tempered_sampling(model, beta = 1., timesteps=T, 
                                 sample_obs=True, start_state_max=True, 
                                 sample_var = 0) for T in steps]
    seqs = [s[1] for s in seqs]
    # for each sample find the number of unique states
    unqs = [np.unique(s) for s in seqs]
    nunqs = np.array([len(u) for u in unqs])
    # what is the mean number of active states
    mean_active = np.mean(nunqs)
    std_active = np.std(nunqs)
    return mean_active, std_active



def compute_AIC_BIC(model, seqs):
    # get num params
    num_params = hmm_num_free_params(model.n_components, model.n_features,
                                     covariance_type=model.covariance_type)
    # get normalized loglikelihood
    logL = get_unnormalized_scores(model, seqs)
    
    num_samps = sum([s.shape[0] for s in seqs])
    aic = 2*num_params - (2 * logL)
    bic = num_params*np.log(num_samps) - (2 * logL)
    return aic, bic
            


        
def compute_AIC_BIC_curve(opts):
    
    # how many models are there?
    days = glob(join(opts['birdpath'], 'day_*'))
    
    ndays = len(days)
    # sort them by number 
    day_ids = [int(d.split('_')[-1]) for d in days]
    day_ids = np.argsort(day_ids)
    days = [days[i] for i in day_ids]

    if ndays == 0:
        return
    
    # variables to record
    AIC_train = [None for _ in range(ndays)]
    AIC_test = [None for _ in range(ndays)]
    AIC_full = [None for _ in range(ndays)]
    BIC_train = [None for _ in range(ndays)]
    BIC_test = [None for _ in range(ndays)]
    BIC_full = [None for _ in range(ndays)]
    best_model_aic = [None for _ in range(ndays)]
    best_model_bic = [None for _ in range(ndays)]
    mean_active = [None for _ in range(ndays)]
    sd_active = [None for _ in range(ndays)]
    
    # cycle over days
    for d in range(ndays):
        
        # find the different hidden state sizes
        modelpaths = glob(join(days[d], 'hmm_hiddensize*'))
        
        hidden_state_sizes = [p.split('_')[-1] for p in modelpaths]
        hidden_state_sizes.sort()
        
        nstates = len(hidden_state_sizes)
        
        AIC_train[d] = {}
        AIC_test[d] = {}
        BIC_train[d] = {}
        BIC_test[d] = {}
        AIC_full[d] = {}
        BIC_full[d] = {}
        mean_active[d] = {}
        sd_active[d] = {}
        
        for k in hidden_state_sizes:
        
            print('\n..... evaluating day %d , hmm state size %s .....'%(d, k))
            
            # get the pupil model
            pupmodel = load_hmm_model(days[d], k)
                
            # get the encoded sequences
            ztrain, ztest = load_sequence_data(join(days[d], 'hmm_hiddensize_' + k))

            aic_train, bic_train = compute_AIC_BIC(pupmodel, ztrain)
            aic_test, bic_test = compute_AIC_BIC(pupmodel, ztest)
            aic_full, bic_full = compute_AIC_BIC(pupmodel, ztrain+ztest)
            
            z = ztrain+ztest
            lengths = [x.shape[0] for x in z]
            
            # num active unit 
            mu, sd = number_of_active_states_sampling(pupmodel, steps = lengths)
            #mu, sd = number_of_active_states_viterbi(pupmodel, z, lengths)
            
            AIC_train[d][k] = aic_train
            AIC_test[d][k] = aic_test
            AIC_full[d][k] = aic_full
            BIC_full[d][k] = bic_full
            BIC_train[d][k] = bic_train
            BIC_test[d][k] = bic_test
            
            mean_active[d][k] = mu
            sd_active[d][k] = sd
            
            print('..... AIC train = %.4f, AIC test = %.4f, AIC_full = %.4f .....'%(aic_train, aic_test, aic_full))
            print('..... BIC train = %.4f, BIC test = %.4f, BIC_full = %.4f .....'%(bic_train, bic_test, bic_full))
            print('..... mean (SD) active states = %.2f (%.2f) .....'%(mu,sd))
        
        best_model_aic[d] = np.argmin(list(AIC_full[d].values()))
        best_model_bic[d] = np.argmin(list(BIC_full[d].values()))
        
        print('\n #### AIC best hidden size for day %d = %s ####'%(d,hidden_state_sizes[best_model_aic[d]]))
        print(' #### BIC best hidden size for day %d = %s ####\n'%(d,hidden_state_sizes[best_model_bic[d]]))
        
    return AIC_train, AIC_test, AIC_full, BIC_train, BIC_test, BIC_full, best_model_aic, best_model_bic, mean_active, sd_active







parser = argparse.ArgumentParser()
parser.add_argument('--birdpath',type=str,required=True)
parser.add_argument('--savepath', type=str, required=True)



def main():
    
    args = parser.parse_args()
    args = vars(args)
    
    os.makedirs(args['savepath'], exist_ok=True)
    
    AIC_train, AIC_test, AIC_full, BIC_train, BIC_test, BIC_full, best_model_aic, \
        best_model_bic, mean_active, sd_active = compute_AIC_BIC_curve(args)
    
    joblib.dump({'AIC_train':AIC_train, 'AIC_test': AIC_test, 'AIC_full': AIC_full,
                 'BIC_train':BIC_train, 'BIC_test':BIC_test, 'BIC_full': BIC_full,
                 'best_model_aic':best_model_aic, 'best_model_bic':best_model_bic,
                 'mean_active':mean_active,
                 'sd_active':sd_active},
                join(args['savepath'], 'information_criteria.pkl'))

    
    
if __name__ == '__main__':
    main()
    

        