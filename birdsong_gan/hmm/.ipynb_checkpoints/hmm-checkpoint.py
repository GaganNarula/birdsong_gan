import sys
import os
from hmmlearn.hmm import GaussianHMM, GMMHMM
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import transform, inverse_transform, save_audio_sample, rescale_spectrogram, overlap_encode, overlap_decode
import argparse
import joblib
from joblib import Parallel, delayed
from time import time
from scipy.stats import entropy, multivariate_normal
import h5py
import pdb
import gc
import warnings
warnings.filterwarnings("ignore")



hmm_opts = {'hidden_state_size' : [5, 10, 15, 20, 30, 50, 75, 100],
           'covariance_type' : 'spherical', 
           'fit_params' : 'stmc',
           'transmat_prior' : 1.,
           'n_iter' : 300,
           'tolerance' : 0.01,
           'nz' : 16, 
           'ngf' : 264,
           'nc' : 1,
            'covars_prior' : 1.,
            'init_params' : 'stmc',
            'imageH': 129,
           'imageW': 16,
            'noverlap' : 0,
           'batchsize' : 2,
            'train_proportion' : 0.7,
           'nsamplesteps' : 128,
           'nsamps': 10,
           'sample_var': 0.,
            'sample_invtemperature' : 1.,
            'munge' : False,
            'munge_len' : 50,
            'n_restarts': 1,
            'do_chaining': False,
            'min_seq_multiplier': 10,
            'cuda' : True,
            'hmm_random_state' : 0,
            'last_day': -1,
            'start_from' : 0,
            'datapath' : '',
            'netEpath' : '',
            'netGpath' : '',
            'outpath' : '',
            'resnet' : False,
            'save_output' : True,
            'get_audio': False
           }



# random state
RNG = hmm_opts['hmm_random_state']




class HMM(object):
    def __init__(hmm_opts, self, datapath='', netEpath='', netGpath='', verbose = True,
                save_output = False):

        self.dataset = None
        if len(datapath)>0:
            self.dataset = bird_dataset(datapath)
        
        self.netE = None
        if len(netEpath)>0:
            self.netE = load_netE(netEpath, nz = hmm_opts['nz'], 
                              ngf = hmm_opts['ngf'], nc = hmm_opts['nc'], 
                              cuda = hmm_opts['cuda'], resnet = hmm_opts['resnet'])
        
        self.netG = None
        if len(netGpath)>0:
            self.netG = load_netG(netGpath, nz = hmm_opts['nz'], 
                              ngf = hmm_opts['ngf'], nc = hmm_opts['nc'], 
                              cuda = hmm_opts['cuda'], resnet = hmm_opts['resnet'])
        self.P = hmm_opts
        self.verbose = verbose
        self.save_output = save_output
        self.Z = [] # cached single day encoded latent vectors
        self.abs_min_files = 250 # there need to be at least so many files to fit 
        
    def encode_spectrograms(self, day_idx, hidden_size): 
        # get the latent space data
        X = self.dataset.get(day = day_idx)
        if len(X) < hidden_size*self.P['min_seq_multiplier'] or len(X) < self.abs_min_files:
            print('\n ### LESS THAN MIN FILES ON DAY %d, SKIPPING! ###\n'%(day_idx))
            return False
        # encode all spectrograms from that day
        self.Z = [overlap_encode(x, self.netE, transform_sample = False, imageW = self.P['imageW'], 
                                 noverlap = self.P['noverlap'], cuda = self.P['cuda']) for x in X]
        # munge sequences
        if self.P['munge']:
            self.Z = munge_sequences(self.Z, self.P['munge_len'])
        return True
    
    def create_training_and_test(self):
        # split into train and validation
        ids = np.random.permutation(len(self.Z))
        ntrain = int(self.P['train_proportion'] * len(ids))
        ztrain = [self.Z[ids[i]] for i in range(ntrain)]
        ztest = [self.Z[ids[i]] for i in range(ntrain, len(ids))]
        ids_train = ids[:ntrain]
        ids_test = ids[ntrain:]
        return ztrain, ztest, ids_train, ids_test
    
    def get_loglikelihood(self, model, data, lengths):
        return model.score(np.concatenate(data), lengths)
    
    def generate_spectrogram_samples(self, model):
        """Call this to generate spectrograms from the learned HMM
        """
        # create samples
        # if the variance of hmm is learned, by default use that to sample
        if 'c' in self.P['fit_params']:
            sample_variance = 0
        else:
            # if covariance was not learned, use given sample variance
            sample_variance = self.P['sample_var']
        seqs = [tempered_sampling(model, beta = self.P['sample_invtemperature'],
                                  timesteps=self.P['nsamplesteps'], 
                                 sample_obs=True, start_state_max=True, 
                                 sample_var = sample_variance)[0] for _ in range(self.P['nsamps'])]
        
        # create spectrogram
        spect_out = [None for _ in range(len(seqs))]
        for i in range(len(seqs)):
            spect_out[i] = overlap_decode(seqs[i], self.netG,  noverlap = self.P['noverlap'],
                                          cuda = self.P['cuda'], get_audio = self.P['get_audio'])
        spects = [s[0] for s in spect_out]
        if self.P['get_audio']:
            audio_out = [a[1] for a in spect_out]
        else:
            audio_out = None
        return spects, audio_out
        
    def fit_single_day(self, day_idx, hidden_size, lastmodel = None):
        
        # encode spectrograms
        if len(self.Z) == 0:
            encoded_z = self.encode_spectrograms(day_idx, hidden_size)
            if not encoded_z:
                return None
            
        # make train and test arrays and ids
        ztrain, ztest, ids_train, ids_test = self.create_training_and_test()
        
        # get lengths of sequences
        Ltrain = [z.shape[0] for z in ztrain]
        # train HMM 
        print('# learning hmm with %d states #'%(hidden_size))
        model = learnKmodels_getbest(ztrain, lastmodel, Ltrain, hidden_size, self.P)

        # compute 2 step full entropy
        print('# computing model entropy #')
        Hsp, Htrans, Hgauss = full_entropy(model)
        print('..... Transition entropy = %.2f, Emission entropy = %.2f .....'%(Htrans, Hgauss))
        
        # compute test log likelihood
        Ltest = [z.shape[0] for z in ztest]
        test_scores = self.get_loglikelihood(model, ztest, Ltest)
        # compute train log likelihood
        train_scores = self.get_loglikelihood(model, ztrain, Ltrain)

        # create 10 samples
        # concatenate the sequences because otherwise they are usually shorter than batch_size
        if self.save_output:
            outputfolder = os.path.join(self.P['outpath'], 'day_'+str(day_idx)+'_hiddensize_'+str(hidden_size))
            if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)
            # choose some ztrain for saving
            inds_to_save = np.random.choice(len(ztrain), size=self.P['nsamps'])
            ztosave = [ztrain[i] for i in inds_to_save]
            ztosave = np.concatenate(ztosave, axis=0)
            # save samples
            create_output(model, outputfolder, hidden_size, day_idx, self.P, self.netG, [])
            # save real files
            create_output(model, outputfolder, hidden_size, day_idx, self.P, self.netG, ztosave)
            print('# generated samples #')

        # get number of active states etc
        # how many active states were there ? 
        med_active, std_active = number_of_active_states_viterbi(model, np.concatenate(ztrain), Ltrain)
        
        if self.save_output:
            joblib.dump({'model':model, 'train_score':train_scores, 'test_score': test_scores, 'med_active':med_active,
                         'ztrain':ztrain,'ztest':ztest, 'std_active':std_active, 'ids_train':ids_train,
                         'ids_test':ids_test, 'Lengths_train':Ltrain,'Lengths_test':Ltest, 
                         'Entropies':[Hsp,Htrans,Hgauss], 'opts':hmm_opts}, 
                         os.path.join(outputfolder, 'model_data_and_scores_day_'+str(day_idx)+'.pkl'))
        
        return model, test_scores, train_scores, med_active, std_active, Ltrain, Ltest
    
    def fit_all_days(self):
        # list of hidden states to try
        K = self.P['hidden_state_size']
        # how many days are in the dataset
        Ndays = self.dataset.nfolders
        # estimate models up to last_day
        if self.P['last_day']== -1:
            last_day = Ndays
        else:
            last_day = self.P['last_day']
        # train models
        print('\n ..... training HMMs ..... \n')
        results = [None for _ in range(Ndays)]
        # loop over days
        for k in range(self.P['start_from'], last_day+1):
            # loop over hidden state sizes
            results[k] = [None for _ in range(len(K))]
            for j in range(len(K)):
                start = time()
                if not self.P['do_chaining']:
                    results[k][j] = self.fit_single_day(k, K[j], None)
                else:
                    
                    # chaining means using fitted params from the previous model to initialize 
                    if k > self.P['start_from'] and results[k-1][j] is not None:
                        results[k][j] = self.fit_single_day(k, K[j], results[k-1][j][0])
                    else:
                        results[k][j] = self.fit_single_day(k, K[j], None)
                
                end = time()
                
                if results[k][j] is not None and self.verbose:
                    print('..... day %d, hidden state size %d, train loglike: %.4f, test loglike: %.4f .....' %(k, K[j],
                                                                                                          results[k][j][2],
                                                                                                          results[k][j][1]))
                    print('..... avg number of active states %.2f, std dev %.2f .....'%(results[k][j][3], results[k][j][4]))
                    print('..... time taken %.1f secs ..... \n\n'%(end-start))
            self.Z =[] # empty this day's cache of latent vectors
            
        print('\n ###### finished training! #######')
        return results
    
    def close(self):
        self.netE = self.netE.cpu()
        self.netG = self.netG.cpu()
        del self.netE
        del self.netG
        del self.Z
        gc.collect()
        self.dataset.close()
        self.dataset = None
        
        
    
def learnKmodels_getbest(data, lastmodel, Lengths, hidden_size, hmm_opts):
    ''' EM based HMM learning with multiple initializations '''
    models = []
    LL = np.nan * np.zeros(hmm_opts['n_restarts'])
    for k in range(hmm_opts['n_restarts']): 
        
        try:
            m = learn_single_hmm_gauss_with_initialization(data, lastmodel, Lengths, hidden_size, 
                                                       hmm_opts['covariance_type'], 
                                                       hmm_opts['transmat_prior'], hmm_opts['n_iter'], 
                                                       hmm_opts['tolerance'], hmm_opts['fit_params'],
                                                       hmm_opts['covars_prior'], hmm_opts['init_params']) 

            # compute train log likelihood
            logl = m.score(np.concatenate(data,axis=0), Lengths)
        except Exception as e:
            print(e)
            pdb.set_trace()
            continue
            
        LL[k] = logl
        models.append(m)
    # choose model with highest LL
    if len(LL)==0 or np.all(np.isnan(LL)):
        return None
    
    best = np.nanargmax(LL)
    return models[best]



def initKmeans_means_and_covs(x, K):
    from sklearn.cluster import KMeans
    kms = KMeans(n_clusters = K)
    kms.fit(x)
    means = kms.cluster_centers_
    covs = np.zeros((K,x.shape[-1],x.shape[-1]))
    for k in range(K):
        covs[k] = np.cov(x[kms.labels_ == k], rowvar=False) + 1e-2*np.eye(x.shape[-1])
    return means, covs



def init_gmm_means_and_covs(x, K, covtype = 'diag'):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components = K, covariance_type = covtype, 
                         random_state = RNG, reg_covar = 1e-4)
    gmm.fit(x)
    return gmm.means_, gmm.covariances_



def learn_single_hmm_gauss_with_initialization(data, lastmodel = None, lengths = [], K=10, covtype='spherical',
                                               transmat_prior=1, n_iter=1000, tol = 0.01, fit_params = 'stmc', 
                                               covars_prior = 1. ,init_params = 'kmeans'):
    """ Learn a single model on the set of sequences in data
        If lastmodel is provided, it is used to initialize the parameters of this model. 
        Params
        ------
            data : list of numpy.ndarrays. Each array has shape (timesteps , dimensions)
            lastmodel : hmmlearn.hmm.GaussianHMM model. Learned from previous day.
            lengths : list, lengths of individual sequences in data
            K : int ,hidden state size
            covtype : str, {'spherical','diag','tied','full'} covariance matrix type
            transmat_prior : float, dirichlet concentration prior. Values > 1 lead to more uniform discrete probs
            n_iter : int, maximum number of EM iterations
            tol : float, tolerance for log-likelihood changes. If log-likelihood change < tol, learning is finished.
            fit_params : str, Any combination of 's' (start_prob), 't' (transition matrix), 'm': mean and 
                                'c' : covariance. The specified parameters will be estimated, others remain fixed.
            init_cov : float, initial covariance along diagonal
            covars_prior : float, covariance matrix prior. Same prior over all dimensions. The actual prior is a 
                            diagonal.
            init_params : str, 'stmc'
    """
    data = np.concatenate(data,axis=0)
    if lastmodel is None:
        if init_params == 'kmeans':
            model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter, covars_prior=covars_prior,params=fit_params, 
                        init_params = '', verbose=False, tol=tol, min_covar = 1e-2)
            #intialize randomly
            model.transmat_ = np.random.dirichlet(transmat_prior * np.ones(K), size = K)
            model.startprob_ = np.random.dirichlet(transmat_prior * np.ones(K))
            means_, covs_ = init_gmm_means_and_covs(data, K, covtype)
            fake_init_data = np.random.multivariate_normal(mean=np.zeros(data.shape[1]),
                                                           cov=np.eye(data.shape[1]), 
                                                           size = data.shape[0])
            model._init(fake_init_data)
            model.means_ = means_
            model.covars_ = covs_
        else:
            model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter, covars_prior=covars_prior,params=fit_params, 
                        init_params = init_params, verbose=False, tol=tol)
        # for hmmlearn vesion 0.2.3
        #fake_init_data = np.random.multivariate_normal(mean=np.zeros(data.shape[1]),
        #                                               cov=covars_prior*np.eye(data.shape[1]), 
        #                                               size = data.shape[0])
        #model._init(fake_init_data)
        model.fit(data, lengths)
        return model
    
    model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter,  covars_prior=covars_prior*np.ones(K), params=fit_params, 
                        init_params = 'c', verbose=False, tol=tol)
    # initiliaze parameters to last model
    model.transmat_ = lastmodel.transmat_
    model.startprob_ = lastmodel.startprob_
    model.means_ = lastmodel.means_
    model.fit(data, lengths)
    return model
    


def learn_single_hmm_gaussmix_with_initialization(data, lastmodel = None, lengths = [], K=10, covtype='spherical',
                                                  nmix = 10, transmat_prior=1, n_iter=1000, tol = 0.01,
                                                  fit_params = 'stmc', covars_prior = 1. ,init_params = 'kmeans'):
    """ Learn a single model on the set of sequences in data
        If lastmodel is provided, it is used to initialize the parameters of this model. 
        Params
        ------
            data : list of numpy.ndarrays. Each array has shape (timesteps , dimensions)
            lastmodel : hmmlearn.hmm.GaussianHMM model. Learned from previous day.
            lengths : list, lengths of individual sequences in data
            K : int ,hidden state size
            nmix: int, how many components in the gaussian mixture at each step
            covtype : str, {'spherical','diag','tied','full'} covariance matrix type
            transmat_prior : float, dirichlet concentration prior. Values > 1 lead to more uniform discrete probs
            n_iter : int, maximum number of EM iterations
            tol : float, tolerance for log-likelihood changes. If log-likelihood change < tol, learning is finished.
            fit_params : str, Any combination of 's' (start_prob), 't' (transition matrix), 'm': mean and 
                                'c' : covariance. The specified parameters will be estimated, others remain fixed.
            init_cov : float, initial covariance along diagonal
            covars_prior : float, covariance matrix prior. Same prior over all dimensions. The actual prior is a 
                            diagonal.
            init_params : str, 'stmc'
    """
    data = np.concatenate(data,axis=0)
    if lastmodel is None:
        if init_params == 'kmeans':
            model = GMMHMM(n_components=K, nmix=nmix, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter, covars_prior=covars_prior,params=fit_params, 
                        init_params = '', verbose=False, tol=tol, min_covar = 1e-2)
            #intialize randomly
            model.transmat_ = np.random.dirichlet(transmat_prior * np.ones(K), size = K)
            model.startprob_ = np.random.dirichlet(transmat_prior * np.ones(K))
            means_, covs_ = init_gmm_means_and_covs(data, K, covtype)
            fake_init_data = np.random.multivariate_normal(mean=np.zeros(data.shape[1]),
                                                           cov=np.eye(data.shape[1]), 
                                                           size = data.shape[0])
            model._init(fake_init_data)
            model.means_ = means_
            model.covars_ = covs_
        else:
            model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter, covars_prior=covars_prior,params=fit_params, 
                        init_params = init_params, verbose=False, tol=tol)
        # for hmmlearn vesion 0.2.3
        #fake_init_data = np.random.multivariate_normal(mean=np.zeros(data.shape[1]),
        #                                               cov=covars_prior*np.eye(data.shape[1]), 
        #                                               size = data.shape[0])
        #model._init(fake_init_data)
        model.fit(data, lengths)
        return model
    
    model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter,  covars_prior=covars_prior*np.ones(K), params=fit_params, 
                        init_params = 'c', verbose=False, tol=tol)
    # initiliaze parameters to last model
    model.transmat_ = lastmodel.transmat_
    model.startprob_ = lastmodel.startprob_
    model.means_ = lastmodel.means_
    model.fit(data, lengths)
    return model



parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type = str)
parser.add_argument('--netGpath', type = str, help = 'path to generator neural network')
parser.add_argument('--netEpath', type = str, help = 'path to encoder neural network')
parser.add_argument('--outpath', type = str, help = 'where to save models and samples')
parser.add_argument('--do_chaining', action = 'store_true')
parser.add_argument('--save_output', action = 'store_true', help ='boolean, turns ON output saving')
parser.add_argument('--hidden_state_size', type = int, nargs = '+', default = [5, 10, 15, 20, 30, 50, 75, 100])
parser.add_argument('--nz', type = int, default = 16, help = 'latent space dimensions')
parser.add_argument('--ngf', type = int, default = 264, help = 'number of generator filters')
parser.add_argument('--covariance_type', type = str, default = 'spherical')
parser.add_argument('--covars_prior', type = float, default = 1., help ='diagnoal term weight on the prior covariance')
parser.add_argument('--fit_params', type = str, default = 'stmc', help = 'which parameters to fit, s = startprob, t = transmat, m = means, c = covariances')
parser.add_argument('--transmat_prior', type = float, default = 1., help = 'transition matrix prior concentration')
parser.add_argument('--n_iter', type = int, default = 400, help = 'number of EM iterations')
parser.add_argument('--tolerance', type = float, default = 0.01)
parser.add_argument('--get_audio', action = 'store_true', help = 'generate audio files as well')
parser.add_argument('--start_from', type = int, default = 0, help = 'start day of learning') 
parser.add_argument('--last_day', type = int, default = -1, help = 'last day of learning')
parser.add_argument('--min_seq_multiplier', type = int ,default = 10, help='the number of files should be at least hidden size x this factor')
parser.add_argument('--init_params', type = str, default = 'str', help='which variables to initialize, or initialize with kmeans, enter kmeans')
parser.add_argument('--munge', action = 'store_true', help='whether to concate')
parser.add_argument('--munge_len', type = int, default = 50, help = 'minimum length of a sequence to which sequences be concatenated')
parser.add_argument('--resnet', action = 'store_true', help = 'whether a resnet is being loaded or not')
parser.add_argument('--cuda', action = 'store_true', help='whether to run on gpu')


if __name__ == '__main__':
    args = parser.parse_args()
    op  = vars(args)
    for k,v in op.items():
        if k in hmm_opts.keys():
            hmm_opts[k] = v
            
    hmm_ = HMM(hmm_opts, args.datapath, args.netEpath, args.netGpath, verbose = True, save_output=args.save_output)
    results = hmm_.fit_all_days()
