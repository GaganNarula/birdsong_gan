import sys
sys.path.append(r'/home/gagan/code/birdsong_gan/birdsong_gan')
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
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
import os


def munge_sequences(seqs, minlen = 50):
    """
        To make sure that the statistics for covariance 
        calculation are robust, elongate each sequence.
    """
    seqs_out = []
    x = []
    for i in range(len(seqs)):
        
        tofill = int(minlen - np.sum([len(y) for y in x])) # how much time span left
        if tofill <= 0:
            # len will now be longer
            x.append(seqs[i])
            seqs_out.append(np.concatenate(x))
            x = []
            continue
            
        dur = seqs[i].shape[0] # assuming time is on first axis
        # if dur is less than what needs to be filled,
        if dur <= tofill:
            x.append(seqs[i])
        else:
            x.append(seqs[i][:tofill])
    
    return seqs_out


def tempered_sampling(model, beta=3., timesteps=64, sample_obs=True, start_state_max=False, sample_var = 0):
    # start probability sample
    if start_state_max:
        row = np.argmax(model.startprob_)
    else:
        # choose a start state
        p = np.exp(beta * np.log(model.startprob_))
        p /= np.sum(p)
        s0 = np.random.multinomial(1,p)
        row = np.where(s0==1.)[0][0]
    s0 = row
    states = np.zeros((timesteps),dtype='int64')
    obs = np.zeros((timesteps, model.means_.shape[-1]))
    for i in range(timesteps):
        # extract the correct row from the transition matrix
        a = model.transmat_[row,:]
        # make the gibbs probability vector
        p = np.exp(beta * np.log(a))
        p /= np.sum(p)
        # sample from it 
        s = np.random.multinomial(1,p)
        row = np.where(s==1.)[0][0]
        states[i] = row
        # sample from the corresponding distribution in the model
        mean_i = model.means_[row]
        sigma_i = model.covars_[row]
        if sample_obs:
            # sample an observation 
            if sample_var == 0:
                obs[i] = np.random.multivariate_normal(mean_i,np.squeeze(sigma_i),size=1)
            else:
                sigma_in = sample_var*np.eye(mean_i.shape[0])
                obs[i] = np.random.multivariate_normal(mean_i,sigma_in,size=1)
        else:
            obs[i] = mean_i
    return obs, states, s0



class Latent_loader(object):
    def __init__(self, merged_list, external_file_path = [], train_val_split=0.8, val_test_split=0.5):
        # split every element of merge_list into train, val and test
        self.id_lists_per_day = []
        for m in merged_list:
            id_list_train, id_list_valtest = split_list(m, train_val_split)
            id_list_val, id_list_test = split_list(id_list_valtest, val_test_split)
            self.id_lists_per_day.append([id_list_train, id_list_val, id_list_test])
        self.days = np.arange(len(merged_list))
        self.external_file_path = external_file_path
        
    def __getitem__(self, index):
        if self.external_file_path:
            birdname = os.path.basename(index['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(index['filepath'], 'r') 
        z = np.array(f.get(index['within_file']))
        f.close()
        return z
    
    def get_whole_day_sequences(self, day):
        id_list_train, id_list_val, id_list_test = self.id_lists_per_day[day]
        Ztrain = [None for i in range(len(id_list_train))]
        Zval = [None for i in range(len(id_list_val))]
        Ztest = [None for i in range(len(id_list_test))]
        for i in range(len(id_list_train)):
            Ztrain[i] = self.__getitem__(id_list_train[i])
        for i in range(len(id_list_test)):
            Ztest[i] = self.__getitem__(id_list_test[i])
        for i in range(len(id_list_val)):
            Zval[i] = self.__getitem__(id_list_val[i])
        return Ztrain, Ztest, Zval
    
    def get_N_random_training_samples(self, day, N=1):
        id_list_train, _ = self.id_lists_per_day[day]
        inds = np.random.choice(np.arange(len(id_list_train)), size=N, replace=False)
        id_list_out = [id_list_train[i] for i in inds]
        Zsample = [self.__getitem__(i) for i in id_list_out]
        return Zsample

    
class songbird_data_sample(object):
    def __init__(self, path2idlist, external_file_path):
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
        self.external_file_path = external_file_path
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def __getitem__(self, index):
        # load one wav file and get a sample chunk from it
        ID = self.id_list[index]
        age_weight = ID['age_weight']
        # this 'ID' is a dictionary containing several fields,
        # use field 'filepath' and 'within_file' to get data
        if self.external_file_path:
            birdname = os.path.basename(ID['filepath'])
            f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            f = h5py.File(ID['filepath'], 'r') 
        x = np.array(f.get(ID['within_file']))
        f.close()
        return x, age_weight
    
    def get_contiguous_minibatch(self, start_idx, mbatchsize=64):
        ids = np.arange(start=start_idx, stop=start_idx+mbatchsize)
        X = [self.__getitem__(i)[0] for i in ids]
        return X

    
def average_entropy(T):
    E = []
    for i in range(T.shape[0]):
        tmp = entropy(T[i], base = 2)
        #tmp = 0.
        #for j in range(T.shape[1]):
            #if T[i,j] > 0.:
            #    tmp += -T[i,j]*np.log(T[i,j])
        E.append(tmp)
    E = np.array(E)
    return E.mean()


def gauss_entropy(model,k):
    # get kth covariance
    cov = model.covars_[k]
    D = cov.shape[-1]
    H = (D/2)*np.log(2*np.pi*np.exp(1.))
    H += 0.5*np.log(np.linalg.det(cov))
    return H


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



def load_multiple_models(birdpath, days):
    models = [None for _ in range(len(days))]
    for i,d in enumerate(days):
        dirpath = os.path.join(birdpath,'day_'+str(d))
        fls = glob.glob(dirpath+'/model*')[0]
        m = joblib.load(fls)
        models[i] = m['model']
    return models



def KLdiv_bw_2multGaussians(p, q):
    '''
    Here p and q are dictionaries
    '''
    mu_p = np.array(p[0])
    mu_q = np.array(q[0])
    var_p = np.array(p[1])
    var_q = np.array(q[1])
    
    var_q_inv = np.linalg.inv(var_q)
    det_var_p = np.linalg.det(var_p)
    det_var_q = np.linalg.det(var_q)
    varprod = np.matmul(var_q_inv, var_p)
    
    term1 = np.dot((mu_p - mu_q).T, np.dot(var_q_inv, (mu_p - mu_q)))
    term2 = np.trace(varprod)
    term3 = np.log(det_var_p / det_var_q)
    term4 = len(mu_p)
    
    return term1+term2-term3-term4


def number_of_active_states_sampling(model, hmm_opts, nsamps = 1000, steps = 100):
    # sample several sequences and extract states
    seqs = [tempered_sampling(model, beta = 1., timesteps=steps, 
                                 sample_obs=True, start_state_max=True, 
                                 sample_var = hmm_opts['sample_var']) for _ in range(nsamps)]
    seqs = [s[1] for s in seqs]
    # for each sample find the number of unique states
    unqs = [np.unique(s) for s in seqs]
    nunqs = np.array([len(u) for u in unqs])
    # what is the median number of active states
    med_active = np.mean(nunqs)
    std_active = np.std(nunqs)
    return med_active, std_active



def number_of_active_states_viterbi(model, seqs, lengths):
    _, state_seqs = model.decode(seqs, lengths=lengths, algorithm='viterbi')
    # for each sample find the number of unique states
    unqs = [np.unique(s) for s in state_seqs]
    nunqs = np.array([len(u) for u in unqs])
    # what is the median number of active states
    med_active = np.mean(nunqs)
    std_active = np.std(nunqs)
    return med_active, std_active
    
    
    
class bird_dataset(object):
    def __init__(self, path2bird):
        self.file = h5py.File(path2bird,'r')
        self.folders = self.file.items()
        self.nfolders = len(list(self.file.keys()))
        print('...total number of folders = %d ...'%(self.nfolders))
        
    def how_many_files(self, day):
        i = 0
        for k,v in self.folders:
            if i == day:
                day = k
                d = v
                print('..... day is %s .....'%(day))
                break
            i += 1
        self.nfiles = list(d.keys())
        print('... available files: %d ...'%(len(self.nfiles)))
        return day
    
    def get(self, day=0):
        day = self.how_many_files(day)
        X = [None for _ in range(len(self.nfiles))]
        for i in range(len(self.nfiles)):
            X[i] = transform(np.array(self.file.get(day + '/' + self.nfiles[i])))
        return X
    
    def close(self):
        self.file.close()
        

        
def create_output(model, outpath, hidden_size, idx, hmm_opts, netG, sequence=[]):
    """
        Creates hmm outputs via sampling, or simply reconstructions from the neural network.
        Outputs can be spectrograms and audio.
        decoder.
        Params
        ------
            model : GaussianHMM model
            outpath : folder where files (images and .wav audio) will be saved
            hidden_size : int, how many hidden states does the model have
            idx : int, index of the day of learning
            hmm_opts : dict, options:
                                - sample_var : float, if 0, it will use the model's learned
                                                variance, else uses sample_var (must be > 0.)
                                - batchsize : int, how many observed vectors in a batch to
                                                decode (LEAVE IT AT 1)
                                - nsamps : 
                                - nsamplesteps : int, how many timesteps to sample
                                - sample_invtemperature : float, inverse temperature parameter
                                                for sampling. Values > 1 lead to more "cold" sampling
                                                . Values < 1 lead to more uniformly random sampling.
                                - imageW : int, spectrogram chunk width (i.e. num time frames)
                                - fit_params : str, same as fit_params for hmm model
                                - get_audio : bool, if True, generates the .wav file from a spectrogram
                                - sample_rate : int, sampling rate for .wav file production
                                
            netG : pytorch generator neural network
            sequence  : list of latent vector sequences (observations) to reconstruct. If empty (default),
                        model is used to create sample sequences
    """
    if len(sequence)==0:
        # create samples
        # if the variance of hmm is learned, by default use that to sample
        if 'c' in hmm_opts['fit_params']:
            sample_variance = 0
        else:
            sample_variance = hmm_opts['sample_var']
        seqs = [tempered_sampling(model, beta = hmm_opts['sample_invtemperature'],
                                  timesteps=hmm_opts['nsamplesteps'], 
                                 sample_obs=True, start_state_max=True, 
                                 sample_var = sample_variance) for _ in range(hmm_opts['nsamps'])]
        seqs = [s[0] for s in seqs]
    else:
        # seqs is a single numpy array of shape [timesteps x latent_dim]
        # so reconstructions are produced 
        seqs = []
        seqs.append(sequence)
        
    # create spectrogram
    spect_out = [None for _ in range(len(seqs))]
    for i in range(len(seqs)):
        spect_out[i] = overlap_decode(seqs[i], netG,  noverlap = hmm_opts['noverlap'],
                                          cuda = hmm_opts['cuda'], get_audio = hmm_opts['get_audio'])
    audio_out = [a[1] for a in spect_out]
    spect_out = [s[0] for s in spect_out]
    # create output folder
    outputfolder = os.path.join(outpath, 'day_'+str(idx)+'_hiddensize_'+str(hidden_size))
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    # save spectrograms
    if len(spect_out)==1:
        plt.figure(figsize=(50,10))
        plt.imshow(spect_out[0], origin='lower', cmap = 'gray')
        # real sequence
        plt.savefig(os.path.join(outputfolder, 
                             'reconstructed_sequence.eps'), dpi = 50, format='eps')
        plt.close()
    else:
        for j in range(nsamps):
            plt.figure(figsize=(50,10))
            plt.imshow(spect_out[j], origin='lower', cmap = 'gray')
            plt.savefig(os.path.join(outputfolder, 
                                 'hiddenstatesize_'+str(hidden_size)+'_sample_'+str(j)+'.eps'), 
                                     dpi = 50, format='eps')
            plt.close()
    # if audio is computed, save that 
    if hmm_opts['get_audio']:
        if len(audio_out)==1:
            save_audio_sample(audio_out[0], 
                              os.path.join(outputfolder,'real_sequence.wav'), hmm_opts['sample_rate'])
        else:
            for j in range(nsamps):
                save_audio_sample(audio_out[j], 
                              os.path.join(outputfolder,
                                           'hiddenstatesize_'+str(hidden_size)+'_sample_'+str(j)+'.wav'),
                                 hmm_opts['sample_rate'])

                
                
def load_netG(netG_file_path, nz = 16, ngf = 128, nc = 1, cuda = False, resnet = False):
    """Load the generator network
    """
    if resnet:
        from models.nets_16col_residual import _netG
    else:
        from models.nets_16col_layernorm import _netG
        
    netG = _netG(nz, ngf, nc)
    netG.load_state_dict(torch.load(netG_file_path))

    if cuda:
        netG = netG.cuda()
    return netG


def load_netE(netE_file_path, nz = 16, ngf = 128, nc = 1, cuda = False, resnet = False):
    """Load the encoder network
    """
    if resnet:
        from models.nets_16col_residual import _netE
    else:
        from models.nets_16col_layernorm import _netE
        
    netE = _netE(nz, ngf, nc)
    netE.load_state_dict(torch.load(netE_file_path))

    if cuda:
        netE = netE.cuda()
    return netE
