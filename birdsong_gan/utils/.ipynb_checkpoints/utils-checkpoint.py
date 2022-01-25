import sys
import os
from os.path import join
import numpy as np
from PIL import Image
import librosa as lc
import pickle
import itertools
import fnmatch
import shutil
import soundfile as sf
import torch
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def segment_image(im,width=8):
    """Split 2D numpy array of shape (L, T) into a sequence of 'width' length arrays
        output will be a (T//width) length list of arrays each of shape (L, width)
    """
    segments = [im[:, i * width : (i+1) * width] for i in range(im.shape[1]//width)]
    return segments

def to_image(seq,nfft):
    ''' Spectrogram computation for a sequence seq 
        Returns
        -------
            abs (magnitude) and angle (phase)
            of spectrogram
    '''
    nfft_padlen = int(len(seq) + nfft / 2)
    stft = lc.stft(fix_length(seq, nfft_padlen), n_fft=nfft)
    return np.array([np.abs(stft), np.angle(stft)]).transpose(1, 2, 0)


def transform(im):
    """
    This function should be used to transform data into the desired format for the network.
    inverse transoform should be an inverse of this function
    """
    im = from_polar(im)
    im, phase = lc.magphase(im)
    im = np.log1p(im)
    return im


def to_batches(segments,batch_size):
    ''' Split a random number of segments into batch_size sized segments '''
    n_batches = int(np.ceil(len(segments)/batch_size))
    batches = [np.zeros(shape=(batch_size,)+tuple(segments[0].shape)) \
               for i in range(n_batches)]
    for i in range(len(segments)):
        batch_idx = i//batch_size
        idx = i%batch_size
        batches[batch_idx][idx] = segments[i]
    return np.array(batches), len(segments)


def phase_restore(mag, random_phases,n_fft, N=50):
    """Restore Phase with Griffin-Lim algorithm"""
    p = np.exp(1j * (random_phases))

    for i in range(N):
        _, p = lc.magphase(lc.stft(
            lc.istft(mag * p), n_fft=n_fft))
    #    update_progress(float(i) / N)
    return p


def from_image(image,clip_len=None):
    if clip_len:
        return fix_length(lc.istft(from_polar(image)), clip_len)
    else:
        return  lc.istft(from_polar(image))

    
def save_image(image,save_path,save_idx,amplitude_only=False):
    if amplitude_only:
        np.save(join(save_path, str(save_idx) + '.npy'), image[:,:,0])
    else:
        np.save(join(save_path, str(save_idx) + '.npy'), image)


def save_spectrogram(filename,D):
    ''' save a spectrogram image '''
    if D.min() < 0:
        D = D-D.min()
    D = D/D.max()
    I = Image.fromarray(np.uint8(D*255))
    I.save(filename)
    
    
def inverse_transform(im, nfft=256, N=50):
    """Phase restoration with Griffin Lim algorithm"""
    random_phase = im.copy()
    np.random.shuffle(random_phase)
    p = phase_restore((np.exp(im) - 1), random_phase, nfft, N)
    return (np.exp(im) - 1) * p


def transform(im):
    """Converts polar coordinate stft of a spectrogram
        to magnitude and phase, then returns log(1 + magnitude)
    """
    im = from_polar(im)
    im,phase = lc.magphase(im)
    im = np.log1p(im)
    return im

def from_polar(image):
    ''' Get complex spectrogram from polar coordinate '''
    return image[:, :, 0]*np.cos(image[:, :, 1]) + 1j*image[:,:,0]*np.sin(image[:,:,1])


def save_audio_sample(sample,path,samplerate):
    sf.write(path,sample,samplerate=int(samplerate))


def normalize_spectrogram(image,threshold):
    if threshold>1.0:
        image = image/threshold
    image = np.minimum(image,np.ones(shape=image.shape))
    image = np.maximum(image,-np.ones(shape=image.shape))
    return image


def segment_spectrogram(x, thresh = 5, mindur = 5):
    ''' Segment a spectrogram with amplitude thresholding 
        Params
        ------
            x : spectrogram, numpy array of shape (nfft,timeframes)
            thresh : float, amplitude threshold
            mindur : int, minimum number of frames that a vocal segment
                        should last 
        Returns
        -------
            vocal_segs : list, vocal (non-silect) spectrogram segments
            durations : list, number of frames of each vocal segment
            gap_length : list, num frames of silent gap between segments
            onsets : list, frame onset
            offsets : list, frame offset
    '''
    vocals = (x.sum(axis=0) > thresh).astype(int)
    # pad with one zero on each side
    vocals = np.concatenate([np.zeros(1,dtype=int), vocals, np.zeros(1,dtype=int)])
    vocals_diff = np.diff(vocals)
    onsets = np.where(vocals_diff > 0)[0]
    offsets = np.where(vocals_diff < 0)[0]
    
    ON = onsets
    OFF = offsets
    # lengths off by 1
    if np.abs(len(onsets)-len(offsets)) == 1:
        if offsets[0] < onsets[0]:
            OFF = offsets[1:]
        elif onsets[-1] > offsets[-1]:
            ON = onsets[:-1]
        else:
            OFF = offsets
            ON = onsets
    vocal_segs = []
    durations = []
    gap_lengths = []
    gap_start = 0
    to_remove = []
    if len(ON)==0:
        return None,None,None,None,None
    for k in range(len(onsets)):
        vocal_segs.append(x[:, ON[k] : OFF[k]])
        durations.append(OFF[k] - ON[k])
        if durations[-1] < mindur:
            vocal_segs.pop()
        gap_lengths.append(ON[k]-gap_start)
        gap_start = OFF[k]
    return vocal_segs, durations, gap_lengths, onsets, offsets



####### NN Functions ########


def load_netG(netG_file_path, nz = 16, ngf = 128, nc = 1, cuda = False, resnet = False):
    """Load the generator network
    
        Params
        ------
            netG_file_path : str, location of decoder/generator network file (torch state_dict)
            nz : int, number of latent dimensions
            ngf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netG : decoder network
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
    
        Params
        ------
            netE_file_path : str, location of encoder network file (torch state_dict)
            nz : int, number of latent dimensions
            ngf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netE : encoder network
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



def load_netD(netD_file_path, ndf = 128, nc = 1, cuda = False, resnet = False):
    """Load the discriminator network
    
        Params
        ------
            netD_file_path : str, location of encoder network file (torch state_dict)
            ndf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netE : encoder network
    """
    if resnet:
        from models.nets_16col_residual import _netD
    else:
        from models.nets_16col_layernorm import _netD
        
    netD = _netD(ndf, nc)
    netD.load_state_dict(torch.load(netD_file_path))

    if cuda:
        netD = netD.cuda()
    return netD



def load_InceptionNet(inception_net_file_path, ndf = 128, nc = 1, cuda = False, resnet = False):
    """Load the inception discriminator network
    
        Params
        ------
            inception_net_file_path : str, location of encoder network file (torch state_dict)
            ndf : int, multiplier of number of filters per conv layers
            nc : int, (leave 1) number of channels (usually just gray-scale)
            cuda : bool, whether to put on gpu (default device 0)
            resnet : bool, whether this is a resnet type model
        
        Returns
        -------
            netE : encoder network
    """
    if resnet:
        from models.nets_16col_residual import InceptionNet
    else:
        from models.nets_16col_layernorm import InceptionNet
        
    netI = InceptionNet(ndf, nc)
    netI.load_state_dict(torch.load(inception_net_file_path))

    if cuda:
        netI = netI.cuda()
    return netI


@torch.no_grad()
def overlap_encode(sample, netE, transform_sample = False, imageW = 16, noverlap = 0,
                   cuda = True):
    """Encode a spectrogram in an overlapping manner.
        Parameters
        -------
            sample : 2D numpy.ndarray, imageH x imageW (height axis is frequency)
            netE : encoder neural network
            transform_sample : bool, default=False, whether to transform the sample
            imageW : int, default=16, length of spectrogram chunk
            noverlap : int, default=0, overlap between spectrogram chunks
            cuda : bool, default=True, whether to push tensors to gpu
        Returns
        -------
            Z : numpy.ndarray shape = (num_chunks, dimensionality of latent space)
    """
    Z = []
    notdone = True
    idx = 0
    
    if transform_sample:
        sample = transform(sample)
        
    while notdone:
        if idx + imageW > sample.shape[-1]:
            notdone = False
            continue
        # take out a slice 
        x = sample[:,idx : idx + imageW]
        # to tensor
        x = torch.from_numpy(x).float()
        # encode
        if cuda :
            x = x.cuda()
        # reshape
        x = x.view(1,1,x.size(0),x.size(1))
        z = netE(x)
        z = z.cpu().numpy()
        Z.append(z)
        idx = idx + imageW - noverlap
        
    Z = np.stack(Z, axis=0).squeeze()
    return Z



@torch.no_grad()
def overlap_decode(z, netG, noverlap = 0, get_audio = False, cuda = True):
    """Overlap decode. For a given numpy array Z of shape (timesteps , latent_dim)
        the output spectrogram (and optionally also audio) is created. 
        
        Parameters
        -------
            Z : numpy.ndarray, (timesteps , latent_dim)
            netG : generator neural network
            noverlap  : int, default = 0, how much overlap (in spectrogram frames) between 
                        consecutive spectrogram chunks
            get_audio : bool, default = False, to generate audio using Griffin Lim
            cuda : bool, default = True, if True pushes computation on gpu
        Returns
        -------
            X : numpy.ndarray, (nfft bins , chunks)
            X_audio : numpy array, reconstructed audio
    """
    X = [] # to store output chunks
    X_audio = None # for audio representation
    
    # in case only one chunk
    if z.ndim==1:
        z = np.expand_dims(z, axis=0) # add a timestep dimension
        
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).float()
        
    if cuda:
        z = z.cuda()
    
    for i in range(z.shape[0]):

        # reshape
        x = netG(z[i].view(1,z.size(-1))).cpu().numpy().squeeze()
        
        # take out any overlap slices
        # first slice is always fully accepted
        if i > 0:
            x = x[:, noverlap:]
            
        X.append(x)

    X = np.concatenate(X, axis=1) # concatenate in time dimension, axis=1
    
    if get_audio:
        X_audio = inverse_transform(X, N=500)
        X_audio = lc.istft(X_audio)
        
    return X, X_audio


def renormalize_spectrogram(s):
    s = np.exp(s) - 1
    if np.min(s) < 0:
        s = s - np.min(s) 
    s = s/np.max(s)
    return 10*np.log10(s + 0.01)


def rescale_spectrogram(s):
    ''' Just to make a 
        brighter spectrogram     
    '''
    if np.min(s) < 0:
        s = s - np.min(s) 
    s = s/np.max(s)
    return np.log(s+0.01)


def gagan_save_spect(path, spect, frmat='eps'):
    ''' save spectrogram '''
    plt.figure()
    plt.imshow(spect, origin='lower', cmap='gray')
    plt.savefig(path, dpi=100, format=frmat)
    plt.close()
    
    
def make_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    dirs = os.listdir(path)
    for d in dirs:
        if len(os.listdir(join(path, d)))<=3:
            try:
                os.rmdir(join(path, d))
            except:
                shutil.rmtree(join(path,d))
    path += str(datetime.now()).replace(':', '-')
    path = path.replace(' ','_')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(join(path,'net_training_losses'))
    return path


