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
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def segment_image(im,width=8):
    segments = [im[:,i*width:(i+1)*width] for i in range(im.shape[1]//width)]
    return segments


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
    random_phase = im.copy()
    np.random.shuffle(random_phase)
    p = phase_restore((np.exp(im) - 1), random_phase, nfft, N)
    return (np.exp(im) - 1) * p


def transform(im):
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


def load_netG(netG_file_path, ngpu = 1, nz = 16, ngf = 128, nc = 1, cuda = False):
    netG = _netG(nz, ngf, nc)
    netG.apply(weights_init)
    netG.load_state_dict(torch.load(netG_file_path))

    if cuda:
        netG.cuda()
    netG.mode(reconstruction=True)
    return netG


def load_netE(netE_file_path, ngpu = 1, nz = 16, ngf = 128, nc = 1, cuda = False):
    netE = _netE(nz, ngf, nc)
    netE.apply(weights_init)
    netE.load_state_dict(torch.load(netE_file_path))

    if cuda:
        netE.cuda()
    return netE


def decode_by_batch(zhat, netG,  batch_size = 64, imageH=129, imageW=16, 
                    cuda = False, get_audio=False):
    ''' Decode latent vectors to get spectrogram chunks '''
    if type(zhat)==np.ndarray:
        zhat = torch.from_numpy(zhat).float()
        zhat = zhat.resize_(zhat.shape[0], zhat.shape[1], 1, 1)
    if cuda:
        zhat = zhat.cuda()
    out_shape = [imageH, imageW]
    reconstructed_samples = []
    recon_audio = []
    # do inference in batches
    nbatches = round(zhat.size(0)/batch_size)
    i = 0
    with torch.no_grad():
        for n in range(nbatches):
            reconstruction = netG(zhat[i:i+batch_size])
            i += batch_size
            for k in range(reconstruction.data.cpu().numpy().shape[0]):
                reconstructed_samples.append(reconstruction.data[k].cpu().numpy().reshape(out_shape))
                if get_audio:
                    recon_audio.append(inverse_transform(reconstruction.data[k].cpu().numpy().reshape(out_shape), N=500))
    reconstructed_samples = np.concatenate(reconstructed_samples, axis=1)
    if get_audio:
        recon_audio = np.concatenate(recon_audio, axis=1)
        recon_audio = lc.istft(reconstructed_samples)*2
    return rescale_spectrogram(reconstructed_samples), recon_audio


def encode(sample, netE, transform_sample=True, batch_size=64, imageH=129, imageW=16, 
           return_tensor=False, cuda=True):
    ''' Encode spectrogram chunks from one spectrogram '''
    sample_segments = segment_image(sample,width=imageW)
    if transform_sample:
        sample_segments = [transform(k) for k in sample_segments]
    sample_batches, num_segments = to_batches(sample_segments, batch_size)
    input = torch.FloatTensor(batch_size, 1, imageH, imageW)
    if cuda:
        input = input.cuda()
    out_shape = [imageH, imageW]
    cnt = 0
    z = []
    with torch.no_grad():
        for j in range(len(sample_batches)):
            input.data.copy_(torch.from_numpy(sample_batches[j].reshape(input.size())))
            encoding = netE(input)
            if return_tensor:
                z.append(encoding)
            else:
                z.append(encoding.data.cpu().numpy().reshape(batch_size, encoding.size(1)))
    if return_tensor:
        z = torch.cat(z, dim=0)
    else:
        z = np.concatenate(z, axis=0)
        # correct length
        nsegs_in_image = int(round(sample.shape[1]/imageW))
        z = z[:nsegs_in_image, :]
    return z


def encode_and_decode(sample, netE, netG, batch_size=64, method=1, \
                      imageH=129, imageW=8, cuda= True, transform_sample=True, return_tensor=False):
    
    sample_segments = segment_image(sample,width=imageW)
    if transform_sample:
        sample_segments = [transform(k) for k in sample_segments]
    sample_batches, num_segments = to_batches(sample_segments, batch_size)
    reconstructed_samples = []    
    z = []
    input = torch.FloatTensor(batch_size, 1, imageH, imageW)
    if cuda:
        input = input.cuda()
    out_shape = [imageH, imageW]
    cnt = 0
    for j in range(len(sample_batches)):
        input.data.copy_(torch.from_numpy(sample_batches[j].reshape(input.size())))
        encoding = netE(input)
        if return_tensor:
            z.append(encoding)
        else:
            z.append(encoding.data.cpu().numpy().reshape(batch_size, encoding.size(1)))
            
        reconstruction = netG(encoding)
        if return_tensor:
            reconstructed_samples.append(reconstruction)
        else:
            for k in range(reconstruction.data.cpu().numpy().shape[0]):
                if cnt<num_segments:
                    if method==1:
                        reconstructed_samples.append(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape))
                    elif method==2:
                        reconstructed_samples.append( \
                                                    inverse_transform(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape)))
                    elif method==3:
                        reconstructed_samples.append(get_spectrogram(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape)))
                    cnt+=1
    
    if return_tensor:
        z = torch.cat(z, dim=0)
        reconstructed_samples = torch.cat(reconstructed_samples, dim = 1)
    else:
        reconstructed_samples = np.concatenate(reconstructed_samples,axis=1)
        if method == 2:
            reconstructed_audio = lc.istft(reconstructed_samples)
        else:
            reconstructed_audio = None
        z = np.concatenate(z, axis = 0)
    return z, reconstructed_samples, reconstructed_audio


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