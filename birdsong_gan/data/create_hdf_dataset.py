import numpy as np
import os
from glob import glob
from scipy.io import wavfile
import os
from os.path import join
import h5py
from time import time
from random import shuffle
import pdb
from scipy.signal import resample, iirfilter, sosfiltfilt
import librosa as lc
from librosa.util import fix_length
from utils.utils import transform
from data.dataset import segment_spectrogram



def load_from_folder(base_path, folder_path, extention):
    ''' Load all wav files in "basepath/"+folder_path+extention" 
        by reading them with wavfile program
    '''
    dd = join(base_path, folder_path, extention)
    if os.path.exists(dd):
        files = glob(dd + '/*.wav')
        if len(files)==0:
            dirs = os.listdir(dd)
            files = []
            for d in dirs:
                if os.path.isdir(join(dd,d)):
                    F = glob(join(dd,d) + '/*.wav')
                    files = files + F
    else:
        print("... Folder %s doesnt exist for this bird, aborting ..."%(folder_path))
    if len(files)==0:
        return None, None, None
    # list of all data
    data = []
    for f in files:
        samples, fs = load_wav_file(f)
        data.append(samples)
    return data,fs,files



def load_wav_file(path):
    ''' Load invidiual wav file '''
    fs,wf = wavfile.read(path)
    if wf.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif wf.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    samples = wf / (max_nb_bit + 1.0) 
    return samples, fs


def bandpass_filter(x, order=6, cutoffs = [400, 12000], ftype='butter',
                   btype='bandpass', fs=16000):
    cutoffs = [cutoffs[0] / (fs/2), cutoffs[1] / (fs/2)]
    sos = iirfilter(order, cutoffs, analog=False, ftype=ftype, 
                   btype=btype, rp=0, rs=60, output='sos')
    return sosfiltfilt(sos, x - np.mean(x)) + np.mean(x)
    
    
def downsample(x,down_factor):
    ''' Downsample audio x by factor down_factor '''
    n = x.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = int(np.power(2, y + 1))
    x = np.concatenate((np.zeros((nextpow2-n), dtype=x.dtype), x))
    x = resample(x,len(x)//down_factor)
    return x[(nextpow2-n)//down_factor:]


def librosa_downsample(x, target_fs, orig_fs, res_type='kaiser_best'):
    """Downsample audio signal using librosa"""
    return lc.resample(x, target_sr=target_fs, orig_sr=orig_fs, res_type=res_type,
               fix=True, scale=True) 
    
    
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


def extract_pseudoage_from_folder_names(folders):
    ''' For a single bird, derive a pseudoage estimate 
        by sorting folder names chronologically. Assumes
        folders are dated , e.g. 2011-03-01, 2011-03-02
        Adult birds get pseudo_age = 100.
        Younger birds get pseudo_age = folder number in 
        sorted order
    '''
    if len(folders)<5: 
        # if only few folders, assumes Adult bird
        age = [100 for i in range(len(folders))]
        return age
    age = []
    # assume folder names are sorted in chronological order
    for (i,f) in enumerate(folders):
        if 'tutor' in f:
            age.append(100)
        else:
            age.append(i)
    return age



def make_IDs(birdhdfpath, birdname, id_list, age_weight_list, cnt):
    ''' Each bird gets an ID list, each element of which contains 
        the meta data of a single spectrogram in the whole dataset.
        Age_weight is a number produced from the pseudoage. age_weight = 0.
        means adult and age_weight = 1. means extremely young (experiment
        start).
    '''
    with h5py.File(birdhdfpath, 'r') as birdfile:
        all_grps = list(birdfile.items())
        # cycle over days for this bird
        for g in all_grps:
            day = birdfile.get(g[0])
            day_wav_list = list(day.items())
            if len(day_wav_list)==0:
                continue
            # cycle over files
            for f in day_wav_list:
                duration = np.array(f[1]).shape[1]
                age_weight = 1 - float(f[0].split('_')[-1])/100
                if age_weight==0.0:
                    # for adults age is automaticaly 100, so weight 
                    # be 0 which is undesirable
                    age_weight = 0.1
                age_weight_list.append(age_weight)
                id_list.append({'id':cnt, 'birdname': birdname, 'filepath': birdhdfpath, \
                                'within_file': '/'+g[0]+'/'+f[0], 'age_weight': age_weight,
                               'duration':duration})
                cnt += 1
    return id_list, age_weight_list, cnt



def create_bird_spectrogram_hdf(birdname, birddatapath, outpath, extention = 'songs', downsample_factor=2, 
                                nfft=256, standardize=False, compress_type='gzip', compress_idx=9) -> None:
    '''Creates HDF file for this bird. Each day of recording is a Group, 
    and the spectrogram of each wav file is a Dataset in a Group.
    Attributes are added for each group.
    
    Params
    ------
        birdname : str, e.g. 'b10r16'
        birddatapath : folder containing song recordings in individual folders.
                        E.g. /b10r16/SAP/
        outpath : str, folder name where hdf files will be created
        downsample_factor : int, amount to downsample audio by
        nfft : int, number of fft points for spectrogram default 256.
        standardize : bool, whether to scale spectrograms by standard deviation
        compress_type : str, one of valid h5py compression formats {'gzip','lzf', 'szip'}
        compress_idx : int [0-9] amount of compression, higher values = more compression
                        but lower disk i/o speed.
    Returns
    -------
        None
    '''
    start = time()
    # create hdf file for this bird in the outpath folder
    with h5py.File(join(outpath, birdname), 'w') as birdfile:
        # get all folder names
        folders = os.listdir(birddatapath)
        folders = sorted(folders)
        # determine pseudo age
        ages = extract_pseudoage_from_folder_names(folders)
        if len(ages)==1:
            ages = [ages[0] for i in range(len(folders))]
        # go through each folder, load files, downsample (optional), standardize (optional) and create STFTs

        for (k,fold) in enumerate(folders):
            # create group
            d = birdfile.create_group(fold)
            d.attrs['CLASS'] = 'STFT_with_Magnitude_and_Phase(2nd index in last dimension)'
            d.attrs['DTYPE'] = 'float32'
            d.attrs['PSEUDO_AGE'] = ages[k]
            d.attrs['nfft'] = nfft
            d.attrs['downsamplerate'] = downsample_factor
            d.attrs['standardized'] = standardize
            # load songs
            songs, fs, filenames = load_from_folder(birddatapath, fold, extention)
            
            if songs:
                if downsample_factor:
                    songs = [downsample(i, downsample_factor) for i in songs]
                    if standardize:
                        songs = [s/np.std(s) for s in songs]
                # compute spectrogram
                ims = [to_image(i,nfft) for i in songs]
                for (i,im) in enumerate(ims):
                    # save spectrogram
                    fnam = filenames[i].split('/')[-1] +'_'+ str(ages[k])
                    d.create_dataset(fnam, data=im, compression = compress_type, compression_opts = compress_idx)

    end = time()
    print('..... bird %s finished in %.2f secs.....'%(birdname, end-start)) 
    
    

def create_bird_spectrogram_hdf_external(birdname, birddatapath, outpath, target_sampling_rate=16000, standardize=False, nfft=256,
                                        min_syll_dur_frames=20, energy_thresh_percent=0.1, compress_type='gzip', compression_idx=9) -> None:
    
    '''Creates HDF file for this bird. External data may be structured in different ways,
        so this function simply finds all available .wav files and then creates an hdf using
        those.
        
    Params
    ------
        birdname : str, e.g. 'b10r16'
        birddatapath : folder containing song recordings in individual folders.
                        E.g. /b10r16/SAP/
        outpath : str, folder name where hdf files will be created
        target_sampling_rate : int, resampling the signal to this rate(Hz)
        nfft : int, number of fft points for spectrogram default 256.
        standardize : bool, whether to scale spectrograms by standard deviation
        compress_type : str, one of valid h5py compression formats {'gzip','lzf', 'szip'}
        compress_idx : int [0-9] amount of compression, higher values = more compression
                        but lower disk i/o speed.
    Returns
    -------
        None
    '''
    start = time()
    # create hdf file for this bird in the outpath folder
    with h5py.File(join(outpath, birdname), 'w') as birdfile:
        
        wav_files = []
        for r,d,f in os.walk(birddatapath):
            for file in f:
                if ".wav" in file:
                    wav_files.append(os.path.join(r,file))
        print('..... bird %s number of songs = %d .....'%(birdname, len(wav_files)))
        
        # go through each folder, load files, downsample (optional), standardize (optional) and create STFTs
        d = birdfile.create_group(birdname)
        d.attrs['CLASS'] = 'STFT_with_Magnitude_and_Phase(2nd index in last dimension)'
        d.attrs['DTYPE'] = 'float32'
        d.attrs['PSEUDO_AGE'] = np.nan
        d.attrs['nfft'] = nfft
        d.attrs['fs'] = target_sampling_rate
        d.attrs['standardized'] = standardize
        
        for wav in wav_files:
            # extract last name
            rel_path1, fname = os.path.split(wav)
            rel_path2, gname = os.path.split(rel_path1)
            
            # load the file
            song, fs = load_wav_file(wav)
            
            # band pass filter low freq noise
            song = bandpass_filter(song, 12, fs=fs)
            # downsample to target rate
            song = librosa_downsample(song, target_sampling_rate, fs)

            if standardize:
                songs = song/np.std(song)
            
            # compute spectrogram
            im = to_image(song, nfft)
            im_mag = transform(im)
            imsum = im_mag.sum(axis=0)
            
            # segment syllables , use only magnitude part.
            # threshold is set as minimum energy + 20% of energy range in the spectrogram
            _, _, _, onsets, offsets = segment_spectrogram(im_mag, thresh=np.min(imsum) + 
                                                           energy_thresh_percent*(np.min(imsum)+np.max(imsum)),
                                                            mindur = min_syll_dur_frames)
            if onsets is None:
                continue
                
            # vocal segments : pad each vocal segment (syllable) with 0 on each side.
            # vocal segments should be at least min_syll_dur_frames long
            
            for k in range(len(onsets)):
                if offsets[k] - onsets[k] < min_syll_dur_frames:
                    continue
                # zero pad this segment and use
                vocal_segment = np.concatenate([np.zeros((im.shape[0], 10, 2)), im[:,onsets[k]:offsets[k],:],
                                               np.zeros((im.shape[0], 10, 2))], axis=1)
                
                d.create_dataset(os.path.join(gname, fname+'_'+str(k)), data=vocal_segment, 
                                 compression = compress_type, compression_opts = compression_idx)

    end = time()
    print('..... bird %s finished in %.2f secs.....'%(birdname, end-start)) 
    
    
def get_dataset_keys(f):
    """Get all dataset names in an hdf file"""
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def make_ID_list(path2birdhdfs):
    ''' Loop over the birds and make id lists '''
    birds = os.listdir(path2birdhdfs)
    id_list = []
    age_weight_list = []
    cnt = 0
    for (i,b) in enumerate(birds):
        id_list, age_weight_list, cnt = make_IDs(join(path2birdhdfs, b), b, id_list, age_weight_list, cnt)
        print('..... %d of %d birds indexed .....'%(i, len(birds)))
    return id_list, age_weight_list, cnt



def split_ids_train_test(id_list, train_test_ratio = 0.9):
    ids = [n for n in range(len(id_list))]
    shuffle(ids)
    idstrain = ids[:round(train_test_ratio * len(id_list))]
    idstest = ids[round(train_test_ratio * len(id_list)):]
    id_list_train = [id_list[i] for i in idstrain]
    id_list_test = [id_list[i] for i in idstest]
    return id_list_train, id_list_test



def create_hdfs(path2wavs, out_path, extention='SAP_allfiles', extention2 = 'songs', 
                downsample_factor = 2, nfft = 256):
    """ Loop over birds and create hdfs per bird """
    birds = os.listdir(path2wavs)
    for b in birds:
        print('\n ..... bird is %s ....\n'%(b))
        if os.path.exists(join(out_path, b)):
            continue
        create_bird_spectrogram_hdf(b, join(path2wavs, b, extention), 
                                    out_path, extention2,
                                    downsample_factor, nfft)
        

