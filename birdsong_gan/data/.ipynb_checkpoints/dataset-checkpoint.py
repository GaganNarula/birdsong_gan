import h5py
import torch
from torch.utils import data
import pdb
import librosa as lc
import numpy as np
import os
import pickle
import re



def norm_transform(im):
    ''' Transform complex spectrogram into magnitude and phase 
        , keep only magnitude and normalize by max value
    '''
    im = from_polar(im)
    im, phase = lc.magphase(im)
    im = im/np.max(im)
    return im


def from_polar(image):
    ''' Get complex spectrogram from polar coordinate '''
    return image[:, :, 0]*np.cos(image[:, :, 1]) + 1j*image[:,:,0]*np.sin(image[:,:,1])


def transform(im):
    """
    This function should be used to transform data into the desired format for the network.
    inverse transoform should be an inverse of this function
    """
    im = from_polar(im)
    im, phase = lc.magphase(im)
    im = np.log1p(im)
    return im
    
    
def transform_log01(im):
    """
    This function should be used to transform data into the desired format for the network.
    inverse transoform should be an inverse of this function
    """
    im = from_polar(im)
    im, phase = lc.magphase(im)
    im = np.log(im + 0.01)
    return im    

    
def inverse_transform(im):
    """
    Inverse (or at least almost) of transofrm()
    """
    random_phase = im.copy()
    np.random.shuffle(random_phase)
    p = phase_restore((np.exp(im) - 1), random_phase, opt.nfft, N=50)
    return (np.exp(im) - 1) * p
    
    
def inverse_norm_transform(im, nfft = 256):
    """
    Inverse (or at least almost) of transofrm()
    """
    random_phase = im.copy()
    np.random.shuffle(random_phase)
    p = phase_restore(im, random_phase, nfft, N=50)
    return im * p


def im_loader(path):
    """
    This is the function between data on disk and the network.
    """
    im = np.load(path)
    im = random_crop(im, width=opt.imageW)
    im = transform(im)
    return im 


def random_crop(im,width=16):
    ''' randomly crop a spectogram "im" of shape [nfft, width] '''
    ceil = im.shape[1]-width
    ind = np.random.randint(ceil)
    return im[:,ind:ind+width]


def random_crop_get_two(im, width = 16):
    """Crops a spectrogram at a random slice and gets two consecutive chunks 
    """
    return


def random_crop_transform_contiguous(im, width=16, bsize=64):
    ceil = im.shape[1] - width*bsize
    if ceil <= 0:
        # pad image
        im = np.concatenate([im, np.zeros((im.shape[0], 
                                           np.abs(ceil)+3, 
                                           im.shape[-1]))],axis=1)
        ceil = im.shape[1] - width*bsize
    ind = np.random.randint(ceil)
    chunk = im[:, ind:ind+ width*bsize]
    chunk = transform(chunk)
    return np.reshape(chunk, (bsize, im.shape[0], width))


def phase_restore(mag, random_phases, n_fft, N=50):
    ''' Griffin lim phase restoration '''
    p = np.exp(1j * (random_phases))
    for i in range(N):
        _, p = librosa.magphase(librosa.stft(
            librosa.istft(mag * p), n_fft=n_fft))
    #    update_progress(float(i) / N)
    return p



class songbird_dataset(data.Dataset):
    '''
        Dataset class used for training neural nets. 
        Params
        ------
            path2idlist : id_list generated when hdfs were created
            imageW : spectrogram width (number of time frames) to crop to.
            external_file_path : file path to hdf for this bird
            subset_age : float tuple, [0. 1.], selects age range 
    '''
    def __init__(self, path2idlist, imageW, external_file_path='', 
                 subset_age = None):
        with open(path2idlist, 'rb') as f:
            id_list = pickle.load(f)
        if subset_age:
            w = np.array([i['age_weight'] for i in id_list])
            idx = np.where((subset_age[0] < w) & (w <= subset_age[1]))[0]
            self.id_list = [id_list[i] for i in idx]
        else:
            self.id_list = id_list
        self.imageW = imageW
        self.external_file_path = external_file_path
        
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def __getitem__(self, index):
        ''' Get a random chunk from a spectrogram at index 
            "index" in id_list 
        '''
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
        # get whole spectrogram
        X = np.array(f.get(ID['within_file']))
        f.close()
        # transform 
        X = self.crop_and_transform(X)
        return torch.from_numpy(X).float(), torch.Tensor([age_weight])
    
    def crop_and_transform(self, X):
        X = random_crop(X, width=self.imageW)
        X = transform(X)
        return X

    

class songbird_contiguous_dataset(data.Dataset):
    '''
        Extracts a contiguous set of spectrogram chunks for training
    '''
    def __init__(self, path2idlist, imageW=16, external_file_path=[],
                 subset_age = None, batchSize=64, minlen = None):
        with open(path2idlist, 'rb') as f:
            id_list = pickle.load(f)
        if subset_age:
            w = np.array([i['age_weight'] for i in id_list])
            idx = np.where((subset_age[0] < w) & (w <= subset_age[1]))[0]
            self.id_list = [id_list[i] for i in idx]
        else:
            self.id_list = id_list
        # remove short duration files
        if minlen:
            d = np.array([d['duration'] for d in self.id_list])
            idx = np.where(d > minlen)[0]
            self.id_list = [self.id_list[i] for i in idx]    
        self.bsize = batchSize
        self.imageW = imageW
        self.external_file_path = external_file_path
        self.file = None
        
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def open_h5py_file(self, ID):
        if self.external_file_path:
            birdname = os.path.basename(ID['filepath'])
            self.file = h5py.File(os.path.join(self.external_file_path, birdname),'r')
        else:
            self.file = h5py.File(ID['filepath'], 'r')
        
    def close_file(self):
        self.file.close()
        self.file = None
        
    def __getitem__(self, index):
        # load one wav file and get batchsize contiguous chunk from it
        ID = self.id_list[index]
        age_weight = ID['age_weight']
        # this 'ID' is a dictionary containing several fields,
        # use field 'filepath' and 'within_file' to get data
        # get whole spectrogram
        self.open_h5py_file(ID)
        X = np.array(self.file.get(ID['within_file']))
        self.close_file()
        # transform 
        X = self.crop_and_transform(X)
        return torch.from_numpy(X).float(), torch.Tensor([age_weight])
    
    def crop_and_transform(self, X):
        X = random_crop_transform_contiguous(X, self.imageW, self.bsize)
        return X

    

class songbird_random_sample(object):
    ''' For sampling random spectrograms from training/test id_list '''
    def __init__(self, path2idlist, external_file_path=[]):
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
            self.external_file_path = external_file_path
            
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def get(self, nsamps=1):
        # choose nsamp random files
        idx = np.random.choice(self.__len__(), size=nsamps, replace=False)
        
        X = [None for i in range(nsamps)]
        age_weights = [None for i in range(nsamps)]
        
        for (k,i) in enumerate(idx):
            ID = self.id_list[i]
            age_weights[k] = ID['age_weight']
            
            if self.external_file_path:
                birdname = os.path.basename(ID['filepath'])
                f = h5py.File(os.path.join(self.external_file_path, birdname),'r')
            else:
                f = h5py.File(ID['filepath'], 'r')
                
            X[k] = np.array(f.get(ID['within_file']))
            f.close()
            
        return X, age_weights

    
    
class songbird_full_spectrogram(data.Dataset):
    """This dataset retreives full spectrograms
        Useful for recurrent network training.
        
        __getitem__() method is used by pytorch to
        retrieve a single sample from the dataset.
    """
    def __init__(self, path2idlist, external_file_path = '', subset_age = None,
                max_length = 300):
        
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
            
        if subset_age is not None:
            w = np.array([i['age_weight'] for i in self.id_list])
            idx = np.where((subset_age[0] < w) & (w <= subset_age[1]))[0]
            self.id_list = [self.id_list[i] for i in idx]
            
        self.external_file_path = external_file_path
        self.max_length = max_length
        
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
        
        x = transform(x)
        x = self.pad_to_maxlength(x)
        
        return torch.from_numpy(x).float(), torch.Tensor([age_weight]).float()
    
    def pad_to_maxlength(self, x):
        if x.shape[1] >= self.max_length:
            return x[:,:self.max_length]
        # else pad right
        x = np.concatenate([x, np.zeros((x.shape[0], self.max_length-x.shape[1]))],axis=1)
    
        return x
    
    def get_contiguous_minibatch(self, start_idx, mbatchsize=64):
        ids = np.arange(start=start_idx, stop=start_idx+mbatchsize)
        X = [self.__getitem__(i)[0] for i in ids]
        return X



    
class songbird_full_spectrogram_single_file(data.Dataset):
    """This dataset retreives full spectrograms
        Useful for recurrent network training.
        
        __getitem__() method is used by pytorch to
        retrieve a single sample from the dataset.
    """
    def __init__(self, path2idlist, path2hdf, max_length = 300):
        
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
            
        self.max_length = max_length
        self.birdfile = h5py.File(path2hdf,'r')
        
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def get_random_item(self):
        idx = np.random.choice(len(self.id_list),size=1)
        return self.__getitem__(idx)
        
    def __getitem__(self, index):
        # load one wav file and get a sample chunk from it
        ID = self.id_list[index]
        # this 'ID' is a dictionary containing several fields,
        # use field 'within_file' to get data
        x = np.array(self.birdfile.get(ID['within_file']))
        
        x = transform(x)
        x = self.pad_to_maxlength(x)
        
        return torch.from_numpy(x).float()
    
    def pad_to_maxlength(self, x):
        if x.shape[1] >= self.max_length:
            return x[:,:self.max_length]
        # else pad right
        x = np.concatenate([x, np.zeros((x.shape[0], self.max_length-x.shape[1]))],axis=1)
    
        return x
    
    
    
    
class songbird_spectrogram_chunks_single_file(data.Dataset):
    """This dataset retreives full spectrograms
        Useful for recurrent network training.
        
        __getitem__() method is used by pytorch to
        retrieve a single sample from the dataset.
    """
    def __init__(self, path2idlist, path2hdf, imageW=16):
        
        with open(path2idlist, 'rb') as f:
            self.id_list = pickle.load(f)
            
        self.birdfile = path2hdf
        self.imageW = imageW
        
    def __len__(self):
        # total number of samples
        return len(self.id_list)
    
    def __getitem__(self, index):
        with h5py.File(self.birdfile, 'r') as file:
            # load one wav file and get a sample chunk from it
            ID = self.id_list[index]
            # this 'ID' is a dictionary containing several fields,
            # use field 'within_file' to get data
            x = np.array(file.get(ID['within_file']))
        
        x = self.crop_and_transform(x)
        age = torch.zeros(1) 
        return torch.from_numpy(x).float(), age # none is for the age weight
    
    def crop_and_transform(self, X):
        X = random_crop(X, width=self.imageW)
        X = transform(X)
        return X
    
    
    



class bird_dataset(object):
    ''' Main dataset class for loading, transforming spectrograms 
        from a single bird. No data.Dataset subclass. This class is
        used for analysis, spectrogram display or hmm learning.
    '''
    def __init__(self, path2bird):
        self.file = h5py.File(path2bird,'r')
        self.folders = self.file.items()
        self.nfolders = len(list(self.file.keys()))
        print('...total number of folders = %d ...'%(self.nfolders))
    
    def how_many_files(self, day):
        ''' Find number of song spectrogram files in a day '''
        i = 0
        for k,v in self.folders:
            if i == day:
                day = k
                d = v
                print('... total available files = %d ...'%(len(list(d.keys()))))
                break
            i += 1
            
    def get_file_names(self, day):
        ''' Get folder name corresponding to day "day" '''
        i = 0
        for k,v in self.folders:
            if i == day:
                day = k
                d = v
                break
            i += 1
        nfiles = list(d.keys())
        print('... total available files = %d ...' %(len(nfiles)))
        return nfiles, day
    
    def get(self, day=0, nsamps=-1):
        ''' get "nsamps" spectrograms from day number "day" ''' 
        nfiles,day = self.get_file_names(day)
        # choose nsamp random files
        if nsamps == -1:
            # get all songs
            idx = np.arange(len(nfiles))
        else:
            idx = np.random.choice(len(nfiles), size = nsamps, replace=False)
        X = [None for i in range(len(idx))]
        for (k,i) in enumerate(idx):
            X[k] = transform(np.array(self.file.get(day + '/' + nfiles[i])))
        return X
    
    def make_chunk_tensor_dataset(self, day=0, nsamps=-1, imageW=16,
                                shuffle_chunks=True):
        files = self._filter_files(day)
        nfiles = len(files)
        
        X = [] # list of all the chunks 
        for i in range(nfiles):
            # get and transform spectrogram 
            seq = transform(np.array(self.file.get(files[i])))
            
            L = seq.shape[1] # duration of spectrogram
            idx = 0
            while idx + imageW <= L:
                X.append(seq[:, idx : idx+imageW])
                idx += imageW
                
        # stack
        X = np.stack(X)
        
        if shuffle_chunks:
            np.random.shuffle(X)
            
        # output a tensor dataset instance and the age of the day as a tensor
        return data.TensorDataset(torch.from_numpy(X).float()), torch.FloatTensor([day])
            
    def close(self):
        self.file.close()
        
        

class bird_dataset_single_hdf(object):
    '''Like bird_dataset but works on a single hdf object containing all 
        different birds' data.

        Main dataset class for loading, transforming spectrograms 
        from a single bird. No data.Dataset subclass. This class is
        used for analysis, spectrogram display or hmm learning.
    '''
    def __init__(self, path2hdf, birdname):
        
        self.bird = birdname
        self.file = h5py.File(path2hdf,'r')
        # filter out this birds files
        self.filtered_keys = self._filter_for_bird()
        # find which days exist for this bird
        # assume format of days is YYYY-MM-DD and this 
        # string is in the file name
        self.day_names = self._which_days()
        self.ndays = len(self.day_names)
        print(f'... total number of folders for bird {birdname} = {self.ndays} ...')

    def _filter_for_bird(self):
        keyss = list(self.file.keys())
        return list(filter(lambda x: self.bird in x, keyss))

    def _which_days(self):
        day_names = []
        for f in self.filtered_keys:
            match = re.search(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',f)
            if match is not None:
                day_names.append(match.group())
        return list(set(day_names))

    def _filter_files(self, day=0):
        dayname = self.day_names[day]
        return list(filter(lambda x: dayname in x, self.filtered_keys))

    def get(self, day=0, nsamps=-1):
        ''' get "nsamps" spectrograms from day number "day" ''' 
        files = self._filter_files(day)
        nfiles = len(files)
        # choose nsamp random files
        if nsamps == -1:
            # get all songs
            ids = np.arange(nfiles)
        else:
            ids = np.random.choice(nfiles, size = nsamps, replace=False)

        X = [None for i in range(len(ids))]
        for (k,i) in enumerate(ids):
            X[k] = transform(np.array(self.file.get(files[i])))
        return X
    
    def make_chunk_tensor_dataset(self, day=0, nsamps=-1, imageW=16,
                                shuffle_chunks=True):
        files = self._filter_files(day)
        nfiles = len(files)
        
        X = [] # list of all the chunks 
        for i in range(nfiles):
            # get and transform spectrogram 
            seq = transform(np.array(self.file.get(files[i])))
            
            L = seq.shape[1] # duration of spectrogram
            idx = 0
            while idx + imageW <= L:
                X.append(seq[:, idx : idx+imageW])
                idx += imageW
                
        # stack
        X = np.stack(X)
        
        if shuffle_chunks:
            np.random.shuffle(X)
            
        # output a tensor dataset instance and the age of the day as a tensor
        return data.TensorDataset(torch.from_numpy(X).float()), torch.FloatTensor([day])
            
    def close(self):
        self.file.close()
        
        
        
        
class songbird_syllable_dataset(data.Dataset):
    ''' A syllable dataset. Random sample gets a random syllable from a 
        random spectrogram. 
    '''
    def __init__(self, path2idlist, external_file_path=[], 
                 subset_age = None, sequence_len = 30):
        with open(path2idlist, 'rb') as f:
            id_list = pickle.load(f)
        if subset_age:
            w = np.array([i['age_weight'] for i in id_list])
            idx = np.where((subset_age[0] < w) & (w <= subset_age[1]))[0]
            self.id_list = [id_list[i] for i in idx]
        else:
            self.id_list = id_list
        self.external_file_path = external_file_path
        self.N = sequence_len
        
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
        # get whole spectrogram
        X = np.array(f.get(ID['within_file']))
        f.close()
        # transform 
        X = self.select_one_syll(X, self.N)
        return torch.from_numpy(X).float(), torch.Tensor([age_weight])
    
    def select_one_syll(self, X, N = 30):
        X = transform(X)
        out = segment_spectrogram(X, thresh = 1.1, mindur = 10)
        if out is None:
            # choose a random slice of X
            Sylls = []
        else:
            Sylls = out[0]
        if len(Sylls)==0:
            # short silence
            #X = np.zeros((129,N))
            # random segment
            if X.shape[-1] > N:
                X = random_crop(X, N)
            elif X.shape[-1] < N:
                X = np.concatenate([X, 
                                np.zeros((129, N - X.shape[1]))],axis=1)
        else:
            # randomly choose one of them
            idx = np.random.choice(len(Sylls),size=1)
            X = [Sylls[i] for i in idx][0]
            # X is of size 129 x T
            # pad with 0 to 100, with some offset silence
            if X.shape[1] < N:
                X = np.concatenate([X, 
                                np.zeros((129, N - X.shape[1]))],axis=1)
            elif X.shape[1] > N:
                X = X[:,:N]
        return X
    



    
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