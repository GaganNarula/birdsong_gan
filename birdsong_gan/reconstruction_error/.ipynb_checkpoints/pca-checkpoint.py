import numpy as np
from sklearn.decomposition import PCA
import pdb

def learn_pca_model(X, n_components = 0.97, random_state = 0):
    """Learn PCA model on spectrogram chunks
        Params
        ------
            X : list of numpy arrays. each array is a spectrogram chunk
                of size (num fft bins, chunk_len)
            n_components : int or float, if int, it is the exact number of 
                principal components extracted
            random_state : int, random seed
        Returns
        -------
            model : sklearn.decomposition.PCA model
    """
    # flatten each sequence and stack into one array
    X = np.stack([x.flatten() for x in X])
    pca = PCA(n_components = n_components, random_state = random_state)
    pca.fit(X)
    return pca


def pca_encode(x, pca):
    """Project on learned principal components """
    return pca.transform(x.flatten().reshape(1,-1))


def pca_decode(x, pca, imageH = 129, imageW = 16):
    """Project back to input space """
    return pca.inverse_transform(x).reshape((imageH,imageW))


def split_song_sequence(x, chunk_len = 16):
    X = []
    nchunks = x.shape[-1] // chunk_len
    idx = 0
    for n in range(nchunks):
        X.append(x[:,idx : idx + chunk_len])
    return X


def reconstruction(X, model = None, n_components = 0.97, random_state = 0, 
                  chunk_len = 16, imageH = 129):
    """For a list of spectrograms, make reconstruction
        Params
        ------
            X :  either a single spectrogram (numpy.ndarray) or a list of 
                spectrograms, or a list of spectrogram chunks]
            model : sklearn.decomposition.PCA model object. If None, learns
                    one
            n_components : int or float, if int, it is the exact number of 
                        principal components extracted
        Returns
        -------
            Xhat : list
    """ 
    if type(X)==np.ndarray:
        # this is a single spectrogram, so split it 
        X = split_song_sequence(X, chunk_len)
    elif type(X) == list and X[0].shape[-1] > chunk_len:
        # this is a list of song spectrograms
        # -> list of list of chunks
        X = [split_song_sequence(x, chunk_len) for x in X]
        
    if model is None:
        model = learn_pca_model(X, n_components, random_state)
        
    # reconstruct
    Xhat = []
    for i in range(len(X)):
        if type(X[i]) == list:
            recon = []
            for k in range(len(X[i])):
                xhat = pca_decode(pca_encode(X[i][k], model), 
                                 model, imageH, chunk_len)
                recon.append(xhat)
            # reconstruct a single song
            recon = np.concatenate(recon,axis=-1)
            Xhat.append(recon)
        else:
            xhat = pca_decode(pca_encode(X[i][k], model))
            Xhat.append(xhat)
    return Xhat

