from evaluate_hmm import *
import h5py
from glob import glob
import pdb

basepath2models = '/media/songbird/datapartition/HMM/trained_models/merge_6000'
tutor_path = '/media/songbird/datapartition/HMM/trained_models/tutor/models/100h_gauss_diag.pkl'

pupil_paths = [glob(os.path.join(basepath2models, 'm0_b7r16','models', 'm0_150h_gauss_diag.pkl'))[0],
                glob(os.path.join(basepath2models, 'm1_b7r16','models', 'm1_150h_gauss_diag.pkl'))[0],
               glob(os.path.join(basepath2models, 'm2_b7r16','models', 'm2_130h_gauss_diag.pkl'))[0],
               glob(os.path.join(basepath2models, 'm3_b7r16','models', 'm3_150h_gauss_diag.pkl'))[0],
               glob(os.path.join(basepath2models, 'm4_b7r16','models', 'm4_150h_gauss_diag.pkl'))[0],
               glob(os.path.join(basepath2models, 'm5_b7r16','models', 'm5_150h_gauss_diag.pkl'))[0],
               glob(os.path.join(basepath2models, 'm6_b7r16','models', 'm6_150h_gauss_diag.pkl'))[0]]

zseq_folders = ['day9_b7r16', 'day12_b7r16', 'day16_b7r16', 'day19_b7r16', 'day22_b7r16', 'day25_b7r16', 'day28_b7r16']

bird_path = '/home/songbird/data/b7r16_val'

def get_random_spectrogram_sample_from_bird(birdhdfpath, folder_names, Nsamps = 1):
    birdfile = h5py.File(birdhdfpath, 'r')
    base_items = list(birdfile.items())
    directory_names = [b[0] for b in base_items]
    # search for folder name and get random samples
    out = [None for k in range(len(folder_names))]
    for j,f in enumerate(folder_names):
        for i,d in enumerate(directory_names):
            if f==d:
                sequencelist = list(birdfile.get(base_items[i][0]).items())
                idx = np.random.randint(low=0,high=len(sequencelist),size=Nsamps)
                out[j] = [None for m in range(len(sequencelist))]
                for k in idx:
                    out[j][k] = np.array(birdfile.get('/'+d+'/'+day1_items[k][0]))
    return out



def get_random_zsequence_sample_from_bird(birdpath, folder_names, Nsamps = 50):
    folder_zseq = []
    for f in folder_names:
        path2files = os.path.join(birdpath, f, 'z_sequences')
        allfiles = os.listdir(path2files)
        idx = np.random.choice(len(allfiles),size=Nsamps,replace=False)
        zseq = [None for i in range(len(idx))]
        for i in range(len(idx)):
            # load z-sequence
            z = np.load(os.path.join(path2files, allfiles[i]))
            zprev = z[0]
            seq = []
            seq.append(zprev)
            for i in range(1,len(z)):
                if np.sum(z[i]-zprev)!=0.0:
                    seq.append(z[i])
                    zprev = z[i]
                else:
                    break
            seq = np.array(seq)
            zseq[i] = seq
        folder_zseq.append(zseq)
    return folder_zseq


        
