import sys
import os
sys.path.append(os.pardir)
from configs.cfg import *
import torch
import torch.nn as nn
from torch.utils.data import Dataloader
from models.nets_recurrent import RecurrentNetv1, train
import argparse


import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True


def make_output_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    dirs = os.listdir(path)
    for d in dirs:
        if len(os.listdir(os.path.join(path, d)))<=3:
            try:
                os.rmdir(os.path.join(path, d))
            except:
                shutil.rmtree(os.path.join(path,d))
    path += str(datetime.now()).replace(':', '-')
    path = path.replace(' ','_')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,'losses'))
        os.makedirs(os.path.join(path, 'hist'))
    return path




opts_dict = {'input_path': EXT_PATH,
       'outf': SAVE_PATH,
        'age_weights_path': '', 
       'distance_fun': 'L1', 'subset_age_weights' : [0., 1.], 'workers': 6, 'batchSize': 128, 
        'imageH': 129, 'imageW': 16, 'noverlap':0, 'nz': 16,'nc': 1, 'ngf': 256, 'ndf': 128,'niter': 50,
       'lr': 1e-5, 'lambdaa': 150, 'zreg_weight': 1, 'schedule_lr':False, 'd_noise': 0.1,
       'beta1': 0.5, 'cuda': True, 'ngpu': 1, 'nreslayers': 3, 'z_reg':False, 'mds_loss':False,
       'netGpath': '','netEpath': '','netD1path':'','netD2path':'','log_every': 300,
       'sample_rate': 16000.,'noise_dist': 'normal','z_var': 1.,'nfft': 256, 'get_audio': False,
        'manualSeed': [], 'do_pca': True, 'npca_samples': 1e6, 'npca_components': 256}



parser = argparse.ArgumentParser()
parser.add_argument('--training_path', required=True, help='path to training dataset')
parser.add_argument('--test_path', required=True, help='path to test dataset')
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--external_file_path', default=EXT_PATH, help='path to folder containing bird hdf files')
parser.add_argument('--age_weights_path', type=str, default='', help='path to file containin age_weight for resampling')
parser.add_argument('--subset_age_weights', nargs = '+', help = 'number between 0 and 1 which selects an age range (1 is younger, 0 is older)')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--train_residual', action = 'store_true')
parser.add_argument('--noise_dist', type=str, default = 'normal', help='noise distribution: {normal, uniform, t}')
parser.add_argument('--lambdaa', type = float, default = 100., help = 'weighting for recon loss')
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--ndf', type=int, default=256)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default = 0.00001, help='learning rate')
parser.add_argument('--mds_loss',action='store_true', help='multidimensional scaling type loss')
parser.add_argument('--z_reg', action="store_true", help='whether to regularize the posterior')
parser.add_argument('--zreg_weight', type = float, default =  1, help = 'weight for z regularization')
parser.add_argument('--z_var', type = float, default = 1., help = 'variance of latent prior')
parser.add_argument('--manualSeed', type=int, default=-1, help='random number generator seed')
parser.add_argument('--schedule_lr', action = 'store_true', help='change learning rate')
parser.add_argument('--log_every', type=int, default=300, help='make images and print loss every X batches')
parser.add_argument('--get_audio', action='store_true', help='write wav file from spectrogram')
parser.add_argument('--do_pca', action = 'store_true', help='to learn a PCA model of spectrogram chunks')
parser.add_argument('--npca_components', type = int, default = 256, help = 'how many PCA components ?')
parser.add_argument('--cuda', action='store_true', help='use gpu')
parser.add_argument('--netEpath',type = str, default = '')
parser.add_argument('--netGpath',type = str, default = '')
parser.add_argument('--netD1path',type = str, default = '')
parser.add_argument('--netD2path',type = str, default = '')



def main():
    
    args = parser.parse_args()
    
    
    # initialize the dataset and dataloader objects
    train_dataset = songbird_dataset(args.training_path, opts_dict['imageW'],
                                     args.external_file_path, opts_dict['subset_age_weights'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=opts_dict['batchSize'], sampler = None,
                                             shuffle=shuffle_, num_workers=int(opts_dict['workers']),
                                        drop_last = True)
    
    test_dataset = songbird_dataset(args.test_path, opts_dict['imageW'], args.external_file_path,
                                   opts_dict['subset_age_weights'])
    
    test_dataloader = DataLoader(test_dataset, batch_size= opts_dict['batchSize'],
                                             shuffle=True, num_workers=int(opts_dict['workers']),
                                drop_last = True)
    
    
    # instantiate networks
    