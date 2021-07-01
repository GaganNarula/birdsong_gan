import sys
import os
sys.path.append(os.pardir)
from configs.cfg import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import songbird_full_spectrogram
from models.nets_recurrent import RecurrentNetv1
from collections import namedtuple
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



def evaluate(model, dataloader, opts):
    """Evaluate model on a validation set dataloader"""
    costfunc = nn.MSELoss(reduction='mean')
    val_loss = []
    
    with torch.no_grad():
        for i, (x,age) in enumerate(dataloader):
            x = x.permute(0,2,1)
            x, y = model.split_input(x, opts['ksteps_ahead'])
            
            if opts['cuda']:
                x = x.cuda()
                y = y.cuda()
                
            yhat = model(x)
            # compute error 
            loss = costfunc(yhat, y)
            val_loss.append(loss.item())
            
    return np.array(val_loss)


def train(model, traindataloader, testdataloader, opts):
    """Traning function. 
        Params
        ------
            model : 
            dataloader : torch.utils.data.Dataloader map style
            opts : options dict
                    - nepochs : int, how many epochs of training
    """
    
    costfunc = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = opts['lr'], weight_decay = opts['l2'], 
                                 betas=(opts['beta1'], 0.999))
    
    N = len(traindataloader)
    train_loss = []
    
    for n in range(opts['nepochs']):

        for i, (x,age) in enumerate(traindataloader):
            
            optimizer.zero_grad()
            model.zero_grad()
            
            # make sure the shape is (N,L,H)
            x = x.permute(0,2,1)
            # input is past values, predicted are future
            # split the tensor on the time axis to get target
            # and input 
            x, y = model.split_input(x, opts['ksteps_ahead'])
            if opts['cuda']:
                x = x.cuda()
                y = y.cuda()
                
            yhat = model(x)
            
            # compute error
            loss = costfunc(yhat , y)
            loss.backward()
            
            optimizer.step()
            train_loss.append(loss.item())

            if i%opts['log_every'] == 0:
                print("..... Epoch %d, minibatch [%d/%d], training loss = %.3f ....."%(n,i,N,train_loss[-1]))
            
        # model checkpoint
        if opts['checkpoint_models']:
            torch.save(model.state_dict(), '%s/recnet_epoch_%d.pth' % (opts['outf'], n))
        
        # evaluate on test / validation set
        val_loss = evaluate(model, testdataloader, opts)
        
    return model, np.array(train_loss)




opts_dict = {'input_path': EXT_PATH, 'outf': SAVE_PATH,
        'age_weights_path': '', 
       'distance_fun': 'L1', 'subset_age_weights' : [0., 1.], 'workers': 6, 'batchSize': 128, 
        'imageH': 129, 'imageW': 16, 'noverlap':0, 'nz': 16, 'nepochs': 50, 'log_every': 500,
        'checkpoint_models':  True, 'ksteps_ahead': 5, 
       'lr': 1e-5, 'l2': 0.0, 'schedule_lr':False, 'beta1': 0.5, 'cuda': True, 'ngpu': 1,
        'log_every': 300, 'sample_rate': 16000.,'noise_dist': 'normal','z_var': 1.,
        'nfft': 256, 'get_audio': False,'manualSeed': [], 'do_pca': True, 
        'num_rnn': 200, 'num_layers': 2, 'dropout': 0.1, 'num_linear':200,
        'bidirectional':True, 'leak': 0.1, 'max_length': 400}



parser = argparse.ArgumentParser()
parser.add_argument('--training_path', required=True, help='path to training dataset')
parser.add_argument('--test_path', required=True, help='path to test dataset')
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--external_file_path', default=EXT_PATH, help='path to folder containing bird hdf files')
parser.add_argument('--age_weights_path', type=str, default='', help='path to file containin age_weight for resampling')
parser.add_argument('--subset_age_weights', nargs = '+', help = 'number between 0 and 1 which selects an age range (1 is younger, 0 is older)')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--nz', type=int, default=129, help='size of the latent z vector')
parser.add_argument('--imageH',type=int, default=129, help='number of input feature dimensions, spectrogram freq bins')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default = 0.00001, help='learning rate')
parser.add_argument('--l2',type=float, default=0.0001, help='L2 weight regularization')
parser.add_argument('--num_rnn',type=int, default=200, help='number of recurrent network neurons')
parser.add_argument('--num_linear', type=int, default=200, help='number of hidden mlp neurons')
parser.add_argument('--ksteps_ahead', type=int, default=5, help='how many steps ahead to predict')
parser.add_argument('--max_length',type=int, default=400, help='maximum spectrogram length, needed for rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of recurrent layers')
parser.add_argument('--dropout',type=float, default=0.1, help='dropout % for recurrent hidden layers')
parser.add_argument('--leak',type=float, default=0.1, help='leaky relu leak value')
parser.add_argument('--manualSeed', type=int, default=-1, help='random number generator seed')
parser.add_argument('--schedule_lr', action = 'store_true', help='change learning rate')
parser.add_argument('--log_every', type=int, default=300, help='make images and print loss every X batches')
parser.add_argument('--cuda', action='store_true', help='use gpu')



def main():
    
    args = parser.parse_args()
    
    outf = make_output_folder(args.outf)
    
    if args.manualSeed==-1:
        opts_dict['manualSeed'] = random.randint(1, 10000) # fix seed
    else:
        opts_dict['manualSeed'] = args.manualSeed
        
    V = vars(args)
    for k,v in V.items():
        if k in opts_dict:
            opts_dict[k] = v
    opts_dict['outf'] = outf
    opts_dict['subset_age_weights'] = [float(w) for w in args.subset_age_weights]
    
    # save opts
    joblib.dump(opts_dict,os.path.join(opts_dict['outf'],'opts_dict.pkl'))
    
    
    # initialize the dataset and dataloader objects
    train_dataset = songbird_full_spectrogram(args.training_path, args.external_file_path, 
                                              opts_dict['subset_age_weights'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=opts_dict['batchSize'], sampler = None,
                                             shuffle=True, num_workers=int(opts_dict['workers']),
                                        drop_last = True)
    
    test_dataset = songbird_full_spectrogram(args.test_path, args.external_file_path,
                                   opts_dict['subset_age_weights'])
    
    test_dataloader = DataLoader(test_dataset, batch_size= opts_dict['batchSize'],
                                             shuffle=True, num_workers=int(opts_dict['workers']),
                                drop_last = True)
    
    
    # instantiate network
    model =  RecurrentNetv1(Hin=opts_dict['imageH'], nz=opts['imageH'],
                            nrnn=opts_dict['num_rnn'], nlin=opts['num_linear'], 
                            nlayers=opts_dict['num_layers'], bidirectional= opts_dict['bidirectional'], 
                            dropout=opts_dict['dropout'], leak=opts_dict['leak'])
    if opts['cuda']:
        model = model.cuda()
        
    # train
    model, train_loss = train(model, train_dataloader, test_dataloader, opts_dict)
    
    
    
    