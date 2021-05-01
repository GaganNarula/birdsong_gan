import sys
sys.path.append(r'/home/gagan/code/birdsong_gan/birdsong_gan')
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from datetime import datetime
from data.dataset import *
from utils.utils import *
from reconstruction_error.pca import learn_pca_model
import pdb
import joblib
import gc

from configs.cfg import EXT_PATH, SAVE_PATH


opts_dict = {'input_path': EXT_PATH,
       'outf': SAVE_PATH,
        'age_weights_path': '',
       'distance_fun': 'L1', 'subset_age_weights' : [0., 1.], 'workers': 6, 'batchSize': 128, 
        'imageH': 129, 'imageW': 16, 'noverlap':0, 'nz': 16,'nc': 1, 'ngf': 256, 'ndf': 128,'niter': 50,
       'lr': 1e-5, 'lambdaa': 150, 'zreg_weight': 1, 'schedule_lr':False, 'd_noise': 0.1,
       'beta1': 0.5, 'cuda': True, 'ngpu': 1, 'nreslayers': 3,
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
    
    
def noise_t():
    ''' t distributed noise '''
    random_sample = t_dist.rvs(10,size=(opts_dict['batchSize'],nz,1,1))
    out = Variable(torch.from_numpy(random_sample.astype(np.float32)))
    if opts_dict['cuda']:
        return out.cuda()
    else:
        return out
        
        
# some cuda / cudnn settings for memory issues
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.deterministic = True
cudnn.benchmark = True
cudnn.allow_tf32 = True


if __name__ == '__main__':
    args = parser.parse_args()
    
    # choose which network type to train
    if args.train_residual:
        from models.nets_16col_residual import _netG, _netE, _netD, weights_init 
    else:
        from models.nets_16col_layernorm import _netG, _netE, _netD, weights_init 
        
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
    
    # fix seed
    print("Random Seed: ", opts_dict['manualSeed'])
    random.seed(opts_dict['manualSeed'])
    torch.manual_seed(opts_dict['manualSeed'])
    if opts_dict['cuda']:
        torch.cuda.manual_seed(opts_dict['manualSeed'])
    
    if opts_dict['cuda']:
        device = 'cuda:0'
    else:
        device = 'cpu'
    if torch.cuda.is_available() and not opts_dict['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    #### setup datasets and dataloaders ####
    # load age_weights
    if len(args.age_weights_path)>0:
        with open(args.age_weights_path, 'rb') as f:
            age_weights = pickle.load(f)
        
        # initialize the dataset and dataloader objects
        train_sampler = WeightedRandomSampler(age_weights, num_samples = len(age_weights), replacement=False)
        shuffle_ = False
    else:
        train_sampler = None
        shuffle_ = True
    # initialize the dataset and dataloader objects
    train_dataset = songbird_dataset(args.training_path, opts_dict['imageW'],
                                     args.external_file_path, opts_dict['subset_age_weights'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=opts_dict['batchSize'], sampler = train_sampler,
                                             shuffle=shuffle_, num_workers=int(opts_dict['workers']),
                                        drop_last = True)
    
    test_dataset = songbird_dataset(args.test_path, opts_dict['imageW'], args.external_file_path,
                                   opts_dict['subset_age_weights'])
    
    test_dataloader = DataLoader(test_dataset, batch_size= opts_dict['batchSize'],
                                             shuffle=True, num_workers=int(opts_dict['workers']),
                                drop_last = True)
    # for example outputs
    sample_dataset = songbird_random_sample(args.training_path, args.external_file_path)
    
    
    ### make models ###
    # useful renaming
    nz = opts_dict['nz']
    ngf = opts_dict['ngf']
    ndf = opts_dict['ndf']
    nc = opts_dict['nc']
    logpt = opts_dict['log_every']
    
    # custom weights initialization called on networks
    netG = _netG(nz, ngf, nc)
    netG.apply(weights_init)
    if opts_dict['netGpath'] != '':
        netG.load_state_dict(torch.load(opts_dict['netG']))
    
    netD1 = _netD(ndf,nc)
    netD1.apply(weights_init)
    if opts_dict['netD1path'] != '':
        netD1.load_state_dict(torch.load(opts_dict['netD1']))

    netD2 = _netD(ndf,nc)
    netD2.apply(weights_init)
    if opts_dict['netD2path'] != '':
        netD2.load_state_dict(torch.load(opts_dict['netD2']))
        
    netE = _netE(nz, ngf, nc)
    netE.apply(weights_init)
    if opts_dict['netEpath'] != '':
        netE.load_state_dict(torch.load(opts_dict['netE']))
    
    if opts_dict['cuda']:
        netD1 = netD1.cuda()
        netD2 = netD2.cuda()
        netG = netG.cuda()
        netE = netE.cuda()
        
    # Define loss function
    criterion_gan = nn.BCELoss()
    if opts_dict['distance_fun']=='L1':
        print('Using L1 loss')
        criterion_dist = nn.L1Loss()
    else:
        print('Using L2 loss')
        criterion_dist = nn.MSELoss()
        
    # noise variable
    noise = torch.FloatTensor(opts_dict['batchSize'],nz,1,1).to(device)
    
    # downsample function for reconstruction error 
    downsample_pth = torch.nn.AvgPool2d(3, stride=4)
    
    
    # setup optimizer
    optimizerD1 = optim.Adam(netD1.parameters(), lr = opts_dict['lr'], betas = (opts_dict['beta1'], 0.999))
    optimizerD2 = optim.Adam(netD2.parameters(), lr = opts_dict['lr'], betas = (opts_dict['beta1'], 0.999))
    optimizerG = optim.Adam(itertools.chain(netG.parameters(),
                                            netE.parameters()), lr = opts_dict['lr'], betas = (opts_dict['beta1'], 0.999))
    
    # optional learning rate scheduler
    if opts_dict['schedule_lr']:
        lambda1 = lambda epoch: (epoch+1) / 2
        schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda = lambda1)
        schedulerD1 = optim.lr_scheduler.LambdaLR(optimizerD1, lr_lambda = lambda1)
        schedulerD2 = optim.lr_scheduler.LambdaLR(optimizerD2, lr_lambda = lambda1)
    
    #losses
    minibatchLossD1 = []
    minibatchLossG1_rec = []
    minibatchLossG1_gan = []
    minibatchLossD2 = []
    minibatchLossG2 = []
    
    # label noise for discriminator
    # probability that a label is wrongly labelled
    d_prob = torch.FloatTensor([1.-opts_dict['d_noise']]).to(device)
    # useful to flip labels randomly
    def true_wp(prob, size, device):
        p = torch.rand(size).to(device).float()
        p = (p < prob).float()
        return p 
    
    # book keeping
    per_epoch_avg_loss_recon = np.zeros(opts_dict['niter'])
    per_epoch_avg_loss_gan = np.zeros(opts_dict['niter'])
    per_epoch_std_loss_recon = np.zeros(opts_dict['niter'])
    per_epoch_std_loss_gan = np.zeros(opts_dict['niter'])
    
    # PCA learning
    if opts_dict['do_pca']:
        Xpca = [] # will store some chunks
        pca_model = None
        
    # training loop
    for epoch in range(opts_dict['niter']):
        for i, (data, age) in enumerate(train_dataloader):
            data = data.view(data.size(0),nc,data.size(1),data.size(2)).to(device).float()
            # map data X -> Z latent
            encoding = netE(data)
            # map Z -> Xhat
            reconstruction = netG(encoding)
            netD1.zero_grad()
            # map  Xhat -> class [0: Fake, 1: Real] (Make discriminator give it value 1)
            pred_rec_d1 = netD1(reconstruction.detach())
            # map X -> class (maximize D)
            pred_real_d1 = netD1(data)
            # For discriminator, the Pr(class=1|X) = 0.9, true_wp = label with that probability
            err_real_d1 = criterion_gan(pred_real_d1, true_wp(d_prob,pred_real_d1.size(),device))
            # For disc, probability this is a reconstruction, the Pr(class=1| Xhat) = 0.1 = d_noise
            err_fake_d1 = criterion_gan(pred_rec_d1, true_wp(1.-d_prob,pred_real_d1.size(),device))
            err_d1 = err_real_d1 + err_fake_d1
            err_d1.backward()
            # minimize  -logD(X) and maximize -log(D(Xhat)) only w.r.t Discriminator params!
            optimizerD1.step()

            netG.zero_grad()
            netE.zero_grad()
            netD1.zero_grad()
            # map Xhat -> class
            pred_rec_d1 = netD1(reconstruction)
            labell = torch.FloatTensor(opts_dict['batchSize'],1).fill_(1.).to(device) # true label
            errG_discrim = criterion_gan(pred_rec_d1, labell)
            errG_recon = (criterion_dist(reconstruction, data) + \
                          criterion_dist(downsample_pth(reconstruction),downsample_pth(data))) * opts_dict['lambdaa']
            #errG_recon = criterion_dist(downsample_pth(reconstruction),downsample_pth(input)) * opts_dict['lambdaa']
            err_g_d1 = errG_discrim + errG_recon
            err_g_d1.backward()
            # maximize log D(Xhat) or minimize -log D(Xhat) + MSE for encoder and generator
            optimizerG.step()

            # ------------- Diffusion step ---------------
            netE.zero_grad()
            netG.zero_grad()
            encoding = netE(data)
            
            reconstruction = netG(encoding)
            netD2.zero_grad()
            pred_rec_d2 = netD2(reconstruction.detach())
            err_real_d2 = criterion_gan(pred_rec_d2, true_wp(d_prob,pred_real_d1.size(),device))
            
            if args.noise_dist == 't':
                noise = noise_t()
            elif args.noise_dist == 'normal':
                noise.normal_(0., opts_dict['z_var'])
            else:
                noise.uniform_(-opts_dict['z_var'],opts_dict['z_var'])
            
            netG.zero_grad()
            fake = netG(noise)
            pred_fake_d2 = netD2(fake.detach())
            err_fake_d2 = criterion_gan(pred_fake_d2, true_wp(1.-d_prob,pred_real_d1.size(),device))
            err_d2 = err_real_d2 + err_fake_d2
            err_d2.backward()
            optimizerD2.step()
            
            #------ extra regularization for z------#
            if args.z_reg:
                err_E = opts_dict['zreg_weight'] * criterion_dist(netE(fake), noise.squeeze())
                err_E.backward(retain_graph=True)
                optimizerG.step()
                
            netE.zero_grad()
            netG.zero_grad()
            netD2.zero_grad()
            pred_fake_d2 = netD2(fake)
            labell = torch.FloatTensor(opts_dict['batchSize'],1).fill_(1.).to(device) # true label
            err_g_d2 = criterion_gan(pred_fake_d2, labell)
            err_g_d2.backward()
            optimizerG.step()
            
            if opts_dict['do_pca']:
                if len(Xpca) < opts_dict['npca_samples']:
                    data = data.detach().cpu().numpy().squeeze()
                    for d in data:
                        Xpca.append(d)
                
            # SAVE LOSSES
            minibatchLossD1.append(err_d1.item())
            minibatchLossD2.append(err_d2.item())
            minibatchLossG1_gan.append(errG_discrim.item())
            minibatchLossG1_rec.append(errG_recon.item())
            minibatchLossG2.append(err_g_d2.item())


            ### SHOW LOSS AFTER SOME BATCHES ####
            if (i % logpt == 0) & (i > 0):
                print('[%d/%d][%d/%d] D1: %.2f (%.2f) D2: %.2f (%.2f) G1_gan: %.2f (%.2f) G1_rec: %.2f (%.2f) G2: %.2f (%.2f)'
                      % (epoch, opts_dict['niter'], i, len(train_dataloader),
                        np.mean(minibatchLossD1[-logpt:]), np.std(minibatchLossD1[-logpt:]),
                         np.mean(minibatchLossD2[-logpt:]), np.std(minibatchLossD2[-logpt:]),
                         np.mean(minibatchLossG1_gan[-logpt:]), np.std(minibatchLossG1_gan[-logpt:]),
                         np.mean(minibatchLossG1_rec[-logpt:]), np.std(minibatchLossG1_rec[-logpt:]),
                         np.mean(minibatchLossG2[-logpt:]), np.std(minibatchLossG2[-logpt:])
                         )
                      )
                
                # sample and reconstruct
                with torch.no_grad():
                    fixed_noise = torch.FloatTensor(opts_dict['batchSize'],nz,1,1).to(device)
                        
                    if args.noise_dist == 't':
                        fixed_noise = noise_t()
                    elif args.noise_dist == 'normal':
                        fixed_noise.normal_(0. ,opts_dict['z_var'])
                    else:
                        fixed_noise.uniform_(-opts_dict['z_var'],opts_dict['z_var'])

                    
                    fake = netG(fixed_noise)
                    out_shape = [opts_dict['imageH'], opts_dict['imageW']]
                    fake_spectrograms =[fake.data[k].cpu().numpy().reshape(out_shape) for k in range(8)]
                    fake_spectrograms = np.concatenate(fake_spectrograms,axis=1)
                    gagan_save_spect('%s/fake_samples_epoch_%03d_batchnumb_%d.png' 
                                         % (opts_dict['outf'], epoch, i),rescale_spectrogram(fake_spectrograms))
                    gagan_save_spect('%s/fake_samples_epoch_%03d_batchnumb_%d.eps' 
                                         % (opts_dict['outf'], epoch, i),rescale_spectrogram(fake_spectrograms))


                    # randomly sample a file and save audio sample
                    sample = sample_dataset.get(nsamps=1)[0] # first element of list output 
                    sample = sample[0]
                    
                    # audio
                    if opts_dict['get_audio']:
                        try:
                            save_audio_sample(lc.istft(inverse_transform(transform(sample))), \
                                                  '%s/input_audio_epoch_%03d_batchnumb_%d.wav' % 
                                                  (opts_dict['outf'], epoch, i), opts_dict['sample_rate'])
                        except:
                            print('..audio buffer error, skipped audio file generation')

                    # save original spectrogram
                    gagan_save_spect('%s/input_spect_epoch_%03d_batchnumb_%d.eps'
                                         % (opts_dict['outf'], epoch, i), 
                                         rescale_spectrogram(transform(sample)))
                    # save reconstruction
                    zvec = overlap_encode(sample, netE, transform_sample = True, imageW = opts_dict['imageW'],
                                           noverlap = opts_dict['noverlap'], cuda = opts_dict['cuda'])
                    spect, audio = overlap_decode(zvec, netG, noverlap = opts_dict['noverlap'], get_audio = opts_dict['get_audio'], 
                                                  cuda = opts_dict['cuda'])
                    # save reconstructed spectrogram
                    spect = rescale_spectrogram(spect)
                    gagan_save_spect('%s/rec_spect_epoch_%03d_batchnumb_%d.eps' % (opts_dict['outf'], epoch, i), spect)
                    
                    if opts_dict['get_audio']:
                        try:
                            save_audio_sample(audio,'%s/rec_audio_epoch_%03d_batchnumb_%d.wav' % 
                                                  (opts_dict['outf'], epoch, i),opts_dict['sample_rate'])
                        except:
                            print('..audio buffer error, skipped audio file generation')

                    

                    # document losses 
                    losspath = os.path.join(opts_dict['outf'], 'losses/')
                    np.save(losspath+'D1',np.array(minibatchLossD1))
                    np.save(losspath+'D2',np.array(minibatchLossD2))
                    np.save(losspath+'G1rec',np.array(minibatchLossG1_rec))
                    np.save(losspath+'G1gan',np.array(minibatchLossG1_gan))
                    np.save(losspath+'G2',np.array(minibatchLossG2))
                
                #netG.train()
                #netE.train()
            # if schedule for learning rate, update lr
            if args.schedule_lr:
                schedulerG.step()
                schedulerD1.step()
                schedulerD2.step()
                             
                
        # do checkpointing of models
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opts_dict['outf'], epoch))
        # saving the discriminators is not necessary, uncomment if you want them
        #torch.save(netD1.state_dict(), '%s/netD1_epoch_%d.pth' % (opts_dict['outf'], epoch))
        #torch.save(netD2.state_dict(), '%s/netD2_epoch_%d.pth' % (opts_dict['outf'], epoch))
        torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opts_dict['outf'], epoch))
        
        # evaluate test error 
        print('\n .... evaluating test loss .... ')
        
        
        # learn a PCA model 
        if opts_dict['do_pca']:
            if len(Xpca) >= opts_dict['npca_samples'] and pca_model is None:
                print('///// learning PCA model /////')
                pca_model = learn_pca_model(Xpca, opts_dict['npca_components'], 
                                            random_state = opts_dict['manualSeed'])
                print('///// PCA model learned, %d components /////'%(pca_model.n_components_))
                del Xpca
                gc.collect()
                joblib.dump({'pca_model':pca_model}, os.path.join(opts_dict['outf'],'pca_model.pkl'))
        
        # test set error 
        test_loss_recon = []
        test_loss_gan = []
        
        with torch.no_grad():
            for k,(data,age) in enumerate(test_dataloader):
                data = data.view(data.size(0),nc,data.size(1),data.size(2)).to(device)
                # map X -> Z
                encoding = netE(data)
                # map Z -> Xhat
                reconstruction = netG(encoding)
                # reconstruction error
                errG_recon = (criterion_dist(reconstruction, data) + \
                          criterion_dist(downsample_pth(reconstruction),downsample_pth(data))) * opts_dict['lambdaa']
                # generate fake image
                noise.normal_(0.,opts_dict['z_var'])
                fake = netG(noise)
                # classify it with D3
                pred_fake_d2 = netD2(fake)
                labell = torch.FloatTensor(opts_dict['batchSize'],1).fill_(1.) # true label
                labell = labell.to(device)
                err_g_d2 = criterion_gan(pred_fake_d2, labell)

                test_loss_recon.append(errG_recon.item())
                test_loss_gan.append(err_g_d2.item())

                
        per_epoch_avg_loss_recon[epoch] = np.mean(np.array(test_loss_recon))
        per_epoch_std_loss_recon[epoch] = np.std(np.array(test_loss_recon))
        per_epoch_avg_loss_gan[epoch] = np.mean(np.array(test_loss_gan))
        per_epoch_std_loss_gan[epoch] = np.std(np.array(test_loss_gan))
        print('[%d/%d] test loss recon: %.2f +/- %.2f , test loss gan: %.2f +/- %.2f'%(epoch, opts_dict['niter'], \
                                                                                      per_epoch_avg_loss_recon[epoch], \
                                                                                      per_epoch_std_loss_recon[epoch], \
                                                                                      per_epoch_avg_loss_gan[epoch], \
                                                                                      per_epoch_std_loss_gan[epoch]))
    
    joblib.dump( {'avg_recon': per_epoch_avg_loss_recon, 'std_recon': per_epoch_std_loss_recon, 
                                     'avg_gan': per_epoch_avg_loss_gan, 'std_gan': per_epoch_std_loss_gan}, losspath+'testloss.pkl')
        
