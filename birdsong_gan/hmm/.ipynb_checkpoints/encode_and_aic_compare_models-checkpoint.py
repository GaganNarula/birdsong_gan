from hmm_utils import *
from hmm_opts import hmm_opts
import pdb
import warnings
warnings.filterwarnings("ignore")


# random state
RNG = 0

def learnKmodels_getbest(data, lastmodel, Lengths, hidden_size, hmm_opts, restarts = 10):
    ''' EM based HMM learning with multiple initializations '''
    models = []
    LL = np.nan * np.zeros(restarts)
    for k in range(restarts):
        try:
            m = learn_single_hmm_gauss_with_initialization(data, lastmodel, Lengths, hidden_size, 
                                                       hmm_opts['covariance_type'], 
                                                       hmm_opts['transmat_prior'], hmm_opts['n_iter'], 
                                                       hmm_opts['tolerance'], hmm_opts['fit_params'],
                                                       hmm_opts['init_cov']) 
            
            # compute train log likelihood
            logl = m.score(data, Lengths)
        except:
            continue
        LL[k] = logl
        models.append(m)
    # choose model with highest LL
    best = np.nanargmax(LL)
    return models[best]



def learn_single_hmm_gauss_with_initialization(data, lastmodel = None, lengths = [], K=10, covtype='spherical',
                                               transmat_prior=1, n_iter=1000, tol = 0.01, fit_params = 'stmc', 
                                               init_cov = 0.1, covars_prior = 1.):
    """ Learn a single model on the list of sequences in data
        If lastmodel is provided, it is used to initialize the parameters of this model. 
        Params
        ------
            data : list of numpy.ndarrays. Each array has shape (timesteps , dimensions)
            lastmodel : hmmlearn.hmm.GaussianHMM model. Learned from previous day.
            lengths : list, lengths of individual sequences in data
            K : int ,hidden state size
            covtype : str, {'spherical','diag','tied','full'} covariance matrix type
            transmat_prior : float, dirichlet concentration prior. Values > 1 lead to more uniform discrete probs
            n_iter : int, maximum number of EM iterations
            tol : float, tolerance for log-likelihood changes. If log-likelihood change < tol, learning is finished.
            fit_params : str, Any combination of 's' (start_prob), 't' (transition matrix), 'm': mean and 
                                'c' : covariance. The specified parameters will be estimated, others remain fixed.
            init_cov : float, initial covariance along diagonal
            covars_prior : float, covariance matrix prior. Same prior over all dimensions. The actual prior is a 
                            diagonal.
    """
    if lastmodel is None:
        model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=RNG, n_iter = n_iter, covars_prior=covars_prior*np.ones(K),params=fit_params, 
                        init_params = 'st', verbose=False, tol=tol)
        # for hmmlearn vesion 0.2.3
        fake_init_data = np.random.multivariate_normal(mean=np.zeros(data.shape[1]),
                                                       cov=init_cov*np.eye(data.shape[1],
                                                                             data.shape[1]), 
                                                       size = data.shape[0])
        model._init(fake_init_data)
        model.fit(data, lengths)
        return model
    model = GaussianHMM(n_components=K, covariance_type=covtype, transmat_prior=transmat_prior, \
                       random_state=0, n_iter = n_iter, covars_weight=covarweight, params=fit_params, 
                        init_params = 'c', verbose=False, tol=tol)
    # initiliaze parameters to last model
    model.transmat_ = lastmodel.transmat_
    model.startprob_ = lastmodel.startprob_
    model.means_ = lastmodel.means_
    model.fit(data, lengths)
    return model
    

    
def load_z_data_and_learn(dataset, lastmodel, idx, hidden_size, netE, netG, outpath, hmm_opts):
    '''
        Loads data for one (merge) day and learns one HMM for all the training sequences
    '''
    
    # get the latent space data
    X = dataset.get(day = idx)
    if len(X) < hidden_size*hmm_opts['min_seq_multiplier'] or len(X) < 250:
        print('\n ### LESS THAN MIN FILES ON DAY %d, SKIPPING! ###\n'%(idx))
        return None
    # encode all 
    Z = [encode(x, netE, transform_sample=False, batch_size=hmm_opts['batchsize'], imageH=129,
           imageW=hmm_opts['imageW'], return_tensor=False, cuda=True) for x in X]
    
    # split into train and validation
    ids = np.random.permutation(len(Z))
    ntrain = int(hmm_opts['train_proportion'] * len(ids))
    ztrain = [Z[ids[i]] for i in range(ntrain)]
    ztest = [Z[ids[i]] for i in range(ntrain, len(ids))]
    
    ids_train = ids[:ntrain]
    ids_test = ids[ntrain:]
    
    # choose some ztrain for saving
    inds_to_save = np.random.choice(len(ztrain), size=hmm_opts['nsamps'])
    
    outputfolder = os.path.join(outpath, 'day_'+str(idx)+'_hiddensize_'+str(hidden_size))
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    # save images of real data
    for i in inds_to_save:
        plt.figure(figsize=(50,10))
        plt.imshow(rescale_spectrogram(X[i]), origin='lower', cmap = 'gray')
        # real sequence
        plt.savefig(os.path.join(outputfolder, 
                             'real_sequence_' + str(i) + '.eps'), dpi = 50, format='eps')
        plt.close()
        
    
    ztosave = [ztrain[i] for i in inds_to_save]
    # get lengths of sequences
    Ltrain = [z.shape[0] for z in ztrain]
    # train HMM 
    ztrain2 = np.concatenate(ztrain, axis=0)
    print('# learning hmm with %d states #'%(hidden_size))
    model = learnKmodels_getbest(ztrain2, lastmodel, Ltrain, hidden_size, hmm_opts, restarts = hmm_opts['n_restarts'])
    
    # compute 2 step full entropy
    print('# computing model entropy #')
    Hsp, Htrans, Hgauss = full_entropy(model)
    
    # compute test log likelihood
    Ltest = [z.shape[0] for z in ztest]
    ztest2 = np.concatenate(ztest, axis=0)
    test_scores = model.score(ztest2, Ltest)
    # compute train log likelihood
    train_scores = model.score(ztrain2, Ltrain)
    
    # create 10 samples
    # concatenate the sequences because otherwise they are usually shorter than batch_size
    ztosave = np.concatenate(ztosave, axis=0)
    create_output(model, outpath, hidden_size, idx, hmm_opts, netG, [], nsamps=hmm_opts['nsamps'])
    
    # save 10 real files
    create_output(model, outpath, hidden_size, idx, hmm_opts, netG, ztosave, nsamps=hmm_opts['nsamps'])
    print('# generated samples #')
    
    # get number of active states etc
    # how many active states were there ? 
    med_active, std_active = number_of_active_states_viterbi(model, ztrain2, Ltrain)
    
    joblib.dump({'model':model, 'train_score':train_scores, 'test_score': test_scores, 'med_active':med_active,
                 'ztrain':ztrain,'ztest':ztest,
                'std_active':std_active, 'ids_train':ids_train,'ids_test':ids_test, 'Lengths_train':Ltrain,
                 'Lengths_test':Ltest, 'Entropies':[Hsp,Htrans,Hgauss], 'opts':hmm_opts},
                os.path.join(outputfolder, 'model_data_and_scores_day_'+str(idx)+'.pkl'))
    return model, test_scores, train_scores, med_active, std_active, Ltrain, Ltest




def train_models(args):
    K = args.hidden_state_size
    # create data object
    dataset = bird_dataset(args.datapath)
    # load encoder
    netE = load_netE(args.netEpath, ngpu = 1, nz = hmm_opts['nz'], ngf = hmm_opts['ngf'],
                     nc = hmm_opts['nc'], cuda = True)
    # load generator
    netG = load_netG(args.netGpath, ngpu = 1, nz = hmm_opts['nz'], ngf = hmm_opts['ngf'],
                     nc = hmm_opts['nc'], cuda = True)

    Ndays = dataset.nfolders
    if args.last_day == -1:
        last_day = Ndays
    else:
        last_day = args.last_day
    # train models
    print('\n ..... training HMMs ..... \n')
    results = [None for _ in range(Ndays)]
    # loop over days
    for k in range(args.start_from, last_day+1):
        # loop over hidden state sizes
        results[k] = [None for _ in range(len(K))]
        for j in range(len(K)):
            start = time()
            if not args.do_chaining:
                # check if this model already exists
                outpath = os.path.join(args.outpath, 'day_'+str(k)+'_hiddensize_'+str(K[j]),
                                       'model_data_and_scores_day_'+str(k)+'.pkl')
                if not os.path.exists(outpath):
                    results[k][j] = load_z_data_and_learn(dataset, None, k, K[j], netE, netG, 
                                           args.outpath,  hmm_opts)
            else:
                if k > args.start_from and results[k-1][j] is not None:
                    results[k][j] = load_z_data_and_learn(dataset, results[k-1][j][0], k, K[j], netE, netG, 
                                           args.outpath,  hmm_opts)
                else:
                    results[k][j] = load_z_data_and_learn(dataset, None, k, K[j], netE, netG, 
                                           args.outpath,  hmm_opts)
            
            end = time()
            if results[k][j]:
                print('..... day %d, hidden state size %d, train lls (avg over steps): %.4f, test lls (avg steps): %.4f .....' %(k, K[j],
                                                                                                      results[k][j][2] / results[k][j][5],
                                                                                                      results[k][j][1] / results[k][j][6]))
                print('..... avg number of active states %.2f, std dev %.2f .....'%(results[k][j][3], results[k][j][4]))
                print('..... time taken %.1f secs ..... \n\n'%(end-start))
    
    print('\n ###### finished training! #######')
    return results



        

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type = str)
parser.add_argument('--netGpath', type = str, help = 'path to generator neural network')
parser.add_argument('--netEpath', type = str, help = 'path to encoder neural network')
parser.add_argument('--outpath', type = str, help = 'where to save models and samples')
parser.add_argument('--do_chaining', action = 'store_true')
parser.add_argument('--hidden_state_size', type = int, nargs = '+', default = [5, 10, 15, 20, 30, 50, 75, 100])
parser.add_argument('--nz', type = int, default = 16, help = 'latent space dimensions')
parser.add_argument('--covariance_type', type = str, default = 'spherical')
parser.add_argument('--covars_prior', type = float, default = 1., help ='diagnoal term weight on the prior covariance')
parser.add_argument('--fit_params', type = str, default = 'stmc', help = 'which parameters to fit, s = startprob, t = transmat, m = means, c = covariances')
parser.add_argument('--transmat_prior', type = float, default = 1., help = 'transition matrix prior concentration')
parser.add_argument('--n_iter', type = int, default = 400, help = 'number of EM iterations')
parser.add_argument('--tolerance', type = float, default = 0.01)
parser.add_argument('--get_audio', action = 'store_true', help = 'generate audio files as well')
parser.add_argument('--start_from', type = int, default = 0, help = 'start day of learning') 
parser.add_argument('--last_day', type = int, default = -1, help = 'last day of learning')
parser.add_argument('--min_seq_multiplier', type = int ,default = 3, help='the number of files should be at least hidden size x this factor')


if __name__ == '__main__':
    args = parser.parse_args()
    op  = vars(args)
    for k,v in op.items():
        if k in hmm_opts.keys():
            hmm_opts[k] = v
    results = train_models(args)

