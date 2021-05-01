hmm_opts = {'hidden_state_size' : [5, 10, 15, 20, 30, 50, 75, 100],
           'covariance_type' : 'spherical', 
           'fit_params' : 'stmc',
           'transmat_prior' : 1.,
           'n_iter' : 300,
           'tolerance' : 0.01,
           'nz' : 16, 
           'ngf' : 264,
           'nc' : 1,
            'covars_prior' : 1.,
            'init_params' : 'stmc',
            'imageH': 129,
           'imageW': 16,
            'noverlap' : 0,
           'batchsize' : 2,
            'train_proportion' : 0.7,
           'nsamplesteps' : 128,
           'nsamps': 10,
           'sample_var': 0.,
            'sample_invtemperature' : 1.,
            'munge_len' : 50,
            'n_restarts': 10,
            'do_chaining': False,
            'min_seq_multiplier': 10,
            'cuda' : True,
            'hmm_random_state' : 0,
            'last_day': -1,
            'start_from' : 0,
            'datapath' : '',
            'netEpath' : '',
            'netGpath' : '',
            'outpath' : '',
            'resnet' : False,
            'save_output' : True,
            'get_audio': False
           }