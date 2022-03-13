#!/bin/bash

train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_2_residual_uniform/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 2 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --noise_dist 'uniform' --z_var 10. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_4_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 4 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual 

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_6_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 6 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_8_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 8 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_8_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 8 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_32_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 32 --niter 12 --ngf 128 --ndf 128  --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_64_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 64 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual


python train_larger_with_FID.py --training_path ~/datapartition/id_list_train_single_file.pkl --test_path ~/datapartition/id_list_test_single_file.pkl --outf ~/datapartition/mdgan_output/recurrent_gan/nz_16 --path2hdf ~/datapartition/all_birds.hdf --batchSize 128 --lr 5e-5 --nz 12 --ngf 128 --ndf 128 --cuda --train_residual --z_reg --log_every 500 --niter 30 --do_pca --npca_components 128



python train_recurrent_gan.py --training_path ~/datapartition/id_list_train_single_file.pkl --test_path ~/datapartition/id_list_test_single_file.pkl --outf ~/datapartition/mdgan_output/recurrent_gan/nz_16  --path2hdf ~/datapartition/all_birds.hdf --batch_size 64 --lr 5e-5 --max_length 120 --nrnn 100 --nlin 100 --ngf 64 --cuda --nepochs 60 --log_every 200  