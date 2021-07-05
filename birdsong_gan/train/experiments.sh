#!/bin/bash

train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_2_residual_uniform/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 2 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --noise_dist 'uniform' --z_var 10. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_4_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 4 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual 

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_6_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 6 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_8_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 8 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_32_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 32 --niter 12 --ngf 128 --ndf 128  --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual

%run train.py --training_path '/media/songbird/datapartition/id_list_train.pkl' --test_path '/media/songbird/datapartition/id_list_test.pkl' --outf '/media/songbird/datapartition/mdgan_output/nz_64_residual/' --subset_age_weights 0. 1. --batchSize 64 --lr 2e-5 --nz 64 --niter 12 --ngf 128 --ndf 128 --manualSeed 800 --lambdaa 100 --z_var 1. --z_reg --log_every 1000 --cuda --train_residual