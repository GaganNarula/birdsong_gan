### Birdsong Generative Models for Sequence Learning

Songbirds, like humans, are [vocal learners](https://en.wikipedia.org/wiki/Vocal_learning). Young male zebra finches, a well-studied songbird species, use the song of an adult male bird as a template to adapt their initial “babbling-like” variable vocalizations into a temporally and acoustically structured form. The process of song learning comes under the umbrella of motor learning. Briefly, the problem faced by the bird is the following: produce appropriate control inputs from a control algorithm that feed into a (generally) nonlinear dynamical system that creates desired motor output. 

To better understand the algorithmic process behind song learning, we wanted to first build a generative model for birdsong. We break the process of song learning into components, and two important one's here.
* Decoder: a function that maps vectors in a 32 dimensioinal space to birdsong spectrogram snippets in 129 x 16 dimensional space.
* Sequence model: a model of the system that controls the decoder. In our case, we use simple Hidden Markov Models.

The interesting thing is, this two component structure has a very close neural implementation as the feedforward network formed by the songbird song control nuclei (i.e. aggregations of neurons) called the *HVC* and *RA*.

## Description
This repository contains Pytorch code for a generative adversarial network that generates short (129 frequency x 16 time frame) snippets of birdsong spectrogram, and hidden markov models to generate them (using the package [hmmlearn] (https://hmmlearn.readthedocs.io/en/latest/)). The generator is jointly trained with an encoder and a discriminator, thus it is an autoencoder trained with a mixed loss function.

We use the generator in combination  with Hidden Markov Models for sequence learning on the snippets.

# Generating birdsong spectrogram snippets
Here is a simple scheme showing what the generator does. A 32-dimensional latent vector (sampled from standard normal distribution) is fed into an upsampling convolutional network.
![generator](https://github.com/GaganNarula/BirdsongGAN_sequencelearning/blob/master/images/Generative%20models%20of%20sequence%20learning%20in%20birdsong.002.jpeg)

# Learning sequences of encoded spectrogram snippets
We use a simultaneously trained encoder to encode real birdsong into the generators latent space. The combined generator + encoder + discriminator network trio is trained with a custom loss function that contains the original GAN loss (i.e. generator and encoder trained to fool the discriminator) as well as a minimization of reconstruction error. We follow the procedure outlined in the paper Mode Regularized GAN's ([Chen et al 2016](https://arxiv.org/abs/1612.02136))
![network](https://github.com/GaganNarula/BirdsongGAN_sequencelearning/blob/master/images/Generative%20models%20of%20sequence%20learning%20in%20birdsong.003.jpeg)

To learn a model of encoded sequences, we use Hidden Markov Models. Thus, the encoded latent z-vectors are treated as observed variables generated a higher level latent state $h$. 
![hmm](https://github.com/GaganNarula/BirdsongGAN_sequencelearning/blob/master/images/Generative%20models%20of%20sequence%20learning%20in%20birdsong.005.jpeg)

# Reconstructions
The generator-encoder pair (autoencoder) does a good job reconstructing real spectrograms, much better than PCA. However, it is more difficult to model the very early noisy babblings of the young zebra finch. The older the bird gets, the easier it becomes to generate (or reconstruct) good quality spectrograms.
![example](https://github.com/GaganNarula/BirdsongGAN_sequencelearning/blob/master/images/Generative%20models%20of%20sequence%20learning%20in%20birdsong.004.jpeg)

# HMM-GAN samples
After learning hidden markov models separately for each day of learning, the system produces very realistic sequences of spectrograms! :-)
![sample](https://github.com/GaganNarula/BirdsongGAN_sequencelearning/blob/master/images/Generative%20models%20of%20sequence%20learning%20in%20birdsong.006.jpeg)
