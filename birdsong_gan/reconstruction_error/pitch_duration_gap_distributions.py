import numpy as np
import torch
import librosa as lc
from birdsong_gan.data.dataset import bird_dataset_single_hdf
from birdsong_gan.models.generative_model import GaussHMMGAN
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import os
from os.path import join
import joblib
import argparse
import pdb
from time import time


def get_pitch_and_vocal_segments(audio, sr=16000,fmin=100.,fmax=8000., frame_length=512,
                                 win_length=256, hop_length=128, switch_prob=0.1, fill_na=np.nan):
    # fundamental frequency f0 and voiced frames computation
    f0, voiced_frames, _ = lc.pyin(audio, sr=sr, fmin=fmin, fmax=fmax, frame_length=frame_length,
                                   win_length=win_length, hop_length=hop_length, switch_prob=switch_prob, fill_na=fill_na)
    return f0, voiced_frames
                                 
                                 
                                 
def get_durations_from_vocal_segments(voiced_frames, min_dur = 3):
    """Computes durations and gap lengths from voiced frames. Ignore
    durations less than min_dur frames.
    """
    onsets = np.where(np.diff(voiced_frames) > 0.)[0]
    offsets = np.where(np.diff(voiced_frames) < 0.)[0]
    
    if len(onsets)==0 or len(offsets)==0:
        return None, None, None, None, None
    
    ON = onsets
    OFF = offsets
    
    # fix if lengths off by 1
    if np.abs(len(onsets)-len(offsets)) == 1:
        if offsets[0] < onsets[0]:
            OFF = offsets[1:]
        elif onsets[-1] > offsets[-1]:
            ON = onsets[:-1]
            
    vocal_segs = []
    durations = []
    gap_lengths = []
    gap_start = 0
    to_remove = []
    
    if len(ON)==0 or len(OFF)!=len(ON):
        return None,None,None,None,None
    
    for k in range(len(OFF)):
        
        if OFF[k] - ON[k] >= min_dur:
            vocal_segs.append(voiced_frames[ON[k] : OFF[k]])
            durations.append(OFF[k] - ON[k])
            
        gap_lengths.append(ON[k]-gap_start)
        gap_start = OFF[k]
        
    return vocal_segs, durations, gap_lengths, onsets, offsets



def get_pitch_duration_gaps(audio_waveforms, pitch_kwargs, min_vocal_duration=3, log_every=10):
    """Compute pitch (Hz), duration (sec) and gap (sec) for a list of audio_waveform arrays.
    """
    pitches = []
    durations = []
    gap_lengths = []
    for i, audio in enumerate(audio_waveforms):
        
        f0, voiced_frames = get_pitch_and_vocal_segments(audio, **pitch_kwargs)
        
        vframes = np.ones(len(voiced_frames))
        vframes[voiced_frames==False] = 0.
        
        if len(voiced_frames == 1.) < min_vocal_duration:
            continue 
            
        _, durs, gaps, _, _ = get_durations_from_vocal_segments(vframes, min_vocal_duration)
    
        if durs is None:
            continue
            
        durs = np.array([lc.frames_to_time(d, sr=pitch_kwargs['sr'], hop_length=pitch_kwargs['hop_length']) for d in durs])
        gaps = np.array([lc.frames_to_time(g, sr=pitch_kwargs['sr'], hop_length=pitch_kwargs['hop_length']) for g in gaps])
        
        durations.append(durs)
        gap_lengths.append(gaps)
        pitches.append(f0[np.isnan(f0)==False])
        
        if i%log_every==0:
            print(f"Done with {i}/{len(audio_waveforms)}")
            
    return pitches, durations, gap_lengths



def main(args):
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    print("Making model")
    
    # load the first model
    path_to_hmm = join(args.path_to_hmm_models, "day_" + str(args.start_day),
                       "hmm_hiddensize_10", "model_day_" + str(args.start_day) + ".pkl")
    model = GaussHMMGAN(netGpath=args.path_to_netG,
                        netEpath=args.path_to_netE,
                        hmmpath=path_to_hmm,
                        ngf=args.ngf,
                        cuda_device=args.device)
    
    model.netG = model.netG.to(args.device)
    model.netE = model.netE.to(args.device)
    
    print("Creating dataset")
    dataset = bird_dataset_single_hdf(args.path_to_bird_hdf, args.birdname)
    
    num_days = dataset.ndays
    print("Total %d days"%(num_days))
    all_pitch_distances = []
    all_duration_distances = []
    all_gap_distances = []
    
    for d in range(args.start_day, num_days):
        
        # load the hmm and replace the older one
        hmm_file_path = join(args.path_to_hmm_models, "day_" + str(d),
                               "hmm_hiddensize_" + str(args.hidden_state_size),
                               "model_day_" + str(d) + ".pkl")
        
        if not os.path.exists(hmm_file_path):
            print(f"\nDay {d} hmm model missing, skipping ....")
            continue
            
        hmm = joblib.load(hmm_file_path)
        model.hmm = hmm["model"]
        
        # get real spectrograms
        Xreal = dataset.get(day=d, nsamps=-1)
        print(f"\nDay {d}, {len(Xreal)} sequences, generating samples ...")
        
        # get sample  spectrograms
        t0 = time()
        Xsamp = model.sample(nsamples=len(Xreal), timesteps=[x.shape[1]//16 for x in Xreal], cuda=True)
        t1 = time()
        
        print("Generating real and fake audio.....")
        t0 = time()
        real_audios = model.generate_audio(Xreal)
        sample_audios = model.generate_audio(Xsamp)
        t1 = time()
        print(f"Done generating audio, took {t1-t0} secs")
        
        print("Calculating pitch, duration and gap lengths")
        pitches_real, durations_real, gap_lengths_real = get_pitch_duration_gaps(real_audios, dict(sr=16000,fmin=100.,fmax=8000.,
                                                                                          frame_length=512, win_length=256,
                                                                                          hop_length=128, switch_prob=0.1,
                                                                                          fill_na=np.nan), min_vocal_duration=3,
                                                                                log_every=100)
        
        
        pitches_samp, durations_samp, gap_lengths_samp = get_pitch_duration_gaps(sample_audios,
                                                                                 dict(sr=16000,fmin=100.,fmax=8000.,
                                                                                      frame_length=512, win_length=256,
                                                                                      hop_length=128, switch_prob=0.1,
                                                                                      fill_na=np.nan),
                                                                                 min_vocal_duration=3,
                                                                                 log_every=100)
        
        pitches_real = np.concatenate(pitches_real)
        durations_real = np.concatenate(durations_real)
        gap_lengths_real = np.concatenate(gap_lengths_real)
        pitches_samp = np.concatenate(pitches_samp)
        durations_samp = np.concatenate(durations_samp)
        gap_lengths_samp = np.concatenate(gap_lengths_samp)
        
        # compute wasserstein and energy distance between pitch, duration and gap durations
        pitch_distance = wasserstein_distance(pitches_real, pitches_samp)
        duration_distance = wasserstein_distance(durations_real, durations_samp)
        gap_distance = wasserstein_distance(gap_lengths_real, gap_lengths_samp)
        
        
        plt.figure(figsize=(15,6))
        plt.hist(pitches_real, bins=args.num_histogram_bins)
        plt.hist(pitches_samp, bins=args.num_histogram_bins, alpha=0.5)
        plt.xlim([0, 2000])
        plt.xlabel("Pitch (Hz)")
        plt.ylabel("Count")
        plt.savefig(join(args.save_path, "pitch_comparison_day_" + str(d) + ".png"), dpi=100, format="png")
        plt.close()
        
        plt.figure(figsize=(15,6))
        plt.hist(durations_real, bins=args.num_histogram_bins)
        plt.hist(durations_samp, bins=args.num_histogram_bins, alpha=0.5)
        plt.xlim([0, 0.5])
        plt.xlabel("Syllable duration (sec)")
        plt.ylabel("Count")
        plt.savefig(join(args.save_path, "duration_comparison_day_" + str(d) + ".png"), dpi=100, format="png")
        plt.close()
        
        plt.figure(figsize=(15,6))
        plt.hist(gap_lengths_real, bins=args.num_histogram_bins)
        plt.hist(gap_lengths_samp, bins=args.num_histogram_bins, alpha=0.5)
        plt.xlim([0, 0.5])
        plt.xlabel("Gap duration (sec)")
        plt.ylabel("Count")
        plt.savefig(join(args.save_path, "gap_lengths_comparison_day_" + str(d) + ".png"), dpi=100, format="png")
        plt.close()
        
        joblib.dump(dict(pitches_real=pitches_real,
                         durations_real=durations_real,
                         gap_lengths_real=gap_lengths_real,
                         pitches_samp=pitches_samp,
                         durations_samp=durations_samp,
                         gap_lengths_samp=gap_lengths_samp,
                         pitch_distance=pitch_distance,
                         gap_distance=gap_distance,
                         duration_distance=duration_distance
                         ),
                    join(args.save_path,
                         "pitch_durations_gaps_day_" + str(d) + "_hiddensize_" + str(args.hidden_state_size) + ".pkl"))
        
        all_pitch_distances.append(pitch_distance)
        all_gap_distances.append(gap_distance)
        all_duration_distances.append(duration_distance)
        
    
    # Plot distances
    plt.figure(10,5)
    plt.plot(np.arange(args.start_day, num_days), np.array(all_pitch_distances), "-ok")
    plt.xlabel("Experiment day")
    plt.ylabel("Wasser L1 between real and fake pitch")
    plt.savefig(join(args.save_path, "pitch_wasserstein_distance.png"), dpi=100, format="png")
    plt.close()
    
    plt.figure(10,5)
    plt.plot(np.arange(args.start_day, num_days), np.array(all_duration_distances), "-ok")
    plt.xlabel("Experiment day")
    plt.ylabel("Wasser L1 between real and fake duration")
    plt.savefig(join(args.save_path, "duration_wasserstein_distance.png"), dpi=100, format="png")
    plt.close()
    
    plt.figure(10,5)
    plt.plot(np.arange(args.start_day, num_days), np.array(all_gap_distances), "-ok")
    plt.xlabel("Experiment day")
    plt.ylabel("Wasser L1 between real and fake gap lengths")
    plt.savefig(join(args.save_path, "gap_wasserstein_distance.png"), dpi=100, format="png")
    plt.close()
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_bird_hdf", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--birdname", type=str, required=True)
    parser.add_argument("--path_to_netG", type=str, required=True)
    parser.add_argument("--path_to_netE", type=str, required=True)
    parser.add_argument("--path_to_hmm_models", type=str)
    parser.add_argument("--hidden_state_size", type=int)
    parser.add_argument("--ngf", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_day", type=int, default=0)
    parser.add_argument("--num_histogram_bins", type=int, default=100)
    args = parser.parse_args()
    
    main(args)