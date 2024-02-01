f"""Huggingface style dataset

Each record in the dataset should have the following structure:

1. bird_name
2. recording_date
3. days_post_hatch
4. filename
5. audio  # array of floats, downsampled / no resampling
6. spectrogram  # magnitude part only, array of floats
7. nfft
8. window_length # samples
9. window_overlap
10. sample_rate
11. tutor_name
12. tutoring_start_date
13. tutored  # bool

"""
import os
import numpy as np
from datasets import Dataset, concatenate_datasets
import shutil
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import librosa
import glob
import asyncio
import datetime


birds = ["b3g20", "b4g20", "b6r17", "b7r16", "b8r17", "b14r16", "b13r16", "g7r15",
         "g19r15", "g20r15", "k3r16", "k6r16", "p3r16", "p20r16", "r15s12", "r15y2", "r15y5"]


tutors = ["g1p2", "g10o15", "b12r14", "b13r14", "p7r13", "r14y2", "b12r14", "b12p14",
          "r14y2", "b13r14", "p7r13", "b9r14", "p7r13", "p7r13", "b12r14", "b12r14", "p7r13"]

start_dph = [45, 38, 37, 39, 38, 37, 38, 39,
             38, 39, 44, 45, 39, 48, 40, 37, 37]

tutoring_start_date = [47, 47, 51, 47, 51, 47, 47, 38,
                       39, 38, 47, 47, 44, 46, 38, 43, 41]

bird_data = { k: {"bird_name": k, "tutor_name": t, "start_dph": d, "tutoring_start_dph": x} for (k, t, d, x) in zip(birds, tutors, start_dph, tutoring_start_date)}

COLUMNS = ["bird_name", "recording_date", "days_post_hatch", "filename", "audio", "spectrogram", "sample_rate",
           "n_fft", "win_length", "hop_length", "tutor_name", "tutoring_start_dph", "tutored"]


@dataclass
class BirdsongHFDatasetBuilder:

    path_to_all_birds: str
    output_dir: str
    bird_metadata: dict
    original_sample_rate: int = 32000
    target_sample_rate: int = 16000
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 128
    batch_write_size: int = 2000
    log_every: int = 500
    folder_stepin_1: str = "SAP"
    folder_stepin_2: str = "songs"
    extention: str = "wav"
    columns: List[str] = field(default_factory=lambda: COLUMNS)
    
    def __post_init__(self):
        self.results = {k: [] for k in self.columns}
        self.curr_num_samples = 0
        self._tmp_first_date = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def _write_batch(self, batch_number: int):
        # make a hugginface Dataset
        dataset = Dataset.from_dict(self.results)
        # save dataset to disk
        dataset.save_to_disk(os.path.join(self.output_dir,
                                  f"dataset_{batch_number}"))
        # reset results
        self.results = {k: [] for k in self.columns}

    def _load_and_concatenate_datasets(self):
        # load all datasets
        datasets = []
        for i in range(self.num_batches):
            dataset = Dataset.load_from_disk(os.path.join(self.output_dir, f"dataset_{i}"))
            datasets.append(dataset)
        # concatenate datasets
        dataset = concatenate_datasets(datasets)
        # save dataset to disk
        print(f"..... Saving dataset to disk .....")
        dataset.save_to_disk(self.output_dir)
        # remove all temporary datasets
        for i in range(self.num_batches):
            shutil.rmtree(os.path.join(self.output_dir, f"dataset_{i}"))
        return dataset

    async def _create_sample(self, path_to_file: str, 
                       recording_date: str, 
                       dph: int,
                       bird_data: Dict) -> None:
        # load the file
        y, _ = librosa.load(path_to_file, sr=self.target_sample_rate)
        
        # create spectrogram
        S = librosa.stft(y, n_fft=self.n_fft, 
                         win_length=self.win_length,
                         hop_length=self.hop_length)
        S = np.abs(S)
        if "tutor" in recording_date:
            # this is tutor bird
            bird_name = bird_data["bird_name"] + "_tutor"
        else:
            bird_name = bird_data["bird_name"]
        self.results["bird_name"].append(bird_name)
        self.results["recording_date"].append(recording_date)
        self.results["days_post_hatch"].append(dph)
        self.results["filename"].append(os.path.basename(path_to_file))
        self.results["audio"].append(y)
        self.results["spectrogram"].append(S)
        self.results["sample_rate"].append(self.target_sample_rate)
        self.results["n_fft"].append(self.n_fft)
        self.results["win_length"].append(self.win_length)
        self.results["hop_length"].append(self.hop_length)
        self.results["tutor_name"].append(bird_data["tutor_name"])
        self.results["tutoring_start_dph"].append(bird_data["tutoring_start_dph"])
        self.results["tutored"].append(dph >= bird_data["tutoring_start_dph"])
        self.curr_num_samples += 1
        
    async def _create_samples(self, folder_path: str, index: int, bird_data: Dict):
        # list all files
        recording_date = os.path.basename(folder_path)
        if "tutor" in recording_date:
            dph = -1
        else:
            # convert to datetime first
            date = datetime.datetime.strptime(recording_date, "%Y-%m-%d")
            if index == 0:
                dph = bird_data["start_dph"]
                self._tmp_first_date = date
            else:
                # calculate dph by difference to tmp first date
                dph = bird_data["start_dph"] + (date - self._tmp_first_date).days

        all_file_paths = glob.glob(os.path.join(folder_path, self.folder_stepin_2, "*" + self.extention))

        for f in all_file_paths:
            await self._create_sample(f, recording_date, dph, bird_data) 
            
            if (self.curr_num_samples + 1) % self.log_every == 0:
                    print(f'..... {self.curr_num_samples} samples created .....')
            # write a batch of samples to disk
            if len(self.results["bird_name"]) >= self.batch_write_size:
                self._write_batch(self.num_batches)
                self.num_batches += 1
                print(f'### {self.num_batches} batches written .....#####')
            
    def _list_folder_paths(self, bird_name: str):
        day_folders = sorted(glob.glob(os.path.join(self.path_to_all_birds, bird_name, self.folder_stepin_1, "20*")))
        tutor_folder = glob.glob(os.path.join(self.path_to_all_birds, bird_name, self.folder_stepin_1, "tutor"))[0]
        return day_folders, tutor_folder

    async def _make_files(self, day_folders: str, tutor_folder: str, bird_data: Dict) -> None:
        for i, folder_path in enumerate(day_folders):
            await self._create_samples(folder_path, i, bird_data)
        # await asyncio.gather(*(self._create_samples(day_folders[i], i, bird_data) for i in range(len(day_folders))))
        await self._create_samples(tutor_folder, -1, bird_data)
        
    async def build_dataset(self):

        birds = list(self.bird_metadata.keys())
        self.num_batches = 0  # keep track of the number of batches written
        for j, bird in enumerate(birds):
            
            print(f"\nProcessing birds {bird}\n\n")
            bird_data = self.bird_metadata[bird]
            day_folders, tutor_folder = self._list_folder_paths(bird)
            print(f"Found {len(day_folders)} days for this bird")
            await self._make_files(day_folders, tutor_folder, bird_data)
            
        # write the last batch
        self._write_batch(self.num_batches)
        # concatenate all batches into a single dataset
        return self._load_and_concatenate_datasets()


async def main():
    builder = BirdsongHFDatasetBuilder(path_to_all_birds="/media/gagan/Gagan_external/songbird_data/WAV_files",
                                    output_dir="/media/gagan/Gagan_external/songbird_data/hf_dataset",
                                    bird_metadata=bird_data,
                                    original_sample_rate=32000,
                                    target_sample_rate=16000,
                                    n_fft=256,
                                    win_length=256,
                                    hop_length=128,
                                    batch_write_size=40000,
                                    log_every=500)
    await builder.build_dataset()  


if __name__ == "__main__":
    asyncio.run(main())