from typing import List, Dict, Tuple, Union, Callable
import torch
import numpy as np
from datasets import load_from_disk, concatenate_datasets, Dataset


class BirdsongHFDataset:
    """A class to load a songbird huggingface dataset from disk. Some utilities for functions
    like:
        - resampling data by age
        - class weights according to given label
        - can be used directly in training

    Input dataset has following columns:
        - bird_name (str)
        - tutor_name (str)
        - filename (str)
        - recording_date (str) either a date in YYYY-m-d format or a string = 'tutor'
        - days_post_hatch (int)
        - tutored (bool) whether the bird was tutored on that day
        - tutoring_start_dph (int) days post hatch when tutoring began
        - sample_rate (int)
        - audio (1D array[float])
        - spectrogram (2D array[float])
        - n_fft (int)
        - win_length (int)
        - hop_length (int)
        
    Usage:

    ```python
        # load from disk
        ds = BirdsongHFDataset(path_to_dataset=/your/dataset/folder)
        # concatenate two datasets
        new_ds = ds.append(other_ds)
        # resample across age and make new dataset
        new_ds = ds.get_age_resample_dataset(max_samples_per_age=50)
        # get only one bird's dataset
        bird_ds = ds.get_single_bird_dataset("bird_x")
    ```

    Args:
        dataset (Dataset, optional): A huggingface dataset. Defaults to None.
        path_to_dataset (str, optional): Path to a huggingface dataset. Defaults to None.
        storage_options (Dict, optional): Storage options for the dataset. Defaults to None.
        label_column (str): The column to use as the label. Defaults to "days_post_hatch".
        feature_column (str): The column to use as the input for training. Defaults to "spectrogram".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    """

    def __init__(self, dataset: Dataset = None,
                 path_to_dataset: str = None,
                 storage_options: Dict = None,
                 label_column: str = "days_post_hatch",
                 feature_column: str = "spectrogram",  # or 'audio'
                 generate_labels: bool = True,
                 verbose: bool = True,
                 ):

        if dataset is None:
            self.ds = load_from_disk(
                path_to_dataset, storage_options=storage_options)
        else:
            self.ds = dataset
        # sets label_column as the ClassLabel
        # self.ds.class_encode_column(label_column)
        self.class_counts = None
        self.class_weights = None
        # label names are by default string reps of label column unique values
        self.label_column = label_column
        self.feature_column = feature_column
        self.label_names = None
        self.generate_labels = generate_labels
        self.verbose = verbose
        if self.generate_labels:
            self._init_weights()

    def _init_weights(self):
        self.label_names, self.class_counts = self.compute_class_counts()
        self.class_weights = self.compute_class_weights()

    def compute_class_counts(self) -> Tuple[List[str], np.ndarray]:
        """Computes the class counts for each class in the dataset."""
        # returns sorted values and respective counts
        vals, counts = np.unique(
            self.ds[self.label_column], return_counts=True)
        order = np.argsort(vals)
        return [str(vals[i]) for i in order], counts[order]

    def compute_class_weights(self) -> torch.Tensor:
        """Computes the class weights for each class in the dataset."""
        if self.class_counts is None:
            self.class_counts = self.compute_class_counts()
        class_counts = torch.from_numpy(self.class_counts).to(torch.float32)
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        return class_weights

    def num_unique_user_ids(self) -> int:
        return len(self.ds.unique("uuid"))

    def get_number_of_samples_per_user(self) -> Dict:
        """Get a dictionary with the number of samples per user"""
        all_user_ids = np.array(self.ds["uuid"])
        unique_user_ids, user_counts = np.unique(
            all_user_ids, return_counts=True)
        return {k: v for k, v in zip(unique_user_ids, user_counts)}

    @property
    def columns(self):
        return self.ds.column_names

    @property
    def features(self):
        return self.ds.features

    def unique(self, column: str):
        return self.ds.unique(column)

    def add_column(self, name: str, data: Union[List, np.ndarray]):
        self.ds = self.ds.add_column(name=name, column=data)

    def map(self, func: Callable, **ds_map_kwargs):
        self.ds = self.ds.map(func, **ds_map_kwargs)

    def get_ds_items(self, items: Union[str, int]):
        """Get an item as dict from the huggingface dataset."""
        return self.ds[items]

    def sample_ds(self, size: int):
        """Sample a random subset of the dataset."""
        sample_ids = np.random.choice(len(self), size=size, replace=False)
        return self.ds[sample_ids]

    def sample(self, size: int) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample a random subset of the dataset and return a list of tensors."""
        sample_ids = list(np.random.choice(
            len(self), size=size, replace=False))
        return [self[int(idx)] for idx in sample_ids]

    def sample_and_collate(self, size: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a random subset of the dataset and return a tensor or a tuple of tensors."""
        batch = self.sample(size)
        if self.generate_labels:
            y = torch.stack([b[0] for b in batch])
            x = torch.stack([b[1] for b in batch])
            return y, x
        return torch.stack(batch)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        example = self.ds[index]
        inputs = np.array(example[self.feature_column])
        inputs = torch.from_numpy(inputs).float()
        if self.generate_labels:
            label = example[self.label_column]
            label = torch.FloatTensor([label])
            return label, inputs
        return inputs

    def get_age_resampled_indices(self, max_samples_per_age: int = np.inf,
                                   min_samples_per_age: int = 1) -> List[int]:
        """Get a set of indices that resamples the dataset to have a maximum number of samples per age 
        and a minimum number of samples per age. Here age = "days_post_hatch"

        Args:
            max_samples_per_age (int, optional): The maximum number of samples per user. Defaults to np.inf.
            min_samples_per_age (int, optional): The minimum number of samples per user. Defaults to 1.

        Returns:
            List[int]: A list of indices that resamples the dataset.
        """
        all_dph = np.array(self.ds["days_post_hatch"])
        unique_age_ids, age_counts = np.unique(all_dph, return_counts=True)
        # find counts per user id
        if self.verbose:
            print(
                f"Resampling dataset. New max samples per age: {max_samples_per_age}, new min samples per age: {min_samples_per_age}")
            print("Number of unique age: ", len(unique_age_ids))
            print("Max samples per age: ", np.max(age_counts))
            print("Min samples per age: ", np.min(age_counts))
            print("Mean samples per age: ", np.mean(age_counts))
            print("Median samples per age: ", np.median(age_counts))
            # how many ages make up 50% of the data?
            cdf = np.cumsum(age_counts) / np.sum(age_counts)
            print("Number of ages that make up 50% of the data: ",
                  np.sum(cdf < 0.5))
        # now resample the dataset
        new_index = []
        for _, (age_id, cnt) in enumerate(zip(unique_age_ids, age_counts)):
            # get all indices for this age_id
            age_indices = np.where(all_dph == age_id)[0]
            # if the number of samples is larger than counts[user], we need to resample
            if cnt > max_samples_per_age:
                new_age_indices = list(np.random.choice(
                    age_indices, size=max_samples_per_age, replace=False))
                new_index.extend(new_age_indices)
            elif min_samples_per_age <= cnt <= max_samples_per_age:
                new_index.extend(age_indices)
        return new_index

    def subset(self, indices):
        """Return a subset of this dataset at the given indices"""
        new_ds = self.ds.select(indices)
        return BirdsongHFDataset(dataset=new_ds, 
                                 verbose=self.verbose, 
                                 label_column=self.label_column,
                                 feature_column=self.feature_column,
                                 generate_labels=self.generate_labels
                                 )

    def get_age_resampled_dataset(self, new_indices: List[int] = None,
                                   max_samples_per_age: int = np.inf,
                                   min_samples_per_age: int = 1):
        """Resample the dataset to select only those ages with number of samples >= min_samples_per_age
            and <= max_samples_per_age

        Args:
            new_indices (List[int], optional): If provided, use these indices to select the new dataset. Defaults to None.
            max_samples_per_age (int, optional): Maximum number of samples per age. Defaults to np.inf.
            min_samples_per_age (int, optional): Minimum number of samples per age. Defaults to 1.
        """
        if new_indices is None:
            new_indices = self.get_age_resampled_indices(
                max_samples_per_age, min_samples_per_age)
        return self.subset(new_indices)

    def get_single_bird_dataset(self, bird_name: str):
        all_bird_ids = np.array(self.ds["bird_name"])
        new_indices = np.where(all_bird_ids == bird_name)[0]
        return self.subset(new_indices)

    def get_list_of_birds_dataset(self, bird_names: List[str]):
        all_bird_ids = np.array(self.ds["bird_name"])
        new_indices = np.where(np.isin(all_bird_ids, bird_names))[0]
        return self.subset(new_indices)

    def save_to_disk(self, path: str):
        self.ds.save_to_disk(path)

    def rename_column(self, old_column: str, new_column: str):
        self.ds = self.ds.rename_column(old_column, new_column)
        
    def append_dataset(self, other_dataset):
        new_ds = concatenate_datasets([self.ds, other_dataset.ds])
        return BirdsongHFDataset(dataset=new_ds, 
                                 verbose=self.verbose, 
                                 label_column=self.label_column,
                                 feature_column=self.feature_column,
                                 generate_labels=self.generate_labels
                                 )
    def __repr__(self):
        return self.ds.__repr__()