import numpy as np
from datasets import load_from_disk, Dataset


def get_random_sample(ds: Dataset, seed: int = 0, n: int = 1) -> Dataset | dict:
    """Get random sample from dataset."""
    sub_ds = ds.shuffle(seed=seed).select(range(n))
    if n == 1:
        return sub_ds[0]
    return sub_ds


def get_all_unique_values(ds: Dataset, column: str) -> list:
    """Get all unique values in a column."""
    return ds.unique(column)


def get_bird_subset(ds: Dataset, bird: str) -> Dataset:
    """Get subset of dataset for a specific bird."""
    bird_names = np.array(ds["bird_name"])
    mask = np.where(bird_names == bird)[0]
    return ds.select(mask)
