import numpy as np
import datetime
from typing import Union, Any
from datasets import load_from_disk, Dataset


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_random_sample(ds: Dataset, seed: int = 0, n: int = 1) -> Union[Dataset, dict]:
    """Get random sample from dataset."""
    sub_ds = ds.shuffle(seed=seed).select(range(n))
    if n == 1:
        return sub_ds[0]
    return sub_ds


def get_all_unique_values(ds: Dataset, column: str) -> list:
    """Get all unique values in a column."""
    return ds.unique(column)


def get_bird_subset(
    ds: Dataset, bird: str, bird_name_column: str = "bird_name"
) -> Dataset:
    """Get subset of dataset for a specific bird."""
    bird_names = np.array(ds[bird_name_column])
    mask = np.where(bird_names == bird)[0]
    return ds.select(mask)


def get_age_range_subset(
    ds: Dataset, dph_lower: int = 30, dph_upper: int = 1000, age_column: str = "age"
) -> Dataset:
    """Get subset of dataset for a specific age."""
    ages = np.array(ds[age_column])
    mask = np.where((ages >= dph_lower) & (ages <= dph_upper))[0]
    return ds.select(mask)


def get_recording_date_subset(
    ds: Dataset,
    time_lower: Union[str, datetime.datetime],
    time_upper: Union[str, datetime.datetime],
    date_column: str = "recording_date",
) -> Dataset:
    """Get subset of dataset for a specific recording date.

    :param ds: input dataset
    :type ds: Dataset
    :param time_lower: lower bound for recording date
    :type time_lower: Union[str, datetime.datetime]
    :param time_upper: upper bound for recording date
    :type time_upper: Union[str, datetime.datetime]
    :param date_column: column name for recording date, defaults to "recording_date"
    :type date_column: str, optional
    :return: subset of dataset
    :rtype: Dataset
    """
    if isinstance(time_lower, str):
        time_lower = datetime.datetime.strptime(time_lower, DATETIME_FORMAT)
    if isinstance(time_upper, str):
        time_upper = datetime.datetime.strptime(time_upper, DATETIME_FORMAT)

    dates = np.array(ds[date_column])
    if isinstance(dates[0], str):
        # convert to datetime
        dates = np.array(
            [datetime.datetime.strptime(d, DATETIME_FORMAT) for d in dates]
        )

    mask = np.where((dates >= time_lower) & (dates <= time_upper))[0]
    return ds.select(mask)


class DataExplorer:
    """Data Explorer class for exploring datasets."""

    def __init__(self, path_to_dataset: str | None) -> None:
        """Initialize DataExplorer instance."""
        if path_to_dataset is not None:
            self.ds = load_from_disk(path_to_dataset)
        else:
            self.ds = None

    def load_dataset(self, path_to_dataset: str) -> None:
        """Load dataset from disk."""
        self.ds = load_from_disk(path_to_dataset)

    def get_random_sample(self, seed: int = 0, n: int = 1) -> Union[Dataset, dict]:
        """Get random sample from dataset."""
        return get_random_sample(self.ds, seed, n)

    def get_all_unique_values(self, column: str) -> list[Any]:
        """Get all unique values in a column."""
        return get_all_unique_values(self.ds, column)

    def get_bird_subset(self, bird: str, bird_name_column: str = "bird_name") -> None:
        """Get subset of dataset for a specific bird."""
        return get_bird_subset(self.ds, bird, bird_name_column)

    def get_age_range_subset(
        self,
        dph_lower: int = 30,
        dph_upper: int = 1000,
        age_column: str = "days_post_hatch",
    ) -> Dataset:
        """Get subset of dataset for a specific age."""
        return get_age_range_subset(self.ds, dph_lower, dph_upper, age_column)

    def get_recording_date_subset(
        self,
        time_lower: Union[str, datetime.datetime],
        time_upper: Union[str, datetime.datetime],
        date_column: str = "recording_date",
    ) -> Dataset:
        """Get subset of dataset for a specific recording date."""
        return get_recording_date_subset(self.ds, time_lower, time_upper, date_column)
