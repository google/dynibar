"""Defining a dictionary of dataset class."""

from .monocular import MonocularDataset

dataset_dict = {
    'monocular': MonocularDataset,
}
