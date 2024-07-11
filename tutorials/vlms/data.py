import collections.abc
import random
import copy

from torch.utils.data import Dataset as _TorchDataset
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from PIL import Image

import pandas as pd
import numpy as np


def loader(dataframe_path, path_images, categories, transforms, batch_size=8, num_workers=0):

    # Load data from dict file in txt
    data = []
    dataframe = pd.read_excel(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()
        labels = [sample_df[iTarget] for iTarget in categories]
        if np.sum(labels) == 1:
            # Image and label
            data_i = {"image_path": path_images + sample_df["image_name"],
                      "label": int(np.argmax([sample_df[iTarget] for iTarget in categories]))}
            data.append(data_i)

    dataset = Dataset(data=data, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return loader

def few_shot_loader(dataframe_path, path_images, categories, transforms, shots=16, batch_size=8, num_workers=0, seed=0):

    # Load data from dict file in txt
    data = []
    dataframe = pd.read_excel(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()
        labels = [sample_df[iTarget] for iTarget in categories]
        if np.sum(labels) == 1:
            # Image and label
            data_i = {"image_path": path_images + sample_df["image_name"],
                      "label": int(np.argmax([sample_df[iTarget] for iTarget in categories]))}
            data.append(data_i)

    # Shuffle
    random.seed(seed)
    random.shuffle(data)

    # Few-shot retrieval
    labels = [data_i["label"] for data_i in data]
    unique_labels = np.unique(labels)

    data_fs = []
    for iLabel in unique_labels:
        idx = list(np.squeeze(np.argwhere(labels == iLabel)))
        [data_fs.append(data[iidx]) for iidx in idx[:shots]]


    dataset = Dataset(data=data_fs, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    return loader


class Dataset(_TorchDataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform: Any = transform

    def __len__(self):
        return len(self.data)

    def _transform(self, index):
        
        # Retrieve sample information
        out = copy.deepcopy(self.data[index])
        
        # Read image
        img = Image.open(self.data[index]["image_path"])
        
        # Transform image if requried: normalization, resize, etc.
        if self.transform is None:
            out["image"] = img
        else:
            out["image"] = self.transform(img)
            
        return out

    def __getitem__(self, index):
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)

        return self._transform(index)
    