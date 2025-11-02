

import os
import numpy as np
import torch
from torch.utils import data


def _load_csv_subset(path, tickers):
 

    # Read first line -> header
    with open(path, "r") as f:
        header = f.readline().strip().split(",")

    # Figure out which columns we need (by name)
    col_idx = [header.index(t) for t in tickers]

    # Load just those columns using numpy
    # np.loadtxt will return shape (n_rows_csv, len(col_idx))
    data_block = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=col_idx,
        dtype=np.float32,
    )

    # data_block is shape (n_cols, len(tickers))
    data_block = data_block.T

    return data_block  # float32 np array


class Dataset_IS(data.Dataset):
 
    def __init__(self, tickers, data_path, length):
        self.tickers = tickers
        self.data_path = data_path
        self.length = length

        # list of CSVs in sorted order
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith(".csv")]
        files.sort()

        # take first `length` files
        files = files[: self.length]

        self.samples = []
        for f in files:
            f_path = os.path.join(self.data_path, f)
            arr = _load_csv_subset(f_path, self.tickers)          # np.float32, (n_rows, n_cols)
            t = torch.from_numpy(arr)                             # torch.float32
            self.samples.append(t)

        # if there are fewer than requested, adjust length
        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # returns tensor of shape (n_rows, n_cols), dtype float32
        return self.samples[index]


class Dataset_OOS(data.Dataset):
 
    def __init__(self, tickers, data_path, length):
        self.tickers = tickers
        self.data_path = data_path
        self.split = length  # IS length cutoff

        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith(".csv")]
        files.sort()

        # take files AFTER first `length`
        files = files[self.split :]

        self.samples = []
        for f in files:
            f_path = os.path.join(self.data_path, f)
            arr = _load_csv_subset(f_path, self.tickers)          # np.float32, (n_rows, n_cols)
            t = torch.from_numpy(arr)                             # torch.float32
            self.samples.append(t)

        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.samples[index]
