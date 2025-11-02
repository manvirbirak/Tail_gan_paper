

import os
from os.path import join, isfile, dirname, abspath

import numpy as np
import pandas as pd
import torch

# local module you already have
from Transform import Tensor, Inc2Price, movingaverage


# Base directory = folder where this script lives
BASE_DIR = dirname(abspath(__file__))

# Location of generated synthetic series
PARENT_DATA_PATH = join(BASE_DIR, "data")

# Where to save thresholds
THRESH_PARENT_PATH = join(BASE_DIR, "data", "Thresholds")
os.makedirs(THRESH_PARENT_PATH, exist_ok=True)


def _build_thresholds_path(data_name, tickers, strategy, percentile_l):

    ticker_str = "_".join(tickers)
    pct_str = "_".join(map(str, percentile_l))

    if "MR" in strategy and "Port" not in strategy:
        fname = f"{ticker_str}_MR_{pct_str}.npy"
    elif "MR" in strategy and "Port" in strategy:
        fname = f"{ticker_str}_Port*MR_{pct_str}.npy"
    elif "TF" in strategy and "Port" not in strategy:
        fname = f"{ticker_str}_TF_{pct_str}.npy"
    elif "TF" in strategy and "Port" in strategy:
        fname = f"{ticker_str}_Port*TF_{pct_str}.npy"
    else:
        raise ValueError("strategy must contain 'MR' or 'TF'")

    folder = join(THRESH_PARENT_PATH, data_name)
    os.makedirs(folder, exist_ok=True)

    return join(folder, fname)


def gen_thresholds(data_name, tickers, strategy, percentile_l, length, WH):
  
    # output path
    thresholds_path = _build_thresholds_path(
        data_name=data_name,
        tickers=tickers,
        strategy=strategy,
        percentile_l=percentile_l,
    )

    # if cached result exists, reuse it
    if isfile(thresholds_path):
        thresholds_array_stocks = np.load(thresholds_path)
        return thresholds_array_stocks

    # load training data
    data_path = join(PARENT_DATA_PATH, data_name)
    files = os.listdir(data_path)
    files.sort()

    data_l = []
    for item in range(length):
        file_path = join(data_path, files[item])
        tmp_data = pd.read_csv(file_path)[tickers].values.T  # shape (#tickers, n_cols)
        data_l.append(tmp_data)

    # stack into tensor of shape:
    # (length, #tickers, n_cols)
    data = np.stack(data_l)

    # convert to whatever Tensor() expects 
    data = Tensor(data)

    # convert increments to prices
    prices_l = Inc2Price(data)  # expected shape: (length, #tickers, n_cols)

    # flatten across samples and time: (length * n_cols, #tickers)
    prices_l_flat = prices_l.view(prices_l.shape[0] * prices_l.shape[1], -1)

    thresholds_array_list = []

    # loop over each ticker or portfolio dimension
    for stk in range(data.shape[1]):
        if "MR" in strategy:
            # rolling mean reversion anchor
            # prices_l[:, :, :WH+1] means first WH+1 timesteps
            prices_l_ma = torch.mean(prices_l[:, :, : WH + 1], dim=2)
            prices_l_ma_flat = prices_l_ma.view(
                prices_l_ma.shape[0] * prices_l_ma.shape[1], -1
            )

            # z-score like deviation from mean
            zscores_MR = (prices_l_flat - prices_l_ma_flat) / 0.01
            zscores_MR = zscores_MR.cpu().detach().numpy()

            # percentiles across all samples/time for this stk
            thresholds_array = np.array(
                [np.percentile(zscores_MR[:, stk], p) for p in percentile_l]
            )
            thresholds_array_list.append(thresholds_array)

        elif "TF" in strategy:
            # short and long moving averages for trend following
            prices_l_ma = movingaverage(prices_l, WH)
            prices_l_ma2 = movingaverage(prices_l, WH * 2)

            # reshape to (length * #tickers, n_cols?) then flatten across time?
            # original code flattens along (batch * asset, -1)
            prices_l_ma_flat = prices_l_ma.reshape(
                prices_l_ma.shape[0] * prices_l_ma.shape[1], -1
            )
            prices_l_ma2_flat = prices_l_ma2.reshape(
                prices_l_ma2.shape[0] * prices_l_ma2.shape[1], -1
            )

            zscores_TF = (prices_l_ma_flat - prices_l_ma2_flat) / 0.01
            zscores_TF = zscores_TF.cpu().detach().numpy()

            thresholds_array = np.array(
                [np.percentile(zscores_TF[:, stk], p) for p in percentile_l]
            )
            thresholds_array_list.append(thresholds_array)

        else:
            raise ValueError("strategy must contain 'MR' or 'TF'")

    # shape (#tickers, len(percentile_l))
    thresholds_array_stocks = np.stack(thresholds_array_list)

    # save for reuse
    np.save(thresholds_path, thresholds_array_stocks)

    return thresholds_array_stocks
