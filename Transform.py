
import numpy as np
from os.path import dirname, abspath, join

import torch
from torch import nn

# Detect CUDA
_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if _cuda else torch.FloatTensor
BoolTensor = torch.cuda.BoolTensor if _cuda else torch.BoolTensor
LongTensor = torch.cuda.LongTensor if _cuda else torch.LongTensor


_BASE_DIR = dirname(abspath(__file__))


_TRANSFORM_BASE = join(_BASE_DIR, "data", "Static_Port_Transform")


def Inc2Price(data: torch.Tensor) -> torch.Tensor:

    price0 = Tensor(data.shape[0], data.shape[1], 1).fill_(1)
    prices_l = torch.cat((price0, data), dim=2)
    prices_l = torch.cumsum(prices_l, dim=2)
    return prices_l


def _static_port_load_matrix(n_rows: int, static_way: str, insample: bool) -> torch.Tensor:
 
    if "Long" in static_way:
        trans_version = "_".join(["Long", "Stk" + str(n_rows)])
    elif "LShort" in static_way:
        trans_version = "_".join(["LShort", "Stk" + str(n_rows)])
    else:
        raise ValueError("static_way must contain 'Long' or 'LShort'")

    trans_data_path = join(_TRANSFORM_BASE, trans_version)

    if insample:
        fpath = join(trans_data_path, "TransMat_IS.npy")
    else:
        fpath = join(trans_data_path, "TransMat_OOS.npy")

    mat = np.load(fpath)  # shape (n_stocks, n_ports)
    return Tensor(mat)


def StaticPort(prices_l: torch.Tensor,
               n_trans: int,
               static_way: str,
               insample: bool) -> torch.Tensor:
  

    n_rows = prices_l.shape[1]

    trans_mat_full = _static_port_load_matrix(n_rows, static_way, insample)
    # Take first n_trans columns
    trans_mat = trans_mat_full[:, :n_trans]  # shape (assets, n_trans)

    # batch mult:
    # prices_l: (B, A, T) -> swap to (B, T, A)
    swap_prices = prices_l.permute(0, 2, 1)  # (B, T, A)

    # broadcast trans_mat to batches
    broad_trans_mat = trans_mat.reshape(1, *trans_mat.shape)       # (1, A, n_trans)
    broad_trans_mat = broad_trans_mat.repeat(swap_prices.size(0), 1, 1)  # (B, A, n_trans)

    # multiply: (B, T, A) x (B, A, n_trans) -> (B, T, n_trans)
    swap_trans_prices = torch.bmm(swap_prices, broad_trans_mat)

    # permute back: (B, n_trans, T)
    port_prices_l = swap_trans_prices.permute(0, 2, 1)
    return port_prices_l


def BuyHold(prices_l: torch.Tensor, Cap: float) -> torch.Tensor:
 
    BH_money_l = prices_l * Cap
    sum_PNL_BH = BH_money_l[:, :, -1] - BH_money_l[:, :, 0]
    return sum_PNL_BH


def movingaverage(values: torch.Tensor, WH: int) -> torch.Tensor:
  
    mean_conv = nn.Conv1d(1, 1, WH)
    kernel_weights = np.ones((1, 1, WH)) / WH
    mean_conv.weight.data = Tensor(kernel_weights)
    mean_conv.weight.requires_grad = False
    mean_conv.bias.data = Tensor(np.zeros(1))
    mean_conv.bias.requires_grad = False

    # apply per asset
    output_l = [mean_conv(values[:, [ch], :].float()) for ch in range(values.shape[1])]
    # stitch back together
    all_output = values.clone()
    all_output[:, :, WH - 1:] = torch.cat(output_l, dim=1)
    return all_output


def Position_MR(zscores: torch.Tensor,
                Cap: float,
                LR: float,
                SR: float,
                ST_tensor: torch.Tensor,
                LT_tensor: torch.Tensor) -> torch.Tensor:
   

    # Short leg
    zero_cross = ((zscores[:, :-1] < 0) & (zscores[:, 1:] >= 0)) | \
                 ((zscores[:, :-1] > 0) & (zscores[:, 1:] <= 0))
    zero_cross = torch.cat((BoolTensor(zero_cross.shape[0] * [True]).reshape(-1, 1), zero_cross), dim=1)

    sigma_plus = (zscores[:, :-1] < ST_tensor) & (zscores[:, 1:] >= ST_tensor)
    sigma_plus = torch.cat((BoolTensor(sigma_plus.shape[0] * [False]).reshape(-1, 1), sigma_plus), dim=1)

    short_l = -1 * zero_cross + 1 * sigma_plus + 1 * (zero_cross & sigma_plus)
    short_flat = short_l.flatten()
    index_l = LongTensor(np.arange(0, len(short_flat), 1))

    short_nonzero = short_flat[short_flat != 0]
    index_nonzero = index_l[short_flat != 0]

    short_time = torch.cat((BoolTensor([False]),
                             (short_nonzero[:-1] < 0) & (short_nonzero[1:] > 0)))
    short_ts = index_nonzero[short_time]

    clear_time = torch.cat((BoolTensor([False]),
                             (short_nonzero[:-1] > 0) & (short_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    pos_short = torch.zeros(len(short_flat))
    pos_short[short_ts[short_ts % zscores.shape[-1] != 0]] = -1

    pos_clear = torch.zeros(len(short_flat))
    pos_clear[clear_ts[clear_ts % zscores.shape[-1] != 0]] = 1

    short_pos = torch.cumsum(pos_short.reshape(zscores.shape), dim=1) + \
                torch.cumsum(pos_clear.reshape(zscores.shape), dim=1)
    short_pos = short_pos.type(Tensor)

    # Long leg
    sigma_minus = (zscores[:, :-1] > LT_tensor) & (zscores[:, 1:] <= LT_tensor)
    sigma_minus = torch.cat((BoolTensor(sigma_minus.shape[0] * [False]).reshape(-1, 1), sigma_minus), dim=1)

    long_l = -1 * zero_cross + 1 * sigma_minus + 1 * (zero_cross & sigma_minus)
    long_flat = long_l.flatten()

    long_nonzero = long_flat[long_flat != 0]
    index_nonzero = index_l[long_flat != 0]

    long_time = torch.cat((BoolTensor([False]),
                            (long_nonzero[:-1] < 0) & (long_nonzero[1:] > 0)))
    long_ts = index_nonzero[long_time]

    clear_time = torch.cat((BoolTensor([False]),
                             (long_nonzero[:-1] > 0) & (long_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    pos_long = torch.zeros(len(long_flat))
    pos_long[long_ts[long_ts % zscores.shape[-1] != 0]] = 1

    pos_clear2 = torch.zeros(len(long_flat))
    pos_clear2[clear_ts[clear_ts % zscores.shape[-1] != 0]] = -1

    long_pos = torch.cumsum(pos_long.reshape(zscores.shape), dim=1) + \
               torch.cumsum(pos_clear2.reshape(zscores.shape), dim=1)
    long_pos = long_pos.type(Tensor)

    position = Cap * SR * short_pos + Cap * LR * long_pos
    position[:, -1] = 0
    return position


def MeanRev(prices_l: torch.Tensor,
            Cap: float,
            WH: int,
            LR: float,
            SR: float,
            ST,
            LT) -> torch.Tensor:
  

    prices_l_flat = prices_l.view(prices_l.shape[0] * prices_l.shape[1], -1)

    ST_tensor = torch.cat(prices_l.shape[0] * [Tensor(ST)]).reshape(-1, 1)
    LT_tensor = torch.cat(prices_l.shape[0] * [Tensor(LT)]).reshape(-1, 1)

    # rolling mean baseline over first WH+1 timesteps
    prices_l_ma = torch.mean(prices_l[:, :, : WH + 1], dim=2)
    prices_l_ma_flat = prices_l_ma.view(prices_l_ma.shape[0] * prices_l_ma.shape[1], -1)

    zscores = (prices_l_flat - prices_l_ma_flat) / 0.01

    position = Position_MR(zscores, Cap, LR, SR, ST_tensor, LT_tensor)

    PNL_MR_l = position[:, :-1] * (prices_l_flat[:, 1:] - prices_l_flat[:, :-1])
    PNL_MR = PNL_MR_l.reshape(prices_l.shape[0], prices_l.shape[1], -1)
    sum_PNL_MR = torch.sum(PNL_MR, dim=2)
    return sum_PNL_MR


def Position_TF(zscores: torch.Tensor,
                Cap: float,
                LR: float,
                SR: float,
                ST_tensor: torch.Tensor,
                LT_tensor: torch.Tensor) -> torch.Tensor:


    trend_cross = ((zscores[:, :-1] < 0) & (zscores[:, 1:] >= 0)) | \
                  ((zscores[:, :-1] > 0) & (zscores[:, 1:] <= 0))
    trend_cross = torch.cat((BoolTensor(trend_cross.shape[0] * [True]).reshape(-1, 1),
                             trend_cross), dim=1)

    sigma_minus = (zscores[:, :-1] > ST_tensor) & (zscores[:, 1:] <= ST_tensor)
    sigma_minus = torch.cat((BoolTensor(sigma_minus.shape[0] * [False]).reshape(-1, 1),
                             sigma_minus), dim=1)

    short_l = -1 * trend_cross + 1 * sigma_minus + 1 * (trend_cross & sigma_minus)
    short_flat = short_l.flatten()

    index_l = LongTensor(np.arange(0, len(short_flat), 1))

    short_nonzero = short_flat[short_flat != 0]
    index_nonzero = index_l[short_flat != 0]

    short_time = torch.cat((BoolTensor([False]),
                             (short_nonzero[:-1] < 0) & (short_nonzero[1:] > 0)))
    short_ts = index_nonzero[short_time]

    clear_time = torch.cat((BoolTensor([False]),
                             (short_nonzero[:-1] > 0) & (short_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    pos_short = torch.zeros(len(short_flat))
    pos_short[short_ts[short_ts % zscores.shape[-1] != 0]] = -1

    pos_clear = torch.zeros(len(short_flat))
    pos_clear[clear_ts[clear_ts % zscores.shape[-1] != 0]] = 1

    short_pos = torch.cumsum(pos_short.reshape(zscores.shape), dim=1) + \
                torch.cumsum(pos_clear.reshape(zscores.shape), dim=1)
    short_pos = short_pos.type(Tensor)

    sigma_plus = (zscores[:, :-1] < LT_tensor) & (zscores[:, 1:] >= LT_tensor)
    sigma_plus = torch.cat((BoolTensor(sigma_plus.shape[0] * [False]).reshape(-1, 1),
                            sigma_plus), dim=1)

    long_l = -1 * trend_cross + 1 * sigma_plus + 1 * (trend_cross & sigma_plus)
    long_flat = long_l.flatten()

    long_nonzero = long_flat[long_flat != 0]
    index_nonzero = index_l[long_flat != 0]

    long_time = torch.cat((BoolTensor([False]),
                            (long_nonzero[:-1] < 0) & (long_nonzero[1:] > 0)))
    long_ts = index_nonzero[long_time]

    clear_time = torch.cat((BoolTensor([False]),
                             (long_nonzero[:-1] > 0) & (long_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    pos_long = torch.zeros(len(long_flat))
    pos_long[long_ts[long_ts % zscores.shape[-1] != 0]] = 1

    pos_clear2 = torch.zeros(len(long_flat))
    pos_clear2[clear_ts[clear_ts % zscores.shape[-1] != 0]] = -1

    long_pos = torch.cumsum(pos_long.reshape(zscores.shape), dim=1) + \
               torch.cumsum(pos_clear2.reshape(zscores.shape), dim=1)
    long_pos = long_pos.type(Tensor)

    position = Cap * SR * short_pos + Cap * LR * long_pos
    position[:, -1] = 0
    return position


def TrendFollow(prices_l: torch.Tensor,
                Cap: float,
                WH: int,
                LR: float,
                SR: float,
                ST,
                LT) -> torch.Tensor:
   

    prices_l_flat = prices_l.reshape(prices_l.shape[0] * prices_l.shape[1], -1)

    ST_tensor = torch.cat(prices_l.shape[0] * [Tensor(ST)]).reshape(-1, 1)
    LT_tensor = torch.cat(prices_l.shape[0] * [Tensor(LT)]).reshape(-1, 1)

    prices_l_ma = movingaverage(prices_l, WH)
    prices_l_ma2 = movingaverage(prices_l, WH * 2)

    prices_l_ma_flat = prices_l_ma.reshape(prices_l_ma.shape[0] * prices_l_ma.shape[1], -1)
    prices_l_ma2_flat = prices_l_ma2.reshape(prices_l_ma2.shape[0] * prices_l_ma2.shape[1], -1)

    zscores = (prices_l_ma_flat - prices_l_ma2_flat) / 0.01

    position = Position_TF(zscores, Cap, LR, SR, ST_tensor, LT_tensor)

    PNL_TF_l = position[:, :-1] * (prices_l_flat[:, 1:] - prices_l_flat[:, :-1])
    PNL_TF = PNL_TF_l.reshape(prices_l.shape[0], prices_l.shape[1], -1)
    sum_PNL_TF = torch.sum(PNL_TF, dim=2)
    return sum_PNL_TF
