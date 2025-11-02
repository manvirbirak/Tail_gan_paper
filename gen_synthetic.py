

import os
from os.path import join, isfile, dirname, abspath

import numpy as np
import pandas as pd
from sklearn.datasets import make_sparse_spd_matrix
from statsmodels.stats.moment_helpers import corr2cov


# Base directory = folder where this script lives
BASE_DIR = dirname(abspath(__file__))

# data directory = <repo>/data
PARENT_DATA_PATH = join(BASE_DIR, "data")


def transform_1d(row_type, r, n_cols, alpha, beta, w):
  
    if row_type == "Gauss":
        tmp_data = r[1:]

    elif "AR" in row_type and "GARCH" not in row_type:
        tmp_data = np.zeros_like(r)

        coef = float(row_type.split("AR")[1])  # e.g. "50" or "-12"
        for k in range(1, r.shape[0]):
            tmp_data[k] = coef / 100.0 * tmp_data[k - 1] + r[k - 1]

        tmp_data = np.delete(tmp_data, 0)

    elif "GARCH" in row_type:
        # innovation
        if row_type == "GARCH":
            eps = r
        elif "-T" in row_type:
            deg_free = int(row_type.split("-T")[1])
            eps = r / np.sqrt(
                np.random.chisquare(df=deg_free, size=r.shape[0]) / deg_free
            )
        else:
            eps = np.zeros_like(r)

        delt_squ = np.zeros_like(eps)
        tmp_data = np.zeros_like(eps)

        for k in range(1, r.shape[0]):
            delt_squ[k] = w + alpha * tmp_data[k - 1] ** 2 + beta * delt_squ[k - 1]
            tmp_data[k] = np.sqrt(delt_squ[k]) * eps[k]

        tmp_data = np.delete(tmp_data, 0)

    else:
        tmp_data = np.zeros_like(r)

    # drop warmup
    return tmp_data[n_cols:]


def extract_data_info(data_name):
  
    raw_row_type_list = data_name.split("+")
    row_type_list = []
    for row_type_str in raw_row_type_list:
        count = int(row_type_str.split("_")[0])
        row_type = row_type_str.split("_")[1]
        row_type_list.extend([row_type] * count)
    return row_type_list


def gen_data(data_name, length, n_rows, n_cols):
    print("[gen_data]", data_name)

    data_path = join(PARENT_DATA_PATH, data_name)
    os.makedirs(data_path, exist_ok=True)

    row_type_list = extract_data_info(data_name)
    assert len(row_type_list) == n_rows, "row_type_list length mismatch with n_rows"

    basic_info_path = join(data_path, "basic_info.npz")

    if isfile(basic_info_path):
        basic_info = np.load(basic_info_path)
        mean = basic_info["mean"]
        cov = basic_info["cov"]
        std = np.sqrt(np.diag(cov))
        alpha_l = basic_info["alpha"]
        beta_l = basic_info["beta"]
        w_l = basic_info["w"]
    else:
        mean = np.zeros(n_rows)

        corr = np.abs(make_sparse_spd_matrix(n_rows, alpha=0.0, norm_diag=True))

        std = np.random.uniform(0.3, 0.5, n_rows) / np.sqrt(250 * n_cols)

        cov = corr2cov(corr, std)

        alpha_l = np.random.uniform(0.08, 0.12, n_rows)
        beta_l = np.random.uniform(0.825, 0.875, n_rows)
        w_l = np.random.uniform(0.03, 0.07, n_rows)

        np.savez(
            basic_info_path,
            mean=mean,
            cov=cov,
            alpha=alpha_l,
            beta=beta_l,
            w=w_l,
        )

    for item in range(length):
        if item % max(1, int(length / 10)) == 0:
            print(f"[{item}/{length}]")

        # shape: (n_rows, 2*n_cols+1)
        data_r = np.random.multivariate_normal(
            mean, cov, size=(2 * n_cols + 1)
        ).T

        tmp_data_l = []
        for i, row_type in enumerate(row_type_list):
            series_vals = transform_1d(
                row_type,
                data_r[i, :],
                n_cols,
                alpha_l[i],
                beta_l[i],
                w_l[i],
            )
            tmp_data_l.append(series_vals)

        data_raw = np.stack(tmp_data_l)  # (n_rows, n_cols)

        # rescale rows to target std
        row_std = np.std(data_raw, axis=1, keepdims=True)  # (n_rows,1)
        data_scaled = data_raw / row_std * std.reshape(-1, 1)

        df = pd.DataFrame(data_scaled).T
        df.columns = row_type_list

        df.to_csv(join(data_path, f"{item + 1}.csv"), index=False)


if __name__ == "__main__":
    data_name = "1_Gauss+1_AR50+1_AR-12+1_GARCH-T5+1_GARCH-T10"
    length = 1000      # keep small for debug not 500000
    n_rows = 5
    n_cols = 100
    gen_data(data_name, length, n_rows, n_cols)

