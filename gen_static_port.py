

import os
from os.path import join, isfile, dirname, abspath

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats


# Base directory = folder where this script lives
BASE_DIR = dirname(abspath(__file__))

# Parent save path in repo data/
TRANS_PARENT_DATA_PATH = join(BASE_DIR, "data", "Static_Port_Transform")
os.makedirs(TRANS_PARENT_DATA_PATH, exist_ok=True)


def IsStock(vec: np.ndarray) -> bool:

    isstock_l = (np.abs(vec) == 1.0) + (np.abs(vec) == 0.0)
    return np.all(isstock_l)


def Gen_StaticPort(static_way: str, n_stocks: int, n_ports: int):
 

    if "Long" in static_way:
        trans_version = "_".join(["Long", "Stk" + str(n_stocks)])
    elif "LShort" in static_way:
        trans_version = "_".join(["LShort", "Stk" + str(n_stocks)])
    else:
        raise ValueError("static_way must contain 'Long' or 'LShort'")

    trans_data_path = join(TRANS_PARENT_DATA_PATH, trans_version)
    os.makedirs(trans_data_path, exist_ok=True)

    # We generate 2 variants:
    #   TransMat_IS.npy
    #   TransMat_OOS.npy
    for inout in range(2):
        if inout == 0:
            store_path = join(trans_data_path, "TransMat_IS.npy")
        else:
            store_path = join(trans_data_path, "TransMat_OOS.npy")

        price_start = np.ones(n_stocks)

        if "Long" in static_way:
            # Long-only portfolios.
            # Dense random {0,1} weights.
            unscale_trans2port_mat = sparse.rand(
                n_stocks,
                max(int(n_stocks ** 2), n_ports),
                density=1.0,
            ).toarray()

            # sanity: no empty rows or cols
            assert np.all(unscale_trans2port_mat.sum(0) > 0)
            assert np.all(unscale_trans2port_mat.sum(1) > 0)

        elif "LShort" in static_way:
            # Long-short portfolios.
            # Mostly dense weights drawn from N(0,1).
            rvs = stats.norm(loc=0, scale=1).rvs
            unscale_trans2port_mat = sparse.random(
                n_stocks,
                max(int(n_stocks ** 2), n_ports),
                density=0.9,
                data_rvs=rvs,
            ).toarray()

        else:
            # shouldn't get here due to earlier raise
            unscale_trans2port_mat = None

        # Scale each column so absolute weights sum to 1.
        # Avoid zero-divide because we asserted no empty cols above.
        scale_trans_mat = unscale_trans2port_mat / np.abs(
            unscale_trans2port_mat
        ).sum(0)

        # Position matrix rescales by starting price exposure.
        position = np.diag(
            1.0 / price_start.dot(np.abs(scale_trans_mat))
        )  # shape (n_cols, n_cols)

        trans_mat = np.dot(scale_trans_mat, position)  # shape (n_stocks, n_cols)

        # Drop any column that is just a single stock long/short position.
        mask_keep = ~np.apply_along_axis(IsStock, 0, trans_mat)
        trans_mat_port = trans_mat[:, mask_keep]

        assert trans_mat_port.shape[1] >= 10, "Not enough diversified portfolios kept"

        # Preview to terminal
        print(f"[{store_path}] shape {trans_mat_port.shape}")
        print(pd.DataFrame(trans_mat_port).head())

        # Save
        np.save(store_path, trans_mat_port)


if __name__ == "__main__":
    # Example call
    Gen_StaticPort(static_way="LShort", n_stocks=5, n_ports=50)
