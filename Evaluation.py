

import os
import random
import numpy as np
import pandas as pd
import torch
from os.path import join, dirname, abspath, isfile, isdir

# import config + helpers from training script
from TailGAN import (
    opt,
    this_version,
    DATA_ROOT,
    GEN_OUT_ROOT,
    gen_data_path,
    Tensor,
    Screen_Ensemble,
)

from Dataset import Dataset_IS
from Transform import (
    Inc2Price,
    StaticPort,
    BuyHold,
    MeanRev,
    TrendFollow,
)

from gen_thresholds import gen_thresholds



# how many samples we draw when we estimate VaR/ES for fake data
SAMPLE_SIZE = 1000


BASE_DIR = dirname(abspath(__file__))
EVAL_ROOT = join(BASE_DIR, f"eval_out_S{SAMPLE_SIZE}")
os.makedirs(EVAL_ROOT, exist_ok=True)


def empirical_var_es(pnl_matrix, alphas):

    results = {}
    for alpha in alphas:
        # VaR at alpha (left tail if alpha<0.5)
        var_vec = np.percentile(pnl_matrix, alpha * 100.0, axis=0)
        # ES: average of tail beyond VaR
        if alpha < 0.5:
            mask = (pnl_matrix <= var_vec[None, :])
        else:
            mask = (pnl_matrix >= var_vec[None, :])

        # replace non-tail entries with NaN so we can mean over tail only
        tail_vals = np.where(mask, pnl_matrix, np.nan)
        es_vec = np.nanmean(tail_vals, axis=0)

        results[alpha] = (var_vec, es_vec)
    return results


def flatten_var_es_dict(stats_dict):
  
    flat_parts = []
    col_labels = []
    for alpha, (var_vec, es_vec) in stats_dict.items():
        for idx in range(len(var_vec)):
            flat_parts.append(var_vec[idx])
            flat_parts.append(es_vec[idx])
            col_labels.append(f"Strat{idx+1}_VaR_{alpha:.2f}")
            col_labels.append(f"Strat{idx+1}_ES_{alpha:.2f}")
    flat_vector = np.array(flat_parts)
    return flat_vector, col_labels


def compute_pnl_numpy(R_batch, opt):
 
    R_tensor = Tensor(R_batch)  # float32, device-aware

    # 1. Turn increments/returns -> price paths starting at 1
    prices_l = Inc2Price(R_tensor)  # (B, n_rows, T)

    # 2. Static portfolios
    port_prices_l = StaticPort(
        prices_l,
        opt.n_trans,
        opt.static_way,
        insample=True
    )

    # 3. Buy & Hold on each asset
    pnl_components = []
    pnl_bh = BuyHold(prices_l, opt.Cap)  # (B, n_rows)
    pnl_components.append(pnl_bh)

    # 4. For each selected strategy, append strategy PnL blocks
    for strat in opt.strategies:
        if strat == "Port":
            # buy-hold across static portfolios
            pnl_port = BuyHold(port_prices_l, opt.Cap)  # (B, n_trans)
            pnl_components.append(pnl_port)

        elif strat == "MR":
            for percentile_pair in opt.thresholds_pct:
                thr = gen_thresholds(
                    opt.data_name,
                    opt.tickers,
                    "MR",
                    percentile_pair,
                    length=100,
                    WH=opt.WH
                )  # shape (n_rows, 2+)
                pnl_mr = MeanRev(
                    prices_l,
                    opt.Cap,
                    opt.WH,
                    LR=opt.ratios[0],
                    SR=opt.ratios[1],
                    ST=thr[:, -1],  # upper trigger
                    LT=thr[:, -2],  # lower trigger
                )  # (B, n_rows)
                pnl_components.append(pnl_mr)

        elif strat == "TF":
            for percentile_pair in opt.thresholds_pct:
                thr = gen_thresholds(
                    opt.data_name,
                    opt.tickers,
                    "TF",
                    percentile_pair,
                    length=100,
                    WH=opt.WH
                )  # shape (n_rows, 2)
                pnl_tf = TrendFollow(
                    prices_l,
                    opt.Cap,
                    opt.WH,
                    LR=opt.ratios[0],
                    SR=opt.ratios[1],
                    ST=thr[:, 0],
                    LT=thr[:, 1],
                )  # (B, n_rows)
                pnl_components.append(pnl_tf)
        else:
            pass

    pnl_tensor = torch.cat(pnl_components, dim=1)  # (B, total_strat_rows)
    return pnl_tensor.detach().cpu().numpy()


def load_real_returns(opt):
  
    data_dir = join(DATA_ROOT, opt.data_name)
    dataset = Dataset_IS(
        tickers=opt.tickers,
        data_path=data_dir,
        length=opt.len
    )
    # dataset.samples is already a list of tensors (n_rows, n_cols)
    arr = [s.detach().numpy() for s in dataset.samples]
    real_returns = np.stack(arr, axis=0)  # (N, n_rows, n_cols)
    return real_returns


def load_generated_returns_for_model(model_index):
  
    files = os.listdir(gen_data_path)
    # pick files for this model
    fake_files = [
        f for f in files
        if f.startswith("Fake_id%d_" % model_index) and f.endswith(".npy")
    ]

    # map epoch -> list of samples
    epoch_map = {}
    for f in fake_files:
        # example name: Fake_id0_E12.npy
        try:
            epoch_str = f.split("_E")[1].split(".npy")[0]
            epoch_num = int(epoch_str)
        except Exception:
            continue

        arr = np.load(join(gen_data_path, f))  # (gen_size_at_save, n_rows, n_cols)
        epoch_map.setdefault(epoch_num, []).append(arr)

    if not epoch_map:
        return [], []

    # sort epochs
    epochs_sorted = sorted(epoch_map.keys())

    fake_batches = []
    for ep in epochs_sorted:
        cat_ep = np.concatenate(epoch_map[ep], axis=0)
        fake_batches.append(cat_ep)
    return fake_batches, epochs_sorted


def compute_ground_truth_stats(opt):
   
    real_returns = load_real_returns(opt)

    # Convert those returns to PnL for all strategies
    real_pnl = compute_pnl_numpy(real_returns, opt)  # shape (N, total_strat_rows)

    # Compute VaR/ES for each alpha in opt.alphas
    gt_stats = empirical_var_es(real_pnl, opt.alphas)

    gt_flat, gt_cols = flatten_var_es_dict(gt_stats)

    # Save to disk once for reference
    out_dir = join(EVAL_ROOT, "GroundTruth")
    os.makedirs(out_dir, exist_ok=True)
    out_path = join(out_dir, f"{this_version}_groundtruth.csv")
    df_gt = pd.DataFrame([gt_flat], index=["Real"], columns=gt_cols)
    df_gt.to_csv(out_path)
    return gt_flat, gt_cols


def estimate_sampling_error(opt, gt_flat, gt_cols):
    

    real_returns = load_real_returns(opt)  # (N, n_rows, n_cols)
    N = real_returns.shape[0]
    if N < SAMPLE_SIZE:
        # if dataset is smaller than SAMPLE_SIZE, just cap it
        effective_sample = N
    else:
        effective_sample = SAMPLE_SIZE

    rel_err_runs = []
    for _ in range(16):
        idx = random.sample(range(N), effective_sample)
        sample_returns = real_returns[idx, :, :]

        sample_pnl = compute_pnl_numpy(sample_returns, opt)
        sample_stats = empirical_var_es(sample_pnl, opt.alphas)
        sample_flat, _ = flatten_var_es_dict(sample_stats)

        # relative error |true - est| / |true|
        # add small epsilon to denominator to avoid /0 if needed
        eps = 1e-9
        rel_err = np.abs(gt_flat - sample_flat) / (np.abs(gt_flat) + eps)
        rel_err_runs.append(rel_err)

    rel_err_runs = np.stack(rel_err_runs, axis=1)  # (num_metrics, 16)

    mean_rel_err = np.mean(rel_err_runs, axis=1)
    std_rel_err = np.std(rel_err_runs, axis=1)

    df = pd.DataFrame(
        np.vstack([mean_rel_err, std_rel_err]),
        index=["Sample-RE-Mean", "Sample-RE-Std"],
        columns=gt_cols
    )

    out_path = join(EVAL_ROOT, f"{this_version}_sampling_error.csv")
    df.to_csv(out_path)

    print("=== Sampling Error Baseline ===")
    print(df.mean(axis=1))  # average over all strategies/alphas
    return mean_rel_err, std_rel_err


def evaluate_model_generated(opt, model_index, gt_flat, gt_cols):


    fake_batches, epoch_list = load_generated_returns_for_model(model_index)
    if not fake_batches:
        print(f"[WARN] No generated samples found for model {model_index}")
        return None

    mean_err_per_epoch = []
    std_err_per_epoch = []

    for batch_idx, fake_returns_epoch in enumerate(fake_batches):
        ep = epoch_list[batch_idx]
        print(f"Evaluating model {model_index} epoch {ep} ...")

        # resample multiple times for robustness
        N_fake = fake_returns_epoch.shape[0]
        effective_sample = min(SAMPLE_SIZE, N_fake)

        epoch_rel_err_runs = []
        for _ in range(16):
            idx = random.sample(range(N_fake), effective_sample)
            sub_fake = fake_returns_epoch[idx, :, :]  # (effective_sample, n_rows, n_cols)

            fake_pnl = compute_pnl_numpy(sub_fake, opt)
            fake_stats = empirical_var_es(fake_pnl, opt.alphas)
            fake_flat, _ = flatten_var_es_dict(fake_stats)

            eps = 1e-9
            rel_err = np.abs(gt_flat - fake_flat) / (np.abs(gt_flat) + eps)
            epoch_rel_err_runs.append(rel_err)

        epoch_rel_err_runs = np.stack(epoch_rel_err_runs, axis=1)  # (num_metrics, 16)
        mean_err = np.mean(epoch_rel_err_runs, axis=1)
        std_err = np.std(epoch_rel_err_runs, axis=1)

        mean_err_per_epoch.append(mean_err)
        std_err_per_epoch.append(std_err)

    mean_err_per_epoch = np.stack(mean_err_per_epoch, axis=0)  # (num_epochs_saved, num_metrics)
    std_err_per_epoch = np.stack(std_err_per_epoch, axis=0)

    model_eval_dir = join(EVAL_ROOT, f"{this_version}_Model_{model_index}")
    os.makedirs(model_eval_dir, exist_ok=True)

    df_mean = pd.DataFrame(mean_err_per_epoch, index=epoch_list, columns=gt_cols)
    df_std = pd.DataFrame(std_err_per_epoch, index=epoch_list, columns=gt_cols)

    df_mean.to_csv(join(model_eval_dir, "Mean_OOS_RE_Mean.csv"))
    df_std.to_csv(join(model_eval_dir, "Std_OOS_RE.csv"))

    return df_mean, df_std


def summarize_models(opt):
   

    folders = [
        f for f in os.listdir(EVAL_ROOT)
        if f.startswith(f"{this_version}_Model_")
        and isdir(join(EVAL_ROOT, f))
    ]

    if not folders:
        print("[WARN] No per-model eval folders found to summarize.")
        return

    all_best_means = []
    all_best_stds = []

    for fldr in folders:
        model_dir = join(EVAL_ROOT, fldr)
        mean_path = join(model_dir, "Mean_OOS_RE_Mean.csv")
        std_path = join(model_dir, "Std_OOS_RE.csv")

        if not (isfile(mean_path) and isfile(std_path)):
            continue

        mean_df = pd.read_csv(mean_path, index_col=0)
        std_df = pd.read_csv(std_path, index_col=0)

        # pick epoch with minimum average relative error across all metrics
        avg_err_per_epoch = mean_df.mean(axis=1)  # Series indexed by epoch
        best_epoch_idx = avg_err_per_epoch.idxmin()
        best_epoch_val = avg_err_per_epoch.min() * 100.0  # convert to %
        best_std_val = std_df.mean(axis=1).loc[best_epoch_idx] * 100.0

        all_best_means.append(best_epoch_val)
        all_best_stds.append(best_std_val)

    if not all_best_means:
        print("[WARN] No valid eval CSVs found in model folders.")
        return

    overall_mean = float(np.mean(all_best_means))
    overall_std = float(np.mean(all_best_stds))

    # escape % in the string OR just use f-strings
    print(f"=== Overall Tail Risk Fit Across Models ===")
    print(f"Mean relative error (best epoch, %): {overall_mean:.2f}")
    print(f"Std of relative error (same epoch, %): {overall_std:.2f}")



if __name__ == "__main__":
    # 1. Ground truth VaR/ES on real data
    gt_flat, gt_cols = compute_ground_truth_stats(opt)

    # 2. Sampling error baseline (how hard is the problem even for real data)
    estimate_sampling_error(opt, gt_flat, gt_cols)

    # 3. Evaluate all selected ensemble members
    selected_models = Screen_Ensemble(thres_perc=50)
    # In debug mode with numNN=1, Screen_Ensemble might just return [0]
    if not selected_models:
        selected_models = [0]

    for mid in selected_models:
        print(f"--- Evaluating model {mid} ---")
        evaluate_model_generated(opt, mid, gt_flat, gt_cols)

    # 4. Summarise across models
    summarize_models(opt)
