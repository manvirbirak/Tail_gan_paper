# Tail-GAN: Generating Market Scenarios That Match Tail Risk

## Overview

Tail-GAN is a copy of the original paper but is a small research/portfolio project (https://arxiv.org/pdf/2203.01664).

- Generate multivariate synthetic asset returns,
- Build trading strategy PnL series from those returns,
- Train a generator so that the **tail behaviour (VaR / ES)** of those PnLs matches the real ones.

This is useful for risk modelling, stress testing, and backtesting when real data is limited or sensitive.

The project is fully local and CPU-friendly at debug scale.

---

## Pipeline

### 1. Synthetic market data generation (`gen_synthetic.py`)
Simulate multivariate time series with different behaviours:
- Gaussian noise
- AR(·) processes with positive/negative autocorrelation
- GARCH-style volatility
- Heavy-tailed shocks (Student-t)


### 2. Static portfolio construction (`gen_static_port.py`)
Create many static long/short portfolios by combining the base assets with random weights.

These transformation matrices let me treat “portfolio PnL series” as extra instruments.

### 3. Threshold calibration for signals (`gen_thresholds.py`)
For dynamic strategies like:
- Mean Reversion (MR)
- Trend Following (TF)

I compute signal thresholds from historical behaviour:
- rolling moving averages
- z-scores
- percentile cutoffs (e.g. 31st and 69th percentiles)

This lets me generate trading logic (when to go long, when to short) in a purely data-driven way.

### 4. PnL engine (`Transform.py`)
Given raw asset returns, I build PnL for:
- Buy & Hold (per asset)
- Static portfolios (the long/short mixes above)
- Mean Reversion strategy
- Trend Following strategy

### 5. NeuralSort + Tail risk scoring (`util.py`, `TailGAN.py`)
The discriminator does NOT try to classify real vs fake directly.

Instead:
1. Compute PnL for each strategy.
2. Soft-sort those PnLs using a differentiable sorting layer (NeuralSort).
3. Estimate tail risk metrics for each strategy:
   - VaR (Value-at-Risk) at some α (e.g. 5%)
   - ES (Expected Shortfall / CVaR)

4. Score how close the generated tail risk is to the real tail risk using a strictly proper scoring rule.

The generator tries to produce return paths whose trade PnLs lead to realistic tails.
The discriminator outputs (VaR, ES)-like pairs for multiple α levels.

### 6. Training (`TailGAN.py`)
Key settings in `TailGAN.py`:
- `n_epochs`: training epochs
- `batch_size`
- `latent_dim`: size of noise vector fed to generator
- `strategies`: which strategies to include in PnL (`['Port','MR','TF']`)
- `alphas`: which tail quantiles to match (e.g. `[0.05]` for 5% VaR/ES)
- `len`: how many real samples to use
- `numNN`: how many GANs to train in an ensemble

### 7. Evaluation (`Evaluation.py`)
This script checks if the GAN actually matches tail risk.

Steps:
1. Compute the “ground truth” VaR & ES from the real data.
2. For each saved generator snapshot (each epoch), generate fake data and recompute VaR & ES from the fake.
3. Measure relative error:
   - lower % error is better: “fake tail matches true tail”.
   - high % error means the generator hasn’t learned realistic crash behaviour yet.

Even at small debug configs (tiny dataset, very few epochs, CPU), the full pipeline runs and produces a quantitative tail-risk gap metric. This is the proof-of-concept.

## Reference

Cont, R., Cucuringu, M., Xu, R., & Zhang, C. (2025). *Tail-GAN: Learning to simulate tail risk scenarios*. Management Science. INFORMS.


---

## Repo layout

```text
Tail_gan_paper/
│
├─ gen_synthetic.py          # generate base multivariate return data
├─ gen_static_port.py        # build + save static long/short portfolios
├─ gen_thresholds.py         # compute MR/TF thresholds from data
├─ Transform.py              # PnL logic for strategies (Buy&Hold, MR, TF, etc.)
├─ util.py                   # NeuralSort and helpers
├─ Dataset.py                # PyTorch dataset wrapper for training
├─ TailGAN.py                # training loop (Generator, Discriminator, scoring)
├─ Evaluation.py             # VaR/ES fit evaluation, sampling error baseline
