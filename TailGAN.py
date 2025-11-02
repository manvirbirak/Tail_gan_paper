

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset import Dataset_IS
from Transform import (
    Inc2Price,
    StaticPort,
    BuyHold,
    MeanRev,
    TrendFollow,
    Tensor,  # Tensor here is FloatTensor or CudaFloatTensor defined in Transform
)
from gen_thresholds import gen_thresholds
from util import deterministic_NeuralSort  # our differentiable NeuralSort

from os.path import join, dirname, abspath

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


BASE_DIR = dirname(abspath(__file__))

# data from generator 
DATA_ROOT = join(BASE_DIR, "data")

# model checkpoints + generated samples 
GEN_OUT_ROOT = join(BASE_DIR, "gen_out")
os.makedirs(GEN_OUT_ROOT, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=3000, help="epochs for training")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr_D", type=float, default=1e-7, help="learning rate for Discriminator")
parser.add_argument("--lr_G", type=float, default=1e-6, help="learning rate for Generator")
parser.add_argument("--temp", type=float, default=0.01, help="multiplier of temperature for NeuralSort")
parser.add_argument("--b1", type=float, default=0.5, help="adam beta1")
parser.add_argument("--b2", type=float, default=0.999, help="adam beta2")
parser.add_argument("--latent_dim", type=int, default=1000, help="dim of latent z")
parser.add_argument("--len", type=int, default=50000, help="number of examples to use from dataset")
parser.add_argument("--n_rows", type=int, default=5, help="number of assets/rows")
parser.add_argument("--n_cols", type=int, default=100, help="number of timesteps/cols")
parser.add_argument("--n_critic_G", type=int, default=1, help="G update frequency")
parser.add_argument("--n_critic_D", type=int, default=1, help="D update frequency")
parser.add_argument("--static_way", type=str, default='LShort', help="static portfolio style: Long or LShort")
parser.add_argument(
    "--strategies",
    type=list,
    default=['Port', 'MR', 'TF'],
    help="strategy names to include in discriminator PNL (subset of ['Port','MR','TF'])",
)
parser.add_argument("--n_trans", type=int, default=50, help="num static portfolios to include")
parser.add_argument("--Cap", type=int, default=10, help="capital scaling for PnL")
parser.add_argument("--WH", type=int, default=10, help="window length for signals (MA window etc.)")
parser.add_argument(
    "--ratios",
    type=list,
    default=[1.0, 1.0],
    help="ratios [LR, SR] for long/short sizing in MR/TF",
)
parser.add_argument(
    "--thresholds_pct",
    type=list,
    default=[[31, 69]],
    help="percentile pairs used to pick signal thresholds",
)
parser.add_argument(
    "--data_name",
    type=str,
    default='1_Gauss+1_AR50+1_AR-12+1_GARCH-T5+1_GARCH-T10',
    help="folder name under ./data containing CSVs",
)
parser.add_argument(
    "--tickers",
    type=list,
    default=['Gauss', 'AR50', 'AR-12', 'GARCH-T5', 'GARCH-T10'],
    help="column names in those CSVs",
)
parser.add_argument(
    "--noise_name",
    type=str,
    default='t5',
    help="latent noise distribution, e.g. 't5' for Student-t df=5 or 'normal'",
)
parser.add_argument(
    "--alphas",
    type=list,
    default=[0.05],
    help="quantile levels for discriminator scoring, e.g. [0.05] for 5%% VaR/ES",
)
parser.add_argument(
    "--W",
    type=float,
    default=10.0,
    help="scale parameter W for S_quant score",
)
parser.add_argument(
    "--score",
    type=str,
    default='quant',
    help="score function: 'quant' or 'stats'",
)
parser.add_argument(
    "--numNN",
    type=int,
    default=10,
    help="number of GANs to train (ensemble members)",
)
parser.add_argument(
    "--project",
    type=bool,
    default=True,
    help="whether discriminator applies projection constraint",
)
parser.add_argument(
    "--version",
    type=str,
    default=f'Test{seed}',
    help="experiment version tag",
)

opt = parser.parse_args()
print(opt)


opt.n_epochs = 50        # longer training
opt.batch_size = 256
opt.len = 1000           # full dataset size
opt.numNN = 1

R_shape = (opt.n_rows, opt.n_cols)


def Infer_Shape(R_shape):
 
    PNL_shape_0 = R_shape[0]  # Buy & Hold per asset
    for strategy in opt.strategies:
        if strategy == 'Port':
            PNL_shape_0 += opt.n_trans
        elif strategy == 'MR':
            PNL_shape_0 += opt.n_rows * len(opt.thresholds_pct)
        elif strategy == 'TF':
            PNL_shape_0 += opt.n_rows * len(opt.thresholds_pct)
        else:
            pass
    return (PNL_shape_0, R_shape[1])

PNL_shape = Infer_Shape(R_shape)


this_version = '_'.join([
    opt.version,
    'Stk' + str(opt.n_rows),
    opt.data_name,
    opt.noise_name,
    'E' + str(opt.n_epochs),
    'N' + str(opt.len),
    'BS' + str(opt.batch_size),
    opt.static_way,
    '_'.join(opt.strategies),
    'P' + str(opt.n_trans),
    'Cap' + str(opt.Cap),
    'WH' + str(opt.WH),
    'R' + '+'.join([str(a) for a in opt.ratios]),
    'T' + '+'.join(['_'.join(map(str, i)) for i in opt.thresholds_pct]),
    'D' + str(opt.n_critic_D),
    'G' + str(opt.n_critic_G),
    'LR' + '-'.join([str(opt.lr_D), str(opt.lr_G)]),
    'Temp' + str(opt.temp),
    'Q' + '+'.join([str(int(100 * a)) for a in opt.alphas]),
    'Esb' + str(opt.numNN),
])

RUN_ROOT = join(GEN_OUT_ROOT, this_version)
os.makedirs(RUN_ROOT, exist_ok=True)

gen_data_path = join(RUN_ROOT, f"gen_data_{this_version}")
model_path = join(RUN_ROOT, f"models_{this_version}")
os.makedirs(gen_data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


def Compute_PNL(R):
  
    # convert increments to price paths starting at 1
    prices_l = Inc2Price(R)  # (B, n_rows, T)

    # Static portfolio prices (for 'Port' strategy)
    port_prices_l = StaticPort(prices_l, opt.n_trans, opt.static_way, insample=True)

    # Buy & Hold for each asset
    PNL_BH = BuyHold(prices_l, opt.Cap)  # (B, n_rows)
    PNL_l = [PNL_BH]

    for strategy in opt.strategies:
        if strategy == 'Port':
            # Buy & Hold for each static portfolio
            PNL_BHPort = BuyHold(port_prices_l, opt.Cap)  # (B, n_trans)
            PNL_l.append(PNL_BHPort)

        elif strategy == 'MR':
            # Mean Reversion for each percentile pair
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    opt.data_name,
                    opt.tickers,
                    strategy,
                    percentile_l,
                    length=100,   # number of files to estimate thresholds from
                    WH=opt.WH
                )
                # thresholds_array shape (n_rows, k)
                # We treat last two entries per row as LT / ST
                PNL_MR = MeanRev(
                    prices_l,
                    opt.Cap,
                    opt.WH,
                    LR=opt.ratios[0],
                    SR=opt.ratios[1],
                    ST=thresholds_array[:, -1],
                    LT=thresholds_array[:, -2],
                )  # (B, n_rows)
                PNL_l.append(PNL_MR)

        elif strategy == 'TF':
            # Trend Following for each percentile pair
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    opt.data_name,
                    opt.tickers,
                    strategy,
                    percentile_l,
                    length=100,
                    WH=opt.WH
                )
                # Assume thresholds_array[:,0] is ST and [:,1] is LT for TF
                PNL_TF = TrendFollow(
                    prices_l,
                    opt.Cap,
                    opt.WH,
                    LR=opt.ratios[0],
                    SR=opt.ratios[1],
                    ST=thresholds_array[:, 0],
                    LT=thresholds_array[:, 1],
                )  # (B, n_rows)
                PNL_l.append(PNL_TF)

        else:
            pass

    # concat all strategy PNL horizontally -> (B, total_rows_across_strategies)
    PNL = torch.cat(PNL_l, dim=1)
    return PNL



class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(R_shape))),
        )

    def forward(self, z):
        out = self.model(z)  # (B, n_rows*n_cols)
        out = torch.clamp(out, min=-1, max=1)
        out = out.view(out.shape[0], *R_shape)  # (B, n_rows, n_cols)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.W = opt.W
        self.project = opt.project
        self.alphas = opt.alphas

        # We'll feed a (total_rows, batch_size) tensor into this MLP.
        self.model = nn.Sequential(
            nn.Linear(opt.batch_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(opt.alphas)),
        )

    def project_op(self, validity):

        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2 * i].clone()
            e = validity[:, 2 * i + 1].clone()

            indicator = torch.sign(torch.as_tensor(0.5 - alpha, device=validity.device))
            mask = (self.W * v < e).float()
            mask_inv = 1.0 - mask

            validity[:, 2 * i] = indicator * (
                mask * v +
                mask_inv * (v + self.W * e) / (1 + self.W ** 2)
            )

            validity[:, 2 * i + 1] = indicator * (
                mask * e +
                mask_inv * (self.W * (v + self.W * e) / (1 + self.W ** 2))
            )

        return validity

    def forward(self, R):
        # R: (B, n_rows, n_cols)
        PNL = Compute_PNL(R)              # (B, total_rows)
        PNL_t = PNL.T                     # (total_rows, B)
        PNL_s = PNL_t.reshape(*PNL_t.shape, 1)  # (total_rows, B, 1)

        # differentiable "soft sort"
        perm_matrix = deterministic_NeuralSort(PNL_s, opt.temp)  # (total_rows, B, B)

        # apply soft permutation to scores
        PNL_sort = torch.bmm(perm_matrix, PNL_s)  # (total_rows, B, 1)

        # flatten across last dim to feed linear layers
        PNL_validity = self.model(PNL_sort.reshape(PNL_t.shape))  # (total_rows, 2*len(alphas))

        if self.project:
            PNL_validity = self.project_op(PNL_validity)

        return PNL, PNL_validity



def G1(v):
    return v

def G2(e, scale=1):
    return scale * torch.exp(e / scale)

def G2in(e, scale=1):
    return scale ** 2 * torch.exp(e / scale)

def G1_quant(v, W=opt.W):
    return - W * v ** 2 / 2

def G2_quant(e, alpha):
    return alpha * e

def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2


def S_stats(v, e, X, alpha):

    if alpha < 0.5:
        rt = ((X <= v).float() - alpha) * (G1(v) - G1(X)) \
             + (1.0 / alpha) * G2(e) * (X <= v).float() * (v - X) \
             + G2(e) * (e - v) - G2in(e)
    else:
        alpha_inv = 1 - alpha
        rt = ((X >= v).float() - alpha_inv) * (G1(X) - G1(v)) \
             + (1.0 / alpha_inv) * G2(-e) * (X >= v).float() * (X - v) \
             + G2(-e) * (v - e) - G2in(-e)
    return torch.mean(rt)


def S_quant(v, e, X, alpha, W=opt.W):
 
    if alpha < 0.5:
        rt = ((X <= v).float() - alpha) * (G1_quant(v, W) - G1_quant(X, W)) \
             + (1.0 / alpha) * G2_quant(e, alpha) * (X <= v).float() * (v - X) \
             + G2_quant(e, alpha) * (e - v) - G2in_quant(e, alpha)
    else:
        alpha_inv = 1 - alpha
        rt = ((X >= v).float() - alpha_inv) * (G1_quant(v, W) - G1_quant(X, W)) \
             + (1.0 / alpha_inv) * G2_quant(-e, alpha_inv) * (X >= v).float() * (X - v) \
             + G2_quant(-e, alpha_inv) * (v - e) - G2in_quant(-e, alpha_inv)
    return torch.mean(rt)


class Score(nn.Module):
 
    def __init__(self):
        super().__init__()
        self.alphas = opt.alphas
        self.score_name = opt.score
        if self.score_name == 'quant':
            self.score_alpha = S_quant
        elif self.score_name == 'stats':
            self.score_alpha = S_stats
        else:
            raise ValueError("score must be 'quant' or 'stats'")

    def forward(self, PNL_validity, PNL):
        # PNL_validity: (total_rows, 2*len(alphas))
        # PNL:          (B, total_rows)
        X = PNL.T  # (total_rows, B)

        loss = 0
        for i, alpha in enumerate(self.alphas):
            PNL_var = PNL_validity[:, [2 * i]]      # (total_rows, 1)
            PNL_es  = PNL_validity[:, [2 * i + 1]]  # (total_rows, 1)
            loss += self.score_alpha(PNL_var, PNL_es, X, alpha)
        return loss



def Train_Single(opt, dataloader, model_index, seed):
    start_time = time.time()
    torch.manual_seed(seed)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = Score().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

    loss_d_l = []
    loss_g_l = []

    gen_size = 1000  # how many synthetic samples we save each epoch

    for epoch in range(opt.n_epochs):
        epoch_loss_D = []
        epoch_loss_G = []

        for i, R in enumerate(dataloader):
            # R from Dataset_IS: (batch_size, n_rows, n_cols)
            # move batch to device, ensure float32
            real_R = R.to(device=device, dtype=torch.float32)

            # sample latent noise z
            if 't' in opt.noise_name:
                df_noise = int(opt.noise_name.split('t')[1])
                z_np = np.random.standard_t(df_noise, (R.shape[0], opt.latent_dim)).astype(np.float32)
            else:
                z_np = np.random.normal(0, 1, (R.shape[0], opt.latent_dim)).astype(np.float32)

            z = torch.from_numpy(z_np).to(device)

            # generate fake returns
            gen_R = generator(z)  # (batch_size, n_rows, n_cols)

            if i % opt.n_critic_D == 0:
                optimizer_D.zero_grad()

                PNL_real, PNL_valid_real = discriminator(real_R)
                PNL_fake, PNL_valid_fake = discriminator(gen_R)

                real_score = criterion(PNL_valid_real, PNL_real)
                fake_score = criterion(PNL_valid_fake, PNL_real)
                loss_D = real_score - fake_score

                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                epoch_loss_D.append(loss_D.item())

          
            if i % opt.n_critic_G == 0:
                optimizer_G.zero_grad()

                PNL_fake2, PNL_valid_fake2 = discriminator(gen_R)
                # generator wants fake to "score well"
                loss_G = criterion(PNL_valid_fake2, PNL_real)

                loss_G.backward()
                optimizer_G.step()

                epoch_loss_G.append(loss_G.item())

        # end batch loop

        D_loss_epoch = float(np.mean(epoch_loss_D)) if epoch_loss_D else 0.0
        G_loss_epoch = float(np.mean(epoch_loss_G)) if epoch_loss_G else 0.0

        loss_d_l.append(D_loss_epoch)
        loss_g_l.append(G_loss_epoch)

        if epoch % 100 == 0:
            elapsed = int(time.time() - start_time)
            print(f"[Epoch {epoch}] [D loss: {D_loss_epoch:.4f}] [G loss: {G_loss_epoch:.4f}]")
            print(f"--- {elapsed} seconds passed ---")

        # save generated samples every epoch
        if 't' in opt.noise_name:
            df_noise = int(opt.noise_name.split('t')[1])
            z_np = np.random.standard_t(df_noise, (gen_size, opt.latent_dim)).astype(np.float32)
        else:
            z_np = np.random.normal(0, 1, (gen_size, opt.latent_dim)).astype(np.float32)

        z_full = torch.from_numpy(z_np).to(device)
        gen_R_full = generator(z_full).detach().cpu().numpy()

        np.save(join(gen_data_path, f"Fake_id{model_index}_E{epoch}.npy"), gen_R_full)

        # checkpoint models every 100 epochs
        if epoch % 100 == 0:
            torch.save(discriminator.state_dict(), join(model_path, f"discriminator_id{model_index}_E{epoch}.pt"))
            torch.save(generator.state_dict(),      join(model_path, f"generator_id{model_index}_E{epoch}.pt"))

    # save loss curves for this model
    loss_d_arr = np.array(loss_d_l, dtype=np.float32)
    loss_g_arr = np.array(loss_g_l, dtype=np.float32)
    loss_stack = np.stack([loss_d_arr, loss_g_arr])  # shape (2, epochs)
    np.save(join(gen_data_path, f"loss_id{model_index}.npy"), loss_stack)



def Train(opt):

    dataset = Dataset_IS(
        tickers=opt.tickers,
        data_path=join(DATA_ROOT, opt.data_name),
        length=opt.len,
    )
    dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)


    for iii in range(opt.numNN):
        print(f"------ Model {iii} Starts with Random Seed {seed} ------")
        Train_Single(opt, dataloader, model_index=iii, seed=seed)



def Screen_Ensemble(thres_perc=50):

    loss_l = []
    for j in range(opt.numNN):
        loss_np = np.load(join(gen_data_path, f"loss_id{j}.npy"))
        # loss_np shape (2, epochs)
        # index 1 is generator loss; take the last epoch
        gen_last = float(loss_np[1, -1])
        loss_l.append(gen_last)

    threshold_loss = np.percentile(loss_l, thres_perc)

    select_l = []
    for j, val in enumerate(loss_l):
        if val <= threshold_loss:
            select_l.append(j)

    return select_l



if __name__ == "__main__":
    Train(opt)
    select_l = Screen_Ensemble()
    # print("Selected Models: ", select_l)
