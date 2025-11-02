
import torch


def _to_device_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:

    return x.to(device=ref.device, dtype=ref.dtype)


def deterministic_NeuralSort(s: torch.Tensor, tau: float) -> torch.Tensor:
   
    # shapes
    if s.dim() != 3 or s.size(-1) != 1:
        raise ValueError(f"deterministic_NeuralSort expects (B, n, 1), got {s.shape}")

    B, n, _ = s.shape

    # 1_n column vector
    one = torch.ones((n, 1), device=s.device, dtype=s.dtype)  # (n,1)

    # pairwise absolute diffs |s_i - s_j|
    # s: (B,n,1), s.permute(0,2,1): (B,1,n)
    # result: (B,n,n)
    A_s = torch.abs(s - s.permute(0, 2, 1))

    # B term from the NeuralSort paper:
    # B = |s_i - s_j| * 1 1^T aggregated in a certain way.
    # torch.matmul(one, one^T) -> (n,1)*(1,n) = (n,n)
    B_term = torch.matmul(one, one.t())  # (n,n)
    # matmul(A_s, B_term): (B,n,n) @ (n,n) -> (B,n,n)
    B = torch.matmul(A_s, B_term)

    # scaling vector (n,) -> (n,). largest rank gets largest score
    # scaling[k] = n+1-2*(k+1) = n+1-2k-2 = n-2k-1
    scaling = torch.arange(n, device=s.device, dtype=s.dtype)
    scaling = (n + 1 - 2 * (scaling + 1))  # (n,), e.g. [n-1, n-3, ..., -n+1]
    # reshape to (1,n,1) so we can matmul with s
    scaling = scaling.view(1, n, 1)  # (1,n,1)

   
    C = torch.matmul(s, scaling.permute(0,2,1))  # (B,n,n)

    # final scores before softmax:
    # P_max[b][i,j] ~ how much item i should be at position j
    P_max = (C - B).permute(0, 2, 1)  # (B,n,n) -> (B,n,n), swapping axes like original code

    sm = torch.nn.Softmax(dim=-1)
    P_hat = sm(P_max / tau)  # (B,n,n)
    return P_hat


def sample_gumbel(shape, device, dtype, eps=1e-10):

    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_NeuralSort(s: torch.Tensor, n_samples: int, tau: float) -> torch.Tensor:
  
    if s.dim() != 3 or s.size(-1) != 1:
        raise ValueError(f"stochastic_NeuralSort expects (B, n, 1), got {s.shape}")

    batch_size, n, _ = s.shape

    # log_s_perturb = log(s) + Gumbel noise
    # Note: if s can be <=0, log(s) will explode.
    # Original code assumes s > 0 (like Plackett–Luce params).
    log_s = torch.log(s)

    g_noise = sample_gumbel(
        shape=(n_samples, batch_size, n, 1),
        device=s.device,
        dtype=s.dtype,
    )

    log_s_perturb = log_s + g_noise  # (n_samples,B,n,1)
    log_s_perturb = log_s_perturb.view(n_samples * batch_size, n, 1)  # (n_samples*B,n,1)

    P_hat = deterministic_NeuralSort(log_s_perturb, tau)  # (n_samples*B,n,n)

    # reshape back
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat
