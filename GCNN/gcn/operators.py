import torch

def hub_laplacian(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the Hub Laplacian for a single graph or a batch of graphs.
    Handles both batched ([B, N, N]) and unbatched ([N, N]) inputs.
    """
    is_batched = A.dim() == 3

    # Sum over the last dimension to get degrees
    deg_dim = 2 if is_batched else 1
    deg = A.sum(dim=deg_dim)

    # Select diagonal function based on input type
    diag_fn = torch.diag_embed if is_batched else torch.diag

    # Handle standard Laplacian case (alpha=0) efficiently
    if alpha == 0:
        return diag_fn(deg) - A

    # Power of degrees for D_alpha and D_-alpha
    # Add a small epsilon to prevent issues with zero-degree nodes
    deg_stable = deg + 1e-9
    deg_alpha = deg_stable.pow(alpha)
    deg_neg_alpha = deg_stable.pow(-alpha)

    D_alpha = diag_fn(deg_alpha)
    D_neg_alpha = diag_fn(deg_neg_alpha)

    # Compute Xi matrix
    if is_batched:
        deg_i = deg_stable.unsqueeze(2)  # [B, N, 1]
        deg_j = deg_stable.unsqueeze(1)  # [B, 1, N]
    else:
        deg_i = deg_stable.view(-1, 1)      # [N, 1]
        deg_j = deg_stable.view(1, -1)      # [1, N]

    # Ratio of degrees
    ratio = (deg_j / deg_i).pow(alpha)

    # Mask out non-edges
    deg_ratio_pow = ratio * A

    # Sum over neighbors to get the diagonal of Xi
    sum_dim = 2 if is_batched else 1
    Xi_diag = deg_ratio_pow.sum(dim=sum_dim)
    Xi = diag_fn(Xi_diag)

    # Hub Laplacian
    return Xi - D_neg_alpha @ A @ D_alpha

def adv_diff(A: torch.Tensor, alpha: float, gamma_diff: float = 1.0, gamma_adv: float = 1.0) -> torch.Tensor:
    """
    Computes the adversarial-diffusive operator for a single graph or a batch of graphs.
    """
    is_batched = A.dim() == 3

    L_diff = hub_laplacian(A, alpha=0)  # Standard Laplacian
    L_hub = hub_laplacian(A, alpha=alpha)

    # Transpose dimensions for batched vs. unbatched
    transpose_dims = (1, 2) if is_batched else (0, 1)

    return gamma_adv * L_hub + gamma_diff * torch.transpose(L_diff, *transpose_dims)


def normalized_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized adjacency matrix D^{-0.5} * (A+I) * D^{-0.5}.
    Assumes unbatched input A: [N, N].
    """
    deg = A.sum(dim=1)
    # Add a small epsilon for stability if degrees can be zero
    D_neg_half = torch.diag(torch.pow(deg + 1e-9, -0.5))
    A_hat = A + torch.eye(A.shape[0], device=A.device)
    return D_neg_half @ A_hat @ D_neg_half


def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized Laplacian I - D^{-0.5} * (A+I) * D^{-0.5}.
    Assumes unbatched input A: [N, N].
    """
    A_norm = normalized_adjacency(A)
    return torch.eye(A.shape[0], device=A.device) - A_norm


def normalized_hub_laplacian(A: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the hub laplacian on a normalized adjacency matrix.
    This was previously `turbohub_laplacian`.
    Assumes unbatched input A: [N, N].
    """
    A_normalized = normalized_adjacency(A)
    return hub_laplacian(A_normalized, alpha)