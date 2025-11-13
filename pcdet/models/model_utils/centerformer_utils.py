import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterFormerPositionEmbedding(nn.Module):
    """Learnable 2D position embedding for BEV center coordinates.

    Paper (Table 9): learnable linear encoding (+2.3 mAPH over sinusoidal).
    Paper §3.2: "We use a linear layer to encode the location of the centers
    into a position embedding."

    Matches the PositionEmbeddingLearned pattern used in dsvt_utils.py and
    transfusion_utils.py in this codebase.

    Input:  (N, 2)  — (x, y) normalized to [0, 1] in BEV space
    Output: (N, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xy_normalized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy_normalized: (N, 2)  values in [0, 1]
        Returns:
            pos_emb: (N, d_model)
        """
        return self.embedding(xy_normalized)


def gather_bev_features(
    bev_feat: torch.Tensor,
    centers_xy_norm: torch.Tensor,
    radius: int = 0,
) -> torch.Tensor:
    """Sample local BEV features around proposal centers via bilinear interpolation.

    Uses F.grid_sample so centers can be sub-pixel (continuous) coordinates.

    Args:
        bev_feat:        (B, C, H, W)
        centers_xy_norm: (B, N, 2)  in grid_sample normalized coords [-1, 1]
                         [..., 0] = x (W axis), [..., 1] = y (H axis)
        radius:          int ≥ 0; samples a (2r+1)×(2r+1) patch per center.
                         radius=0 → single-point sampling → returns (B, N, C)
                         radius>0 → patch sampling      → returns (B, N, C*(2r+1)²)
    Returns:
        feats: (B, N, C) if radius==0, else (B, N, C*(2r+1)²)
    """
    B, C, H, W = bev_feat.shape
    N = centers_xy_norm.shape[1]

    if radius == 0:
        # (B, N, 1, 2) → grid_sample → (B, C, N, 1) → (B, N, C)
        grid = centers_xy_norm.unsqueeze(2)
        sampled = F.grid_sample(bev_feat, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1)

    patch_size = 2 * radius + 1

    # pixel-aligned offsets converted to grid_sample normalized coords
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=bev_feat.device)
    offsets_x = offsets * (2.0 / W)   # one pixel = 2/W in normalized space
    offsets_y = offsets * (2.0 / H)
    grid_y, grid_x = torch.meshgrid(offsets_y, offsets_x, indexing='ij')  # (ps, ps)
    # (1, 1, ps*ps, 2)
    grid_offsets = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, patch_size * patch_size, 2)

    # broadcast: (B, N, ps*ps, 2)
    grid = centers_xy_norm.unsqueeze(2) + grid_offsets
    # flatten N and patch positions for a single grid_sample call
    grid = grid.reshape(B, N * patch_size * patch_size, 1, 2)

    sampled = F.grid_sample(bev_feat, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    # (B, C, N*ps*ps, 1) → (B, C, N, ps*ps) → (B, N, C, ps*ps) → (B, N, C*ps*ps)
    sampled = sampled.squeeze(-1).reshape(B, C, N, patch_size * patch_size)
    return sampled.permute(0, 2, 1, 3).reshape(B, N, C * patch_size * patch_size)


def build_mlp(
    dims: list,
    act: type = nn.ReLU,
    norm: type = nn.LayerNorm,
    last_norm: bool = False,
) -> nn.Sequential:
    """Build a stack of Linear → [Norm] → [Activation] layers.

    Args:
        dims:      channel sizes, e.g. [256, 512, 256]
        act:       activation class (instantiated with no args); None = no activation
        norm:      norm class applied after each linear; None = no norm
        last_norm: whether to apply norm+act after the final linear layer
    Returns:
        nn.Sequential
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if not is_last or last_norm:
            if norm is not None:
                layers.append(norm(dims[i + 1]))
            if act is not None:
                layers.append(act())
    return nn.Sequential(*layers)
