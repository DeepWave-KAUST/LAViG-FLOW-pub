###############################################################################
# Patch Embedding + Positional Encoding (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Patchify video frames and add 2D sinusoidal positional encodings. Adapted
#   from the Video Diffusion reference implementation:
#   https://github.com/explainingai-code/VideoGeneration-PyTorch
###############################################################################


from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange


# --------------------------------------------------
# Positional embedding helper
# --------------------------------------------------

def get_patch_position_embedding(pos_emb_dim: int, grid_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Create 2D sinusoidal positional embeddings for a patch grid."""
    if pos_emb_dim % 4 != 0:
        raise ValueError("Position embedding dimension must be divisible by 4.")
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)

    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    div_term = torch.arange(0, pos_emb_dim // 4, dtype=torch.float32, device=device)
    div_term = torch.pow(10000.0, div_term / (pos_emb_dim // 4))

    grid_h_emb = grid_h_positions[:, None] / div_term
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)

    grid_w_emb = grid_w_positions[:, None] / div_term
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)

    return torch.cat([grid_h_emb, grid_w_emb], dim=-1)


# --------------------------------------------------
# Patch embedding module
# --------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert frames to patch tokens with learnable projection + fixed positional encodings."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        im_channels: int,
        patch_height: int,
        patch_width: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels
        self.hidden_size = hidden_size
        self.patch_height = patch_height
        self.patch_width = patch_width

        patch_dim = im_channels * patch_height * patch_width
        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width

        tokens = rearrange(
            x,
            "b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)",
            ph=self.patch_height,
            pw=self.patch_width,
        )
        tokens = self.patch_embed(tokens)

        pos_embed = get_patch_position_embedding(
            pos_emb_dim=self.hidden_size,
            grid_size=(grid_size_h, grid_size_w),
            device=x.device,
        )
        tokens += pos_embed
        return tokens