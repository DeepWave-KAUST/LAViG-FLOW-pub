###############################################################################
# Joint Evaluation Dataset (Gas + Pressure) (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Thin PyTorch dataset that loads the evaluation tensors for both modalities
#   and returns synchronized sample pairs. Used by the combined metrics scripts.
###############################################################################

from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset


# --------------------------------------------------
# Dataset definition
# --------------------------------------------------

class JointEvaluationDataset(Dataset):
    """
    Full-horizon dataset providing paired gas saturation and pressure build-up targets.
    """

    def __init__(
        self,
        gas_path: str,
        pressure_path: str,
        time_slice: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        gas = torch.load(gas_path)
        pressure = torch.load(pressure_path)

        if gas.shape != pressure.shape:
            raise ValueError(
                f"Gas targets {gas.shape} and pressure targets {pressure.shape} must share the same shape."
            )
        if gas.ndim != 5:
            raise ValueError(f"Expected tensors with shape [N, T, C, H, W], got {gas.shape}.")

        if time_slice is not None:
            gas = gas[:, time_slice]
            pressure = pressure[:, time_slice]

        self.gas = gas.to(dtype)
        self.pressure = pressure.to(dtype)

        (
            self.num_samples,
            self.num_frames,
            self.frame_channels,
            self.frame_height,
            self.frame_width,
        ) = self.gas.shape

        self.use_latents = False
        self.use_images = False
        self.num_images = 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "gas": self.gas[idx],
            "pressure": self.pressure[idx],
        }