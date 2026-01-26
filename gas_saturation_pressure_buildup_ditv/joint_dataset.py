###############################################################################
# Joint Gas + Pressure Targets Dataset (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Thin PyTorch Dataset wrapper that loads synchronized gas saturation and
#   pressure build-up tensors, optionally slices them into autoregressive
#   context/target windows, and exposes samples compatible with the DiTV
#   training/evaluation pipelines.
###############################################################################

import torch
from torch.utils.data import Dataset
from typing import Optional, Sequence, Dict, Any


# --------------------------------------------------
# Dataset definition
# --------------------------------------------------

class JointTargetsDataset(Dataset):
    """
    Dataset that returns paired targets for gas saturation and pressure build-up.

    Each target tensor is expected to have shape [N, T, C, H, W] and already be
    normalized to [-1, 1] in the raw data space (matching the single-task setup).
    """

    def __init__(
        self,
        gas_path: str,
        pressure_path: str,
        time_slice: Optional[Sequence[int]] = None,
        autoregressive: Optional[Dict[str, Any]] = None,
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

        self.autoregressive_cfg = None
        self.indices = None
        if autoregressive is not None:
            context_frames = int(autoregressive["context_frames"])
            predict_frames = int(autoregressive["predict_frames"])
            stride = int(autoregressive.get("window_stride", 1))
            drop_incomplete = bool(autoregressive.get("drop_incomplete", True))

            assert context_frames > 0 and predict_frames > 0, "context/predict must be > 0"
            assert stride > 0, "window_stride must be > 0"

            total_window = context_frames + predict_frames
            indices = []
            for sample_idx in range(self.num_samples):
                seq_len = self.gas[sample_idx].shape[0]
                max_start = seq_len - total_window
                if max_start < 0:
                    if not drop_incomplete and seq_len > context_frames:
                        indices.append((sample_idx, 0))
                    continue
                for start in range(0, max_start + 1, stride):
                    indices.append((sample_idx, start))

            if not indices:
                raise ValueError("No valid windows found for autoregressive configuration.")

            self.autoregressive_cfg = {
                "context_frames": context_frames,
                "predict_frames": predict_frames,
                "stride": stride,
                "drop_incomplete": drop_incomplete,
                "total_window": total_window,
            }
            self.indices = indices

        # Match the flags used in the existing single-task datasets so the training
        # pipeline can make the same assumptions.
        self.use_latents = False
        self.use_images = False
        self.num_images = 0
        if self.autoregressive_cfg:
            self.context_frames = self.autoregressive_cfg["context_frames"]
            self.predict_frames = self.autoregressive_cfg["predict_frames"]
            self.total_window = self.autoregressive_cfg["total_window"]
            self.num_frames = self.total_window

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.indices is None:
            return {
                "gas": self.gas[idx],
                "pressure": self.pressure[idx],
            }

        sample_idx, start = self.indices[idx]
        cfg = self.autoregressive_cfg
        context_end = start + cfg["context_frames"]
        target_end = context_end + cfg["predict_frames"]

        gas_window = self.gas[sample_idx, start:target_end]
        pressure_window = self.pressure[sample_idx, start:target_end]

        context = {
            "gas": gas_window[: cfg["context_frames"]],
            "pressure": pressure_window[: cfg["context_frames"]],
        }
        target = {
            "gas": gas_window[cfg["context_frames"] :],
            "pressure": pressure_window[cfg["context_frames"] :],
        }

        return {"context": context, "target": target}