###############################################################################
# Pressure Build-Up dataset utilities (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

TensorPath = Union[str, Path]


# --------------------------------------------------
# Dataclass: describe dataset split filepaths
# --------------------------------------------------
@dataclass
class dPDatasetPaths:
    """
    Container for a dataset split.

    Parameters
    ----------
    target:
        Path to the tensor with the (already normalised) pressure buildup targets.
    condition:
        Path to the tensor with the conditioning inputs (porosity, or another feature).
        When omitted the dataset will hand back the targets as both features and labels.
    stats:
        Optional path to the dictionary of per-sample statistics produced during dataset
        preparation (e.g. `dP_train_min_max.pt`). These are useful for denormalisation.
    """

    target: Path
    condition: Optional[Path] = None
    stats: Optional[Path] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "dPDatasetPaths":
        return cls(
            target=Path(cfg["target"]),
            condition=Path(cfg["condition"]) if cfg.get("condition") else None,
            stats=Path(cfg["stats"]) if cfg.get("stats") else None,
        )


# --------------------------------------------------
# Dataset: thin wrapper around prepared tensors
# --------------------------------------------------
class PressureBuildUpDataset(Dataset):
    """
    Thin wrapper around the preprocessed tensors emitted by `dP_dataset_preparation.py`.
    """

    def __init__(
        self,
        *,
        target_path: TensorPath,
        condition_path: Optional[TensorPath] = None,
        stats_path: Optional[TensorPath] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.target_path = Path(target_path)
        self.condition_path = Path(condition_path) if condition_path is not None else None
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.dtype = dtype

        self.targets = self._load_tensor(self.target_path)
        if self.condition_path is None:
            self.conditions = self.targets.clone()
        else:
            self.conditions = self._load_tensor(self.condition_path)

        # Ensure tensors have a channel dimension
        if self.targets.ndim == 3:
            self.targets = self.targets.unsqueeze(1)
        if self.conditions.ndim == 3:
            self.conditions = self.conditions.unsqueeze(1)

        if self.targets.shape != self.conditions.shape:
            raise ValueError(
                "Targets and conditions must share the same shape after loading. "
                f"Got targets={tuple(self.targets.shape)}, conditions={tuple(self.conditions.shape)}."
            )

        self._stats = (
            torch.load(self.stats_path, map_location="cpu") if self.stats_path is not None else None
        )

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.targets[index], self.conditions[index]

    @property
    def stats(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Returns the metadata dictionary saved during preprocessing (if available).

        Contents mirror the output of `save_per_sample_stats` in
        `co2_dataset_preparation.py`, namely:
            - `per_sample_min`
            - `per_sample_max`
            - `index_map`
            - `frames_per_clip`
        """
        return self._stats

    def _load_tensor(self, path: Path) -> torch.Tensor:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"File {path} did not contain a torch.Tensor (got {type(tensor)}).")
        return tensor.to(self.dtype)


# --------------------------------------------------
# Factory: instantiate dataset from config dict
# --------------------------------------------------
def load_dP_dataset(split_cfg: Dict[str, Any], dtype: torch.dtype = torch.float32) -> PressureBuildUpDataset:
    """
    Convenience helper that accepts a dict (sourced from YAML/JSON) and returns the dataset.

    Example config section:
    -----------------------
    train:
      target: /.../dP_data/train_dP_target.pt
      condition: /.../dP_data/train_dP_condition.pt
      stats: /.../dP_data/training_dataset_pressure_buildup/dP_train_min_max.pt
    """
    paths = dPDatasetPaths.from_dict(split_cfg)
    return PressureBuildUpDataset(
        target_path=paths.target,
        condition_path=paths.condition,
        stats_path=paths.stats,
        dtype=dtype,
    )