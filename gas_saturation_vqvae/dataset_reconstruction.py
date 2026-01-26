###############################################################################
# CO₂ dataset reconstruction helpers (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
###############################################################################

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from vqvae import VQVAE
from utils import custom_r2_score


PROJECT_ROOT = Path(__file__).resolve().parent


try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as exc:  
    raise RuntimeError(
        "scikit-learn is required to compute reconstruction metrics. "
        "Install it with `pip install scikit-learn`."
    ) from exc

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError as exc: 
    raise RuntimeError(
        "scikit-image is required to compute the SSIM metric. "
        "Install it with `pip install scikit-image`."
    ) from exc


@dataclass
class SplitSpec:
    """Container describing artefacts for a single dataset split."""

    name: str
    target_path: Path
    inputs_path: Path
    stats_path: Path
    metrics_filename: str
    mosaic_prefix: str
    feature_index: int = 2  # porosity feature index inside the raw tensor


@dataclass
class ReconstructionAssets:
    """Pre-loaded tensors and metadata required for reconstruction."""

    targets: torch.Tensor
    conditions: torch.Tensor
    per_sample_min: torch.Tensor
    per_sample_max: torch.Tensor
    porosity_min: float
    porosity_max: float
    spatial_shape: Tuple[int, int]
    frames_per_clip: int


SPLIT_SPECS: Dict[str, SplitSpec] = {
    "train": SplitSpec(
        name="train",
        target_path=Path("co2_data/train_sg_target.pt"),
        inputs_path=Path("co2_data/training_dataset_gas_saturation/sg_train_inputs.pt"),
        stats_path=Path("co2_data/training_dataset_gas_saturation/sg_train_min_max.pt"),
        metrics_filename="vqvae_train_co2_average_metrics.txt",
        mosaic_prefix="train",
    ),
    "validation": SplitSpec(
        name="validation",
        target_path=Path("co2_data/val_sg_target.pt"),
        inputs_path=Path("co2_data/validation_dataset_gas_saturation/sg_val_inputs.pt"),
        stats_path=Path("co2_data/validation_dataset_gas_saturation/sg_val_min_max.pt"),
        metrics_filename="vqvae_validation_co2_average_metrics.txt",
        mosaic_prefix="val",
    ),
    "test": SplitSpec(
        name="test",
        target_path=Path("co2_data/test_sg_target.pt"),
        inputs_path=Path("co2_data/test_dataset_gas_saturation/sg_test_inputs.pt"),
        stats_path=Path("co2_data/test_dataset_gas_saturation/sg_test_min_max.pt"),
        metrics_filename="vqvae_test_co2_average_metrics.txt",
        mosaic_prefix="test",
    ),
}


# --------------------------------------------------
# Utility: resolve project-relative paths
# --------------------------------------------------
def make_absolute(path: Path) -> Path:
    """Resolve paths relative to the project root when needed."""
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------
# Utility: pick available compute device
# --------------------------------------------------
def resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve the torch device respecting an optional user override."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - M1 friendly
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------
# IO: load YAML configuration from disk
# --------------------------------------------------
def load_yaml_config(path: Path) -> Dict:
    path = make_absolute(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------
# Model: instantiate VQ-VAE and restore weights
# --------------------------------------------------
def build_model(config: Dict, checkpoint: Path, device: torch.device) -> VQVAE:
    """Instantiate the VQ-VAE architecture and load weights from checkpoint."""
    model_cfg = config.get("model")
    if model_cfg is None:
        raise KeyError("Missing `model` section in configuration.")
    im_channels = model_cfg.get("im_channels")
    auto_cfg = model_cfg.get("autoencoder")
    if im_channels is None or auto_cfg is None:
        raise KeyError("`model.im_channels` or `model.autoencoder` missing in configuration.")

    model = VQVAE(im_channels=im_channels, autoencoder_model_config=auto_cfg)

    checkpoint = make_absolute(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint {checkpoint} does not contain a valid state_dict (found {type(state_dict)})."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Preprocessing: flatten porosity feature frames
# --------------------------------------------------
def _extract_porosity_views(raw_inputs: torch.Tensor, feature_index: int, frames: int) -> torch.Tensor:
    """
    Extract the porosity feature, keep the first `frames` temporal slices,
    and flatten B x T into a single sample dimension.
    """
    if raw_inputs.ndim not in (4, 5):
        raise ValueError(
            "Expected raw inputs tensor with 4 or 5 dimensions "
            f"(got shape={tuple(raw_inputs.shape)})."
        )

    tensor = raw_inputs
    if tensor.ndim == 5:
        if feature_index >= tensor.shape[-1]:
            raise IndexError(
                f"Feature index {feature_index} out of bounds for raw inputs "
                f"with last-dimension size {tensor.shape[-1]}."
            )
        tensor = tensor[..., feature_index]

    if frames > tensor.shape[-1]:
        raise ValueError(
            f"Requested {frames} temporal frames but raw inputs only expose {tensor.shape[-1]}."
        )

    tensor = tensor[..., :frames]  # [..., T]
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # (B, T, H, W)
    b, t, h, w = tensor.shape
    return tensor.view(b * t, 1, h, w)


# --------------------------------------------------
# Preprocessing: load targets/conditions/statistics
# --------------------------------------------------
def load_split_assets(spec: SplitSpec) -> ReconstructionAssets:
    """Load tensors for the requested dataset split."""
    target_path = make_absolute(spec.target_path)
    stats_path = make_absolute(spec.stats_path)
    inputs_path = make_absolute(spec.inputs_path)

    targets = torch.load(target_path, map_location="cpu").float()
    stats = torch.load(stats_path, map_location="cpu")

    if "per_sample_min" not in stats or "per_sample_max" not in stats:
        raise KeyError(
            f"Stats file {spec.stats_path} is missing `per_sample_min` or `per_sample_max`."
        )
    per_sample_min = stats["per_sample_min"].float()
    per_sample_max = stats["per_sample_max"].float()
    frames = int(stats.get("frames_per_clip", 1))

    raw_inputs = torch.load(inputs_path, map_location="cpu").float()
    conditions = _extract_porosity_views(raw_inputs, spec.feature_index, frames)
    porosity_min = float(conditions.min().item())
    porosity_max = float(conditions.max().item())

    if math.isclose(porosity_min, porosity_max):
        raise ValueError(
            f"Porosity min/max collapse to a single value ({porosity_min}); "
            "cannot perform denormalisation."
        )

    conditions = 2.0 * (conditions - porosity_min) / (porosity_max - porosity_min) - 1.0

    if targets.shape != conditions.shape:
        raise ValueError(
            "Targets and conditions shapes differ after preprocessing: "
            f"{tuple(targets.shape)} vs {tuple(conditions.shape)}."
        )
    if per_sample_min.numel() != targets.shape[0] or per_sample_max.numel() != targets.shape[0]:
        raise ValueError(
            "Per-sample min/max entries do not match number of flattened samples."
        )

    spatial_shape = (targets.shape[2], targets.shape[3])

    return ReconstructionAssets(
        targets=targets,
        conditions=conditions,
        per_sample_min=per_sample_min,
        per_sample_max=per_sample_max,
        porosity_min=porosity_min,
        porosity_max=porosity_max,
        spatial_shape=spatial_shape,
        frames_per_clip=frames,
    )


# --------------------------------------------------
# Math: map [-1,1] field back to data range
# --------------------------------------------------
def denormalize_field(normalized: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """Bring a [-1, 1] normalised field back to its original range."""
    return (normalized + 1.0) * 0.5 * (max_value - min_value) + min_value


# --------------------------------------------------
# Preprocessing: mask fields and compute thickness
# --------------------------------------------------
def compute_masked_views(
    condition: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    porosity_min: float,
    porosity_max: float,
    saturation_min: float,
    saturation_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Denormalise tensors, apply the porosity mask, and return masked views."""
    denorm_cond = denormalize_field(condition, porosity_min, porosity_max)
    denorm_orig = denormalize_field(original, saturation_min, saturation_max)
    denorm_recon = denormalize_field(reconstructed, saturation_min, saturation_max)

    mask = denorm_cond[0] != 0
    thickness = int(mask[:, 0].sum())
    if thickness <= 0:
        raise ValueError("Encountered porosity mask with zero thickness.")

    masked_original = denorm_orig[0][mask].reshape(thickness, -1)
    masked_reconstructed = denorm_recon[0][mask].reshape(thickness, -1)
    masked_error = np.abs(masked_original - masked_reconstructed)
    return masked_original, masked_reconstructed, masked_error, thickness


# --------------------------------------------------
# Metrics: compute error statistics per sample
# --------------------------------------------------
def compute_sample_metrics(
    masked_original: np.ndarray,
    masked_reconstructed: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """Compute numerical metrics between original and reconstructed fields."""
    eps = 1e-6

    flat_orig = masked_original.flatten()
    flat_recon = masked_reconstructed.flatten()

    nonzero_mask = flat_orig != 0
    if not np.any(nonzero_mask):
        raise ValueError("All masked elements are zero; metrics would be ill-defined.")

    mae = mean_absolute_error(flat_orig[nonzero_mask], flat_recon[nonzero_mask])
    mse = mean_squared_error(flat_orig[nonzero_mask], flat_recon[nonzero_mask])
    rmse = math.sqrt(mse)

    smape = np.mean(
        2.0 * np.abs(masked_original - masked_reconstructed)
        / (np.abs(masked_original) + np.abs(masked_reconstructed) + eps)
    ) * 100.0

    r2 = custom_r2_score(masked_original, masked_reconstructed)

    data_range = masked_original.max() - masked_original.min()
    if np.isclose(data_range, 0.0):
        ssim_score = 1.0 if np.allclose(masked_original, masked_reconstructed) else 0.0
    else:
        ssim_score = ssim(masked_original, masked_reconstructed, data_range=data_range)

    return mae, mse, rmse, r2, smape, ssim_score


# --------------------------------------------------
# Plotting: generate non-uniform spatial grid
# --------------------------------------------------
def build_xy_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the non-uniform longitudinal grid used in the notebooks."""
    dx = np.cumsum(3.5938 * np.power(1.035012, np.arange(width))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=height))
    return X, Y


# --------------------------------------------------
# Plotting: pad arrays up to shared size
# --------------------------------------------------
def pad_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad an array with NaNs up to the target spatial shape."""
    if arr.shape == (target_h, target_w):
        return arr
    out = np.full((target_h, target_w), np.nan, dtype=arr.dtype)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


# --------------------------------------------------
# Plotting: save colorbar reference grid
# --------------------------------------------------
def save_colorbar_grid(
    colorbar_dir: Optional[Path],
    prefix: str,
    identifier: str,
    original_ranges: List[Tuple[float, float]],
    reconstructed_ranges: List[Tuple[float, float]],
    error_ranges: List[Tuple[float, float]],
    cmap_targets,
    cmap_error,
) -> None:
    """Render a grid of colorbars (original, reconstructed, errors) for the given mosaic chunk."""
    if colorbar_dir is None or not original_ranges:
        return

    colorbar_dir.mkdir(parents=True, exist_ok=True)

    cols = len(original_ranges)
    fig_width = max(3.0, 1.0 * cols)
    rows = [
        ("Original (-)", original_ranges, cmap_targets),
        ("Reconstructed (-)", reconstructed_ranges, cmap_targets),
        ("Error (-)", error_ranges, cmap_error),
    ]

    fig, axes = plt.subplots(
        len(rows),
        cols,
        figsize=(fig_width, 4.5),
        squeeze=False,
    )

    for row_idx, (label, ranges, cmap) in enumerate(rows):
        for col_idx, (vmin, vmax) in enumerate(ranges):
            ax = axes[row_idx, col_idx]
            cb = matplotlib.colorbar.ColorbarBase(
                ax,
                cmap=cmap,
                norm=colors.Normalize(vmin=vmin, vmax=vmax),
                orientation="vertical",
            )
            cb.ax.tick_params(labelsize=8)
            title_prefix = ordinal(col_idx + 1)
            ax.set_title(f"{title_prefix} {label}", fontsize=5, fontweight="bold", pad=6)

    for ax in axes.flatten():
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path = colorbar_dir / f"{prefix}_{identifier}_colorbars.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# --------------------------------------------------
# Plotting: render flattened-sample mosaics
# --------------------------------------------------
def render_mosaic_pages(
    samples: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]],
    output_dir: Path,
    prefix: str,
    cols_per_page: int,
    cmap_targets: str = "jet",
    cmap_error: str = "magma",
    error_vmax: float = 0.25,
    colorbar_dir: Optional[Path] = None,
) -> None:
    """Render 3-row mosaics (original, reconstruction, absolute error)."""
    if not samples:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = samples[0][1].shape
    X, Y = build_xy_grid(height, width)

    pages = (len(samples) + cols_per_page - 1) // cols_per_page
    targets_cmap = matplotlib.cm.get_cmap(cmap_targets).copy()
    targets_cmap.set_bad(color="white")
    error_cmap = matplotlib.cm.get_cmap(cmap_error).copy()
    error_cmap.set_bad(color="white")

    for page_idx in range(pages):
        chunk = samples[page_idx * cols_per_page : (page_idx + 1) * cols_per_page]
        if not chunk:
            continue

        max_thickness = max(item[4] for item in chunk)
        width = chunk[0][1].shape[1]

        fig, axes = plt.subplots(3, len(chunk), figsize=(len(chunk) * 3.5, 6.0), facecolor="white")
        fig.patch.set_facecolor("white")
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(3, 1)

        orig_ranges: List[Tuple[float, float]] = []
        recon_ranges: List[Tuple[float, float]] = []
        error_ranges: List[Tuple[float, float]] = []

        for col_idx, (sample_idx, original, reconstructed, error, thickness) in enumerate(chunk):
            orig_img = pad_to(original[:thickness, :], max_thickness, width)
            recon_img = pad_to(reconstructed[:thickness, :], max_thickness, width)
            err_img = pad_to(error[:thickness, :], max_thickness, width)

            orig_min = float(np.nanmin(original[:thickness, :]))
            orig_max = float(np.nanmax(original[:thickness, :]))
            if not np.isfinite(orig_min):
                orig_min = 0.0
            if not np.isfinite(orig_max):
                orig_max = orig_min
            if math.isclose(orig_min, orig_max):
                orig_max = orig_min + 1e-6
            orig_ranges.append((orig_min, orig_max))

            recon_min = float(np.nanmin(reconstructed[:thickness, :]))
            recon_max = float(np.nanmax(reconstructed[:thickness, :]))
            if not np.isfinite(recon_min):
                recon_min = 0.0
            if not np.isfinite(recon_max):
                recon_max = recon_min
            if math.isclose(recon_min, recon_max):
                recon_max = recon_min + 1e-6
            recon_ranges.append((recon_min, recon_max))

            error_ranges.append((0.0, max(error_vmax, 1e-6)))

            axes[0, col_idx].pcolor(
                X[:max_thickness, :],
                Y[:max_thickness, :],
                np.flipud(orig_img),
                shading="auto",
                cmap=targets_cmap,
                vmin=0.0,
                vmax=1.0,
            )
            axes[0, col_idx].set_facecolor("white")
            axes[0, col_idx].set_title(ordinal(col_idx + 1), fontsize=9, fontweight="bold", pad=6)
            if col_idx == 0:
                axes[0, col_idx].text(
                    -0.05,
                    0.5,
                    "Original",
                    transform=axes[0, col_idx].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )
            axes[1, col_idx].pcolor(
                X[:max_thickness, :],
                Y[:max_thickness, :],
                np.flipud(recon_img),
                shading="auto",
                cmap=targets_cmap,
                vmin=0.0,
                vmax=1.0,
            )
            axes[1, col_idx].set_facecolor("white")
            if col_idx == 0:
                axes[1, col_idx].text(
                    -0.05,
                    0.5,
                    "Reconstructed",
                    transform=axes[1, col_idx].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )
            axes[2, col_idx].pcolor(
                X[:max_thickness, :],
                Y[:max_thickness, :],
                np.flipud(err_img),
                shading="auto",
                cmap=error_cmap,
                vmin=0.0,
                vmax=error_vmax,
            )
            if col_idx == 0:
                axes[2, col_idx].text(
                    -0.05,
                    0.5,
                    "Error",
                    transform=axes[2, col_idx].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )
            for row in range(3):
                axes[row, col_idx].set_xticks([])
                axes[row, col_idx].set_yticks([])
                axes[row, col_idx].set_xlim([0, 1500])
                axes[row, col_idx].set_ylim([0, 200])
                axes[row, col_idx].set_facecolor("white")
                axes[row, col_idx].spines["top"].set_visible(False)
                axes[row, col_idx].spines["bottom"].set_visible(False)
                axes[row, col_idx].spines["left"].set_visible(False)
                axes[row, col_idx].spines["right"].set_visible(False)

        plt.subplots_adjust(wspace=0.005, hspace=0.009)
        page_name = f"mosaic_page_{page_idx + 1:03d}"
        output_path = output_dir / f"{prefix}_{page_name}.png"
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        save_colorbar_grid(
            colorbar_dir,
            prefix,
            page_name,
            orig_ranges,
            recon_ranges,
            error_ranges,
            targets_cmap,
            error_cmap,
        )


# --------------------------------------------------
# Plotting: render per-clip mosaic pages
# --------------------------------------------------
def render_clip_mosaics(
    clips: List[Tuple[int, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]]],
    output_dir: Path,
    prefix: str,
    cmap_targets: str = "jet",
    cmap_error: str = "magma",
    error_vmax: float = 0.25,
    colorbar_dir: Optional[Path] = None,
) -> None:
    """Render mosaics where each PNG corresponds to a full clip (all frames)."""
    if not clips:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    targets_cmap = matplotlib.cm.get_cmap(cmap_targets).copy()
    targets_cmap.set_bad(color="white")
    error_cmap = matplotlib.cm.get_cmap(cmap_error).copy()
    error_cmap.set_bad(color="white")

    for clip_idx, frames in clips:
        frames_sorted = sorted(frames, key=lambda item: item[0])
        frame_count = len(frames_sorted)
        if frame_count == 0:
            continue

        height, width = frames_sorted[0][1].shape
        max_thickness = max(frame[4] for frame in frames_sorted)
        X, Y = build_xy_grid(height, width)

        fig, axes = plt.subplots(3, frame_count, figsize=(frame_count * 3.5, 6.0))
        fig.patch.set_facecolor("white")
        axes = np.asarray(axes)
        if frame_count == 1:
            axes = axes.reshape(3, 1)

        orig_ranges: List[Tuple[float, float]] = []
        recon_ranges: List[Tuple[float, float]] = []
        error_ranges: List[Tuple[float, float]] = []

        for col_idx, (_, original, reconstructed, error, thickness) in enumerate(frames_sorted):
            orig_img = pad_to(original[:thickness, :], max_thickness, width)
            recon_img = pad_to(reconstructed[:thickness, :], max_thickness, width)
            err_img = pad_to(error[:thickness, :], max_thickness, width)

            orig_min = float(np.nanmin(original[:thickness, :]))
            orig_max = float(np.nanmax(original[:thickness, :]))
            if not np.isfinite(orig_min):
                orig_min = 0.0
            if not np.isfinite(orig_max):
                orig_max = orig_min
            if math.isclose(orig_min, orig_max):
                orig_max = orig_min + 1e-6
            orig_ranges.append((orig_min, orig_max))

            recon_min = float(np.nanmin(reconstructed[:thickness, :]))
            recon_max = float(np.nanmax(reconstructed[:thickness, :]))
            if not np.isfinite(recon_min):
                recon_min = 0.0
            if not np.isfinite(recon_max):
                recon_max = recon_min
            if math.isclose(recon_min, recon_max):
                recon_max = recon_min + 1e-6
            recon_ranges.append((recon_min, recon_max))

            error_ranges.append((0.0, max(error_vmax, 1e-6)))

            axes[0, col_idx].pcolor(
                X[:max_thickness, :],
                Y[:max_thickness, :],
                np.flipud(orig_img),
                shading="auto",
                cmap=targets_cmap,
                vmin=0.0,
                vmax=1.0,
            )
            axes[0, col_idx].set_facecolor("white")
            axes[0, col_idx].set_title(ordinal(col_idx + 1), fontsize=9, fontweight="bold", pad=6)
            if col_idx == 0:
                axes[0, col_idx].text(
                    -0.05,
                    0.5,
                    "Original",
                    transform=axes[0, col_idx].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )

            axes[1, col_idx].pcolor(
                X[:max_thickness, :],
                Y[:max_thickness, :],
                np.flipud(recon_img),
                shading="auto",
                cmap=targets_cmap,
                vmin=0.0,
                vmax=1.0,
            )
            axes[1, col_idx].set_facecolor("white")
            if col_idx == 0:
                axes[1, col_idx].text(
                    -0.05,
                    0.5,
                    "Reconstructed",
                    transform=axes[1, col_idx].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )

            axes[2, col_idx].pcolor(
                X[:max_thickness, :],
                Y[:max_thickness, :],
                np.flipud(err_img),
                shading="auto",
                cmap=error_cmap,
                vmin=0.0,
                vmax=error_vmax,
            )
            axes[2, col_idx].set_facecolor("white")
            if col_idx == 0:
                axes[2, col_idx].text(
                    -0.05,
                    0.5,
                    "Error",
                    transform=axes[2, col_idx].transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )

            for row in range(3):
                axes[row, col_idx].set_xticks([])
                axes[row, col_idx].set_yticks([])
                axes[row, col_idx].set_xlim([0, 1500])
                axes[row, col_idx].set_ylim([0, 200])
                axes[row, col_idx].set_facecolor("white")
                axes[row, col_idx].spines["top"].set_visible(False)
                axes[row, col_idx].spines["bottom"].set_visible(False)
                axes[row, col_idx].spines["left"].set_visible(False)
                axes[row, col_idx].spines["right"].set_visible(False)

        plt.subplots_adjust(wspace=0.005, hspace=0.009)
        clip_name = f"{clip_idx + 1:04d}"
        output_path = output_dir / f"{prefix}_{clip_name}.png"
        plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        save_colorbar_grid(
            colorbar_dir,
            prefix,
            clip_name,
            orig_ranges,
            recon_ranges,
            error_ranges,
            targets_cmap,
            error_cmap,
        )


# --------------------------------------------------
# Evaluation: reconstruct samples and collect analytics
# --------------------------------------------------
def evaluate_split(
    model: VQVAE,
    assets: ReconstructionAssets,
    device: torch.device,
    eval_indices: Sequence[int],
    sample_mosaic_indices: Sequence[int],
    clip_mosaic_indices: Sequence[int],
) -> Tuple[
    Dict[str, float],
    List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]],
    List[Tuple[int, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]]],
    List[int],
]:
    """Run reconstructions, collect metrics, and stash data for mosaic rendering."""

    sample_set = set(sample_mosaic_indices)
    clip_set = set(clip_mosaic_indices)
    frames_per_clip = assets.frames_per_clip

    mosaic_payload: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    clip_storage: Dict[int, List[Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]]] = {
        clip_idx: [None] * frames_per_clip for clip_idx in clip_set
    }
    skipped_samples: List[int] = []

    mae_vals: List[float] = []
    mse_vals: List[float] = []
    rmse_vals: List[float] = []
    r2_vals: List[float] = []
    smape_vals: List[float] = []
    ssim_vals: List[float] = []

    model.eval()
    for idx in tqdm(eval_indices, desc="Reconstructing", unit="sample"):
        clip_idx = idx // frames_per_clip
        frame_pos = idx % frames_per_clip
        try:
            data = assets.targets[idx : idx + 1].to(device)
            with torch.inference_mode():
                reconstructed, _, _ = model(data)
            original_np = data[0].cpu().numpy()
            reconstructed_np = reconstructed[0].cpu().numpy()
            condition_np = assets.conditions[idx].cpu().numpy()

            masked_original, masked_reconstructed, masked_error, thickness = compute_masked_views(
                condition_np,
                original_np,
                reconstructed_np,
                assets.porosity_min,
                assets.porosity_max,
                float(assets.per_sample_min[idx].item()),
                float(assets.per_sample_max[idx].item()),
            )

            mae, mse, rmse, r2, smape, ssim_score = compute_sample_metrics(
                masked_original,
                masked_reconstructed,
            )

            mae_vals.append(mae)
            mse_vals.append(mse)
            rmse_vals.append(rmse)
            r2_vals.append(r2)
            smape_vals.append(smape)
            ssim_vals.append(ssim_score)

            if idx in sample_set:
                mosaic_payload.append((idx, masked_original, masked_reconstructed, masked_error, thickness))

            if clip_idx in clip_storage:
                clip_storage[clip_idx][frame_pos] = (
                    frame_pos,
                    masked_original,
                    masked_reconstructed,
                    masked_error,
                    thickness,
                )

        except Exception:  # pragma: no cover - runtime guard for user runs
            skipped_samples.append(idx)

    if not mae_vals:
        raise RuntimeError("No valid samples were processed; aborting metric aggregation.")

    clip_payload: List[Tuple[int, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]]] = []
    incomplete_clips: List[int] = []
    for clip_idx, frames in clip_storage.items():
        complete_frames = [frame for frame in frames if frame is not None]
        if len(complete_frames) != frames_per_clip:
            incomplete_clips.append(clip_idx)
            continue
        ordered = sorted(complete_frames, key=lambda item: item[0])
        clip_payload.append((clip_idx, ordered))

    if incomplete_clips:
        preview = ", ".join(str(c + 1) for c in incomplete_clips[:5])
        print(
            f"Warning: skipped {len(incomplete_clips)} clip(s) for mosaic rendering due to missing frames "
            f"(first few indices: {preview})."
        )

    rng = 1.0  # saturation maps are normalised to [0, 1]

    metrics = {
        "average_mae": float(np.mean(mae_vals) / (rng + 1e-8)),
        "average_mse": float(np.mean(mse_vals) / (rng + 1e-8) ** 2),
        "average_rmse": float(np.mean(rmse_vals) / (rng + 1e-8)),
        "average_r2": float(np.mean(r2_vals)),
        "average_smape": float(np.mean(smape_vals)),
        "average_ssim": float(np.mean(ssim_vals)),
        "evaluated_samples": len(mae_vals),
    }
    return metrics, mosaic_payload, clip_payload, skipped_samples


# --------------------------------------------------
# IO: persist aggregate metrics
# --------------------------------------------------
def write_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    """Persist aggregated metrics to a plain-text file."""
    lines = [
        "====== AVERAGE METRICS ACROSS ALL SAMPLES ======",
        f"Average MAE: {metrics['average_mae']:.4f}",
        f"Average MSE: {metrics['average_mse']:.4f}",
        f"Average RMSE: {metrics['average_rmse']:.4f}",
        f"Average R²: {metrics['average_r2']:.4f}",
        f"Average sMAPE: {metrics['average_smape']:.2f}%",
        f"Average SSIM: {metrics['average_ssim']:.4f}",
        f"Evaluated Samples: {metrics['evaluated_samples']}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# --------------------------------------------------
# CLI: parse shared reconstruction arguments
# --------------------------------------------------
def parse_common_args(default_split: str, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments shared by all per-split entry points."""
    parser = argparse.ArgumentParser(
        description="Generate CO₂ gas saturation reconstructions, mosaics, and metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("trained_models/used_config.yaml"),
        help="Path to the YAML config used for training (default: trained_models/used_config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("trained_models/checkpoints/last.pt"),
        help="Checkpoint to load for inference (default: trained_models/checkpoints/last.pt).",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_SPECS.keys()),
        default=default_split,
        help=f"Dataset split to process (default: {default_split}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where artefacts will be stored. "
        "Defaults to results/reconstructions/<split>/ if omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. cpu, cuda, cuda:1, mps).",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Index of the first flattened sample to evaluate (default: 0).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of flattened samples to evaluate. "
        "Defaults to all remaining samples after `--sample-offset`.",
    )
    parser.add_argument(
        "--mosaic-count",
        type=int,
        default=17,
        help="Number of samples to include across mosaic pages (default: 17).",
    )
    parser.add_argument(
        "--mosaic-cols",
        type=int,
        default=17,
        help="Number of columns per mosaic page (default: 17).",
    )
    parser.add_argument(
        "--mosaic-error-vmax",
        type=float,
        default=0.25,
        help="Upper bound for the colour scale of the absolute-error mosaic row (default: 0.25).",
    )
    parser.add_argument(
        "--mosaic-mode",
        choices=("clip", "sample"),
        default="clip",
        help="Mosaic layout: 'clip' groups all 17 frames per simulation into one PNG, "
        "'sample' keeps the flattened per-frame layout.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------
# CLI: orchestrate reconstruction workflow
# --------------------------------------------------
def run_cli(default_split: str, argv: Optional[Sequence[str]] = None) -> None:
    """Entry point used by the per-split scripts."""
    args = parse_common_args(default_split, argv)

    spec = SPLIT_SPECS[args.split]
    output_dir = args.output_dir or Path("results") / "reconstructions" / spec.name
    output_dir = make_absolute(output_dir)
    mosaic_dir = output_dir / "mosaics"
    colorbar_dir = mosaic_dir / "colorbars"

    device = resolve_device(args.device)
    config = load_yaml_config(args.config)
    model = build_model(config, args.checkpoint, device)
    assets = load_split_assets(spec)

    total_samples = assets.targets.shape[0]
    start = max(0, args.sample_offset)
    end = total_samples if args.max_samples is None else min(total_samples, start + args.max_samples)
    eval_indices = list(range(start, end))
    if not eval_indices:
        raise ValueError("Resolved empty evaluation range; check `--sample-offset` and `--max-samples`.")

    mosaic_mode = args.mosaic_mode
    frames_per_clip = assets.frames_per_clip
    sample_mosaic_indices: List[int] = []
    clip_mosaic_indices: List[int] = []

    if mosaic_mode == "clip":
        total_clips = total_samples // frames_per_clip
        clip_idx = start // frames_per_clip
        while len(clip_mosaic_indices) < max(0, args.mosaic_count) and clip_idx < total_clips:
            clip_base = clip_idx * frames_per_clip
            clip_end = clip_base + frames_per_clip
            if clip_base < start or clip_end > end:
                clip_idx += 1
                continue
            clip_mosaic_indices.append(clip_idx)
            clip_idx += 1
        if len(clip_mosaic_indices) < max(0, args.mosaic_count):
            print(
                f"Warning: requested {args.mosaic_count} clip mosaics but only "
                f"{len(clip_mosaic_indices)} clip(s) fall entirely inside the evaluation range."
            )
    else:
        sample_mosaic_indices = eval_indices[: max(0, args.mosaic_count)]

    metrics, sample_payload, clip_payload, skipped = evaluate_split(
        model,
        assets,
        device,
        eval_indices,
        sample_mosaic_indices,
        clip_mosaic_indices,
    )
    write_metrics(metrics, output_dir / spec.metrics_filename)

    if mosaic_mode == "clip":
        if clip_payload:
            render_clip_mosaics(
                clip_payload,
                output_dir=mosaic_dir,
                prefix=spec.mosaic_prefix,
                error_vmax=args.mosaic_error_vmax,
                colorbar_dir=colorbar_dir,
            )
    elif sample_payload:
        render_mosaic_pages(
            sample_payload,
            output_dir=mosaic_dir,
            prefix=spec.mosaic_prefix,
            cols_per_page=max(1, args.mosaic_cols),
            error_vmax=args.mosaic_error_vmax,
            colorbar_dir=colorbar_dir,
        )

    if skipped:
        skipped_path = output_dir / "skipped_samples.txt"
        skipped_path.write_text(
            "Samples skipped due to runtime issues:\n" + "\n".join(str(i) for i in skipped)
        )

    print(f"Processed {metrics['evaluated_samples']} samples for split '{spec.name}' on device {device}.")
    print(f"Average MAE: {metrics['average_mae']:.4f} | Average SSIM: {metrics['average_ssim']:.4f}")
    if (mosaic_mode == "clip" and clip_payload) or (mosaic_mode == "sample" and sample_payload):
        print(f"Mosaics saved under: {mosaic_dir} (mode: {mosaic_mode})")
    else:
        print("No mosaics were generated.")


# --------------------------------------------------
# Utility: format integers as ordinals
# --------------------------------------------------
def ordinal(n: int) -> str:
    """Return the ordinal representation of an integer (1 -> 1st)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
