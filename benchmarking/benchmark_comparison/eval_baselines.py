"""
###############################################################################
# Benchmarking Python Module (2026)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description: benchmark_comparison/eval_baselines.py
#              Baseline benchmarking source file for training, modeling,
#              evaluation, or utilities in the benchmarking workflow.
###############################################################################
"""

import argparse
from copy import deepcopy
import json
import os
import sys
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from exc


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIRS = {
    "conv_fno": BASE_DIR / "conv-fno",
    "fno": BASE_DIR / "fno",
    "ufno": BASE_DIR / "ufno",
    "vanilla_mionet": BASE_DIR / "vanilla_MIONet",
    "fourier_mionet": BASE_DIR / "Fourier_MIONet",
}

# Resolve canonical repo roots either from paths.py (preferred) or from this file location.
try:
    from paths import (  # type: ignore
        BENCHMARK_COMPARISON_ROOT as _PATHS_COMPARE_ROOT,
        GAS_SATURATION_VQVAE_ROOT as _PATHS_GAS_ROOT,
        JOINT_DITV_ROOT as _PATHS_JOINT_ROOT,
        PRESSURE_BUILDUP_VAE_ROOT as _PATHS_PRESSURE_ROOT,
        REPO_ROOT as _PATHS_REPO_ROOT,
    )
except Exception:  # pragma: no cover
    _PATHS_REPO_ROOT = BASE_DIR.parent
    _PATHS_COMPARE_ROOT = BASE_DIR / "benchmark_comparison"
    _PATHS_GAS_ROOT = BASE_DIR.parent / "gas_saturation_vqvae"
    _PATHS_PRESSURE_ROOT = BASE_DIR.parent / "pressure_buildup_vae"
    _PATHS_JOINT_ROOT = BASE_DIR.parent / "gas_saturation_pressure_buildup_ditv"

DEFAULT_REPO_ROOT = Path(_PATHS_REPO_ROOT)
DEFAULT_COMPARE_ROOT = Path(_PATHS_COMPARE_ROOT)
DEFAULT_GAS_INPUTS_ROOT = Path(_PATHS_GAS_ROOT)
DEFAULT_PRESSURE_INPUTS_ROOT = Path(_PATHS_PRESSURE_ROOT)
DEFAULT_EVAL_ROOT = Path(_PATHS_JOINT_ROOT)

FIELD_INDICES = [0, 1, 2, 9, 10]
MIO_INDICES = [4, 5, 6, 7, 8, 9, 10]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _as_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "null":
        return None
    return Path(value)


def _torch_load(path: Path, device: torch.device):
    # PyTorch 2.6 defaults to weights_only=True; force full load for trusted checkpoints.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't support weights_only.
        return torch.load(path, map_location=device)


def _resolve_eval_root(cfg: dict, modality: str) -> Path:
    data_cfg = cfg.get("data", {})
    eval_root = _as_path(data_cfg.get("eval_root"))
    eval_co2_root = _as_path(data_cfg.get("eval_co2_root"))
    eval_dp_root = _as_path(data_cfg.get("eval_dp_root"))
    if modality == "gas" and eval_co2_root is not None:
        return eval_co2_root
    if modality == "pressure" and eval_dp_root is not None:
        return eval_dp_root
    if eval_root is not None:
        return eval_root
    return DEFAULT_EVAL_ROOT


def _resolve_targets_path(cfg: dict, modality: str, inputs_root: Path) -> Path:
    """Resolve target path for evaluation.

    targets_source:
      - "lavig": use LAViG evaluation targets (default; *_evaluation/val_*_target.pt)
      - "baseline": use baseline validation outputs (benchmarking/*_val_outputs.pt)
    """
    data_cfg = cfg.get("data", {})
    targets_source = str(data_cfg.get("targets_source", "lavig")).lower()
    targets_root = _as_path(data_cfg.get("targets_root"))

    if targets_source == "baseline":
        root = targets_root or inputs_root
        if root is None:
            raise ValueError("targets_root (or inputs_root) must be set when targets_source=baseline")
        if modality == "gas":
            return root / "co2_data" / "validation_dataset_gas_saturation" / "sg_val_outputs.pt"
        if modality == "pressure":
            return root / "dP_data" / "validation_dataset_pressure_buildup" / "dP_val_outputs.pt"
        raise ValueError(f"Unknown modality: {modality}")

    # Default: LAViG evaluation targets
    eval_root = _resolve_eval_root(cfg, modality)
    if modality == "gas":
        return eval_root / "co2_data_evaluation" / "val_sg_target.pt"
    if modality == "pressure":
        return eval_root / "dP_data_evaluation" / "val_dP_target.pt"
    raise ValueError(f"Unknown modality: {modality}")


def _resolve_inputs_root(cfg: dict, modality: str) -> Path:
    data_cfg = cfg.get("data", {})
    inputs_root = _as_path(data_cfg.get("inputs_root"))
    inputs_co2_root = _as_path(data_cfg.get("inputs_co2_root"))
    inputs_dp_root = _as_path(data_cfg.get("inputs_dp_root"))

    if modality == "gas":
        root = inputs_co2_root or inputs_root
    elif modality == "pressure":
        root = inputs_dp_root or inputs_root
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if root is not None:
        return root
    if modality == "gas":
        return DEFAULT_GAS_INPUTS_ROOT
    return DEFAULT_PRESSURE_INPUTS_ROOT


def _resolve_basic_denorm(metrics_cfg: dict, modality: str) -> Tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[float]]:
    denorm_cfg = (metrics_cfg.get("denorm") or {}).get(modality)
    if not denorm_cfg:
        return None, None
    scale = float(denorm_cfg.get("scale", 1.0))
    shift = float(denorm_cfg.get("shift", 0.0))

    def _fn(x: torch.Tensor) -> torch.Tensor:
        return x * scale + shift

    return _fn, scale


def _resolve_basic_data_range(
    metrics_cfg: dict,
    modality: str,
    denorm_scale: Optional[float],
    targets: Optional[torch.Tensor] = None,
    denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> float:
    data_range = metrics_cfg.get("basic_data_range")
    if isinstance(data_range, dict):
        data_range = data_range.get(modality)

    if isinstance(data_range, str) and data_range.lower() == "auto":
        if targets is None:
            raise ValueError("basic_data_range='auto' requires targets to be loaded.")
        if denorm_fn is not None:
            targets = denorm_fn(targets)
        tmin = float(targets.min().item())
        tmax = float(targets.max().item())
        if not math.isfinite(tmin) or not math.isfinite(tmax):
            return 2.0 * denorm_scale if denorm_scale is not None else 2.0
        rng = tmax - tmin
        if rng <= 0:
            return 2.0 * denorm_scale if denorm_scale is not None else 2.0
        return rng

    if isinstance(data_range, (int, float)):
        return float(data_range)

    if denorm_scale is not None:
        return 2.0 * denorm_scale
    return 2.0


def _resolve_pressure_physical_transform(
    metrics_cfg: dict,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], float, float, Optional[float]]:
    """
    Return a transform that maps standardized baseline pressure to physical units.

    Defaults follow the existing baseline eval scripts in this workspace.
    """
    pressure_cfg = metrics_cfg.get("pressure_standardization") or {}
    scale = float(pressure_cfg.get("scale", 18.772821433027488))
    shift = float(pressure_cfg.get("shift", 4.172939172019009))
    clamp_min_val = pressure_cfg.get("clamp_min", 0.0)
    clamp_min = float(clamp_min_val) if clamp_min_val is not None else None

    def _fn(x: torch.Tensor) -> torch.Tensor:
        y = x * scale + shift
        if clamp_min is not None:
            y = torch.clamp(y, min=clamp_min)
        return y

    return _fn, scale, shift, clamp_min


def _compute_reference_minmax(
    targets: torch.Tensor,
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-(sample, frame) min/max over spatial dimensions [H, W].
    Returned tensors keep broadcastable shape [B, T, 1, 1, 1].
    """
    ref = preprocess_fn(targets) if preprocess_fn is not None else targets
    ref_min = ref.amin(dim=(3, 4), keepdim=True).cpu()
    ref_max = ref.amax(dim=(3, 4), keepdim=True).cpu()
    return ref_min, ref_max


def _normalize_with_reference_minmax(
    tensor: torch.Tensor,
    ref_min: torch.Tensor,
    ref_max: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Map tensor to [-1, 1] using broadcastable reference min/max statistics."""
    ref_range = ref_max - ref_min
    zero_mask = ref_range.abs() <= eps
    safe_range = torch.where(zero_mask, torch.ones_like(ref_range), ref_range)
    norm = 2.0 * (tensor - ref_min) / safe_range - 1.0
    # Degenerate constant maps carry no contrast; keep them centered.
    return torch.where(zero_mask, torch.zeros_like(norm), norm)


def _load_chunk_metrics(path: Path, chunk: int) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and str(chunk) in data:
        return data[str(chunk)]
    if isinstance(data, dict) and chunk in data:
        return data[chunk]
    if isinstance(data, dict) and "gas" in data and "pressure" in data:
        return data
    return None


def _merge_lavig_metrics(
    stages: List[int],
    metrics_dir: Optional[Path],
    quality_dir: Optional[Path],
) -> Optional[Dict[int, Dict[str, Dict[str, Dict[str, float]]]]]:
    if metrics_dir is None and quality_dir is None:
        return None
    metrics_dir = metrics_dir or quality_dir
    quality_dir = quality_dir or metrics_dir
    if metrics_dir is None or quality_dir is None:
        return None

    merged: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for stage in stages:
        basic_path = metrics_dir / f"chunk{stage}_autoreg.json"
        quality_path = quality_dir / f"chunk{stage}_quality.json"

        basic = _load_chunk_metrics(basic_path, stage) or {}
        quality = _load_chunk_metrics(quality_path, stage) or {}

        if not basic and not quality:
            continue

        def _pick_metrics(src: dict, keys: List[str]) -> dict:
            out: Dict[str, Dict[str, float]] = {}
            for key in keys:
                val = src.get(key)
                if isinstance(val, dict):
                    out[key] = val
            return out

        gas_basic = _pick_metrics(basic.get("gas", {}), ["mse", "mae", "rmse", "psnr"])
        pressure_basic = _pick_metrics(basic.get("pressure", {}), ["mse", "mae", "rmse", "psnr"])
        gas_quality = _pick_metrics(quality.get("gas", {}), ["ssim", "psnr", "lpips", "fvd"])
        pressure_quality = _pick_metrics(quality.get("pressure", {}), ["ssim", "psnr", "lpips", "fvd"])

        evaluated_samples = basic.get("evaluated_samples") or quality.get("evaluated_samples") or {}
        predicted_frames = basic.get("predicted_frames") or quality.get("predicted_frames") or {}

        merged[stage] = {
            "gas": {"basic": gas_basic, "quality": gas_quality},
            "pressure": {"basic": pressure_basic, "quality": pressure_quality},
            "evaluated_samples": evaluated_samples,
            "predicted_frames": predicted_frames,
        }

    return merged or None


def load_inputs_targets(cfg: dict, modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs_root = _resolve_inputs_root(cfg, modality)

    if modality == "gas":
        inputs_path = inputs_root / "co2_data" / "validation_dataset_gas_saturation" / "sg_val_inputs.pt"
    elif modality == "pressure":
        inputs_path = inputs_root / "dP_data" / "validation_dataset_pressure_buildup" / "dP_val_inputs.pt"
    else:
        raise ValueError(f"Unknown modality: {modality}")
    targets_path = _resolve_targets_path(cfg, modality, inputs_root)

    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing inputs: {inputs_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing targets: {targets_path}")

    inputs = torch.load(inputs_path)
    targets = torch.load(targets_path)

    # inputs expected: [B, H, W, T, C]
    if inputs.ndim != 5:
        raise ValueError(f"Expected inputs shape [B,H,W,T,C], got {inputs.shape}")

    # targets expected: [B, T, C, H, W] or [B, T, H, W]
    # baseline outputs may be [B, H, W, T] (or [B, H, W, T, C])
    if targets.ndim == 4:
        if targets.shape[1] == inputs.shape[1] and targets.shape[2] == inputs.shape[2]:
            # [B, H, W, T] -> [B, T, H, W]
            targets = targets.permute(0, 3, 1, 2)
        targets = targets.unsqueeze(2)
    elif targets.ndim == 5:
        if targets.shape[1] == inputs.shape[1] and targets.shape[2] == inputs.shape[2]:
            # [B, H, W, T, C] -> [B, T, C, H, W]
            targets = targets.permute(0, 3, 4, 1, 2)
    if targets.ndim != 5:
        raise ValueError(f"Expected targets shape [B,T,C,H,W], got {targets.shape}")

    return inputs, targets


# -----------------------------
# Metrics (basic)
# -----------------------------

def compute_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, data_range: float = 2.0):
    diff = predictions - targets
    mse = diff.pow(2).flatten(start_dim=1).mean(dim=1)
    mae = diff.abs().flatten(start_dim=1).mean(dim=1)
    rmse = torch.sqrt(mse)

    mse_clamped = torch.clamp(mse, min=1e-10)
    max_val = torch.tensor(data_range, device=predictions.device)
    psnr = 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse_clamped)
    psnr = torch.where(torch.isfinite(psnr), psnr, torch.full_like(psnr, 100.0))

    return {"mse": mse, "mae": mae, "rmse": rmse, "psnr": psnr}


def aggregate_metrics(metric_lists: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in metric_lists.items():
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[key] = {"mean": float(tensor.mean().item()), "std": float(tensor.std(unbiased=False).item())}
    return summary


def _extract_fvd_features(
    lavig_metrics,
    videos: torch.Tensor,
    detector: torch.nn.Module,
    detector_kwargs: Dict[str, object],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract detector features for FVD from prepared [N,C,T,H,W] videos."""
    features_list: List[np.ndarray] = []
    total = int(videos.shape[0])
    if total == 0:
        raise ValueError("Cannot compute FVD features with zero samples.")

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = videos[start:stop]
        batch = lavig_metrics._ensure_min_temporal_frames(batch)
        batch = lavig_metrics._ensure_three_channels(batch)
        with torch.no_grad():
            feats = detector(batch.to(device), **detector_kwargs)
        if isinstance(feats, (list, tuple)):
            if not feats:
                raise ValueError("Detector returned empty feature list.")
            feats = feats[0]
        if not isinstance(feats, torch.Tensor):
            raise TypeError("Detector output is not a tensor.")
        feats = feats.view(feats.shape[0], -1)
        features_list.append(feats.detach().to(dtype=torch.float64, device="cpu").numpy())

    return np.concatenate(features_list, axis=0)


def _feature_mean_cov(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected 2D non-empty feature matrix, got {features.shape}.")
    mean = features.mean(axis=0)
    cov = np.cov(features, rowvar=False, bias=True)
    return mean, cov


def _compute_fvd_with_bootstrap(
    lavig_metrics,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    stylegan_src: Optional[str],
    fvd_batch_size: int,
    fvd_target_frames: Optional[int],
    bootstrap_repeats: int,
    bootstrap_sample_size: Optional[int],
    bootstrap_seed: int,
) -> Tuple[float, float]:
    """
    Compute full-set FVD mean and bootstrap std.

    Bootstrap draws are performed in feature space to avoid re-running the detector.
    """
    metric_utils = lavig_metrics._import_stylegan_metric_utils(stylegan_src)
    detector_resource = lavig_metrics._resolve_detector_resource(
        lavig_metrics.FVD_DETECTOR_FILENAME,
        lavig_metrics.FVD_DETECTOR_URL,
    )
    detector = metric_utils.get_feature_detector(
        url=detector_resource,
        device=device,
        num_gpus=1,
        rank=0,
        verbose=False,
    )
    detector_kwargs = {"rescale": True, "resize": True, "return_features": True}

    preds_unit = lavig_metrics.to_unit_interval(predictions.detach().clone()).cpu()
    gts_unit = lavig_metrics.to_unit_interval(targets.detach().clone()).cpu()
    pred_videos = lavig_metrics._prepare_videos_for_fvd(preds_unit, fvd_target_frames)
    target_videos = lavig_metrics._prepare_videos_for_fvd(gts_unit, fvd_target_frames)
    if pred_videos.shape[2] != target_videos.shape[2]:
        raise ValueError(
            "Predictions and targets must share the same number of frames after FVD alignment."
        )

    pred_features = _extract_fvd_features(
        lavig_metrics, pred_videos, detector, detector_kwargs, fvd_batch_size, device
    )
    target_features = _extract_fvd_features(
        lavig_metrics, target_videos, detector, detector_kwargs, fvd_batch_size, device
    )

    mu_pred, sigma_pred = _feature_mean_cov(pred_features)
    mu_real, sigma_real = _feature_mean_cov(target_features)
    fvd_mean = float(lavig_metrics._calculate_frechet_distance(mu_pred, sigma_pred, mu_real, sigma_real))

    if bootstrap_repeats <= 1:
        return fvd_mean, 0.0

    n_pred = pred_features.shape[0]
    n_real = target_features.shape[0]
    sample_size = min(n_pred, n_real)
    if bootstrap_sample_size is not None:
        sample_size = min(sample_size, max(2, int(bootstrap_sample_size)))
    if sample_size < 2:
        return fvd_mean, 0.0

    rng = np.random.default_rng(int(bootstrap_seed))
    boot_vals: List[float] = []
    for _ in range(int(bootstrap_repeats)):
        idx_pred = rng.integers(0, n_pred, size=sample_size)
        idx_real = rng.integers(0, n_real, size=sample_size)
        try:
            mu_b_pred, sigma_b_pred = _feature_mean_cov(pred_features[idx_pred])
            mu_b_real, sigma_b_real = _feature_mean_cov(target_features[idx_real])
            val = float(
                lavig_metrics._calculate_frechet_distance(mu_b_pred, sigma_b_pred, mu_b_real, sigma_b_real)
            )
            if math.isfinite(val):
                boot_vals.append(val)
        except Exception:
            continue

    if len(boot_vals) < 2:
        return fvd_mean, 0.0
    return fvd_mean, float(np.std(np.asarray(boot_vals, dtype=np.float64), ddof=0))


# -----------------------------
# Quality metrics (reuse LAViG)
# -----------------------------

_lavig_metrics_mod = None


def load_lavig_metrics_module(lavig_repo_root: Path):
    global _lavig_metrics_mod
    if _lavig_metrics_mod is not None:
        return _lavig_metrics_mod

    if not lavig_repo_root.exists():
        raise FileNotFoundError(f"LAViG repo root not found: {lavig_repo_root}")

    # Ensure repo root and subdir are importable
    sys.path.insert(0, str(lavig_repo_root))
    sys.path.insert(0, str(lavig_repo_root / "gas_saturation_pressure_buildup_ditv"))

    module_path = lavig_repo_root / "gas_saturation_pressure_buildup_ditv" / "common_metrics_on_video_quality.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing LAViG metrics module: {module_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("lavig_metrics", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    _lavig_metrics_mod = mod
    return mod


# -----------------------------
# Model adapters
# -----------------------------

@dataclass
class ModelSpec:
    name: str
    gas_ckpt: Optional[Path]
    pressure_ckpt: Optional[Path]


def load_torch_model(ckpt_path: Path, device: torch.device):
    # Ensure the checkpoint's model module directory is importable for unpickling.
    model_dir = ckpt_path.parent.parent
    if model_dir.exists():
        sys.path.insert(0, str(model_dir))
    obj = _torch_load(ckpt_path, device)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        raise ValueError(
            f"Checkpoint {ckpt_path} is a state_dict-only checkpoint. "
            "Provide a full model checkpoint or update the loader to rebuild the model."
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    model.to(device)
    model.eval()
    return model


def _ensure_bhwt(pred: torch.Tensor, batch_size: int) -> torch.Tensor:
    # Expect [B,H,W,T] (preferred) or [H,W,T] if batch squeezed
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if pred.ndim == 5 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if pred.ndim != 4:
        raise ValueError(f"Expected prediction dims 4 (B,H,W,T) or 3 (H,W,T), got {pred.shape}")
    return pred


def predict_torch_volume(model: torch.nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs),
        batch_size=batch_size,
        shuffle=False,
    )
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            y = model(x)
            y = _ensure_bhwt(y, x.shape[0])
            preds.append(y.cpu())
    pred = torch.cat(preds, dim=0)
    # [B,H,W,T] -> [B,T,1,H,W]
    pred = pred.permute(0, 3, 1, 2).unsqueeze(2)
    return pred


def _import_module_from_path(module_name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def build_xrt_full(n_frames: int) -> np.ndarray:
    t = np.linspace(0, 1, n_frames).astype(np.float32)
    z = np.linspace(0, 1, 96).astype(np.float32)
    r = np.linspace(0, 1, 200).astype(np.float32)
    return np.array([[a, b, c] for a in t for b in z for c in r], dtype=np.float32)


def build_xrt_time_only(n_frames: int) -> np.ndarray:
    t = np.linspace(0, 1, n_frames).astype(np.float32)
    return np.array([[c] for c in t], dtype=np.float32)


def build_mionet_inputs(inputs: torch.Tensor, n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_field = inputs[:, :, :, 0, FIELD_INDICES]
    x_mio = inputs[..., MIO_INDICES].mean(dim=(1, 2, 3))
    xrt = build_xrt_full(n_frames)
    return x_field.cpu().numpy(), x_mio.cpu().numpy(), xrt


def build_mionet_targets(targets: torch.Tensor, n_frames: int) -> np.ndarray:
    # targets: [B,T,1,H,W]
    y = targets[:, :n_frames].permute(0, 3, 1, 2).reshape(targets.shape[0], -1)
    return y.cpu().numpy()


def make_mionet_vanilla_predictor(model_dir: Path, ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(model_dir))
    module = _import_module_from_path(f"vanilla_mionet_{model_dir.name}", model_dir / "model.py")

    # Build network once (architecture does not depend on n_frames)
    net = module.MIONetCartesianProd(
        layer_sizes_branch1=[96 * 200 * 17 * 3, module.Encoder()],
        layer_sizes_branch2=[7, 512, 512, 512, 512],
        layer_sizes_trunk=[3, 512, 512, 512, 512],
        activation="relu",
        kernel_initializer="Glorot normal",
        regularization=("l2", 4e-6),
        trunk_last_activation=False,
        merge_operation="mul",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=None,
    )
    state = _torch_load(ckpt_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, torch.nn.Module):
        net = state
    else:
        incompatible = net.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            print(
                f"[mionet] Ignoring unexpected keys in {ckpt_path.name}: {len(incompatible.unexpected_keys)}"
            )

    net.to(device)
    net.eval()

    def _predict(inputs: torch.Tensor, targets: torch.Tensor, n_frames: int, batch_size: int) -> torch.Tensor:
        x_field, x_mio, xrt = build_mionet_inputs(inputs, n_frames)
        xrt_t = torch.from_numpy(xrt).to(device)

        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                end = min(start + batch_size, inputs.shape[0])
                xf = torch.from_numpy(x_field[start:end]).to(device)
                xm = torch.from_numpy(x_mio[start:end]).to(device)
                y = net((xf, xm, xrt_t))
                preds.append(y.cpu())

        pred = torch.cat(preds, dim=0)
        pred = pred.view(inputs.shape[0], n_frames, 96, 200)
        pred = pred.unsqueeze(2)
        return pred

    return _predict


def make_mionet_fourier_predictor(model_dir: Path, ckpt_path: Path, device: torch.device, modality: str):
    sys.path.insert(0, str(model_dir))
    module = _import_module_from_path(f"fourier_mionet_{model_dir.name}", model_dir / "model.py")

    # Import deepxde from this folder
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    import deepxde as dde  # type: ignore

    # Load hyperparameters from config if available
    cfg_name = "train_sg.yaml" if modality == "gas" else "train_dp.yaml"
    cfg_path = model_dir / cfg_name
    if cfg_path.exists():
        cfg = load_yaml(str(cfg_path))
        modes1 = int(cfg.get("modes1", 10))
        modes2 = int(cfg.get("modes2", 10))
        width = int(cfg.get("width", 36))
        width2 = int(cfg.get("width2", 128))
    else:
        modes1, modes2, width, width2 = 10, 10, 36, 128

    gelu = torch.nn.GELU()

    net = dde.nn.pytorch.mionet.MIONetCartesianProd(
        layer_sizes_branch1=[96 * 200 * 17 * 3, module.branch1(width)],
        layer_sizes_branch2=[7 * width, module.branch2(width)],
        layer_sizes_trunk=[1, width, width, width, width],
        activation={
            "branch1": gelu,
            "branch2": gelu,
            "trunk": gelu,
            "merger": gelu,
            "output merger": gelu,
        },
        kernel_initializer="Glorot normal",
        regularization=("l2", 4e-6),
        trunk_last_activation=False,
        merge_operation="sum",
        layer_sizes_merger=None,
        output_merge_operation="mul",
        layer_sizes_output_merger=[width, module.decoder(modes1, modes2, width, width2)],
    )

    state = _torch_load(ckpt_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, torch.nn.Module):
        net = state
    else:
        incompatible = net.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            print(
                f"[fourier-mionet] Ignoring unexpected keys in {ckpt_path.name}: {len(incompatible.unexpected_keys)}"
            )

    net.to(device)
    net.eval()

    def _predict(inputs: torch.Tensor, targets: torch.Tensor, n_frames: int, batch_size: int) -> torch.Tensor:
        x_field = inputs[:, :, :, 0, FIELD_INDICES]
        x_mio = inputs[..., MIO_INDICES].mean(dim=(1, 2, 3))
        xrt = build_xrt_time_only(n_frames)
        xrt_t = torch.from_numpy(xrt).to(device)

        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                end = min(start + batch_size, inputs.shape[0])
                xf = torch.from_numpy(x_field[start:end].cpu().numpy()).to(device)
                xm = torch.from_numpy(x_mio[start:end].cpu().numpy()).to(device)
                y = net((xf, xm, xrt_t))
                preds.append(y.cpu())

        pred = torch.cat(preds, dim=0)
        pred = pred.view(inputs.shape[0], n_frames, 96, 200)
        pred = pred.unsqueeze(2)
        return pred

    return _predict

# -----------------------------
# Evaluation
# -----------------------------

def evaluate_model(
    model_name: str,
    model_pred_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    stages: List[int],
    context_frames: int,
    predict_frames: int,
    batch_size: int,
    device: torch.device,
    compute_quality: bool,
    quality_cfg: dict,
    lavig_repo_root: Optional[Path],
    max_samples: Optional[int],
    max_samples_quality: Optional[int],
    basic_denorm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    basic_data_range: float = 2.0,
    clamp_quality: bool = False,
    pred_metric_pre_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    target_metric_pre_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    metric_ref_min: Optional[torch.Tensor] = None,
    metric_ref_max: Optional[torch.Tensor] = None,
    normalize_pred_with_ref: bool = False,
    normalize_target_with_ref: bool = False,
) -> Dict[int, Dict[str, Dict[str, Dict[str, float]]]]:
    results: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}

    # LAViG metrics module (loaded when quality metrics are enabled)
    lavig_metrics = None
    if compute_quality:
        if lavig_repo_root is None:
            raise ValueError("lavig_repo_root must be set when compute_quality=true")
        lavig_metrics = load_lavig_metrics_module(lavig_repo_root)

    num_samples = inputs.shape[0]
    if max_samples is not None:
        num_samples = min(num_samples, int(max_samples))
    inputs = inputs[:num_samples]
    targets = targets[:num_samples]

    for stage in stages:
        total_frames = context_frames + predict_frames * stage
        if total_frames > targets.shape[1]:
            raise ValueError(
                f"Stage {stage} requires {total_frames} frames but targets have {targets.shape[1]} frames"
            )

        inputs_stage = inputs[..., :total_frames, :]
        targets_stage = targets[:, :total_frames]

        preds = model_pred_fn(inputs_stage, targets_stage, total_frames)

        pred_suffix = preds[:, context_frames:total_frames]
        target_suffix = targets_stage[:, context_frames:total_frames]

        # Optional metric-space alignment so all models are compared in the same domain.
        if pred_metric_pre_fn is not None:
            pred_suffix = pred_metric_pre_fn(pred_suffix)
        if target_metric_pre_fn is not None:
            target_suffix = target_metric_pre_fn(target_suffix)

        if (
            metric_ref_min is not None
            and metric_ref_max is not None
            and (normalize_pred_with_ref or normalize_target_with_ref)
        ):
            stage_ref_min = metric_ref_min[: pred_suffix.shape[0], context_frames:total_frames].to(
                device=pred_suffix.device, dtype=pred_suffix.dtype
            )
            stage_ref_max = metric_ref_max[: pred_suffix.shape[0], context_frames:total_frames].to(
                device=pred_suffix.device, dtype=pred_suffix.dtype
            )
            if normalize_pred_with_ref:
                pred_suffix = _normalize_with_reference_minmax(pred_suffix, stage_ref_min, stage_ref_max)
            if normalize_target_with_ref:
                target_suffix = _normalize_with_reference_minmax(target_suffix, stage_ref_min, stage_ref_max)

        metric_accumulators: Dict[str, List[float]] = {"mse": [], "mae": [], "rmse": [], "psnr": []}

        # basic metrics in batches (CPU)
        batch_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pred_suffix, target_suffix),
            batch_size=batch_size,
            shuffle=False,
        )
        for pred_b, target_b in batch_loader:
            pred_b = pred_b.to(device)
            target_b = target_b.to(device)
            if basic_denorm_fn is not None:
                pred_b = basic_denorm_fn(pred_b)
                target_b = basic_denorm_fn(target_b)
            metrics = compute_batch_metrics(pred_b, target_b, data_range=basic_data_range)
            for key, tensor in metrics.items():
                metric_accumulators[key].extend(tensor.detach().cpu().tolist())

        summary = {
            "basic": aggregate_metrics(metric_accumulators),
            "quality": {},
            "evaluated_samples": {"count": int(pred_suffix.shape[0])},
            "predicted_frames": {"count": int(predict_frames * stage)},
        }

        # Quality metrics
        if compute_quality and lavig_metrics is not None:
            q_samples = pred_suffix.shape[0]
            if max_samples_quality is not None:
                q_samples = min(q_samples, int(max_samples_quality))
            pred_q = pred_suffix[:q_samples].to(device)
            target_q = target_suffix[:q_samples].to(device)
            if clamp_quality:
                pred_q = pred_q.clamp(-1.0, 1.0)
                target_q = target_q.clamp(-1.0, 1.0)

            compute_fvd_requested = bool(quality_cfg.get("compute_fvd", False))
            fvd_batch_size = int(quality_cfg.get("fvd_batch_size", 16))
            fvd_target_frames = quality_cfg.get("fvd_num_frames")
            fvd_bootstrap_repeats = int(quality_cfg.get("fvd_bootstrap_repeats", 8))
            fvd_bootstrap_seed = int(quality_cfg.get("fvd_bootstrap_seed", 12345))
            fvd_bootstrap_sample_size_cfg = quality_cfg.get("fvd_bootstrap_sample_size")
            fvd_bootstrap_sample_size = (
                None if fvd_bootstrap_sample_size_cfg is None else int(fvd_bootstrap_sample_size_cfg)
            )
            use_bootstrap_fvd = compute_fvd_requested and fvd_bootstrap_repeats > 1

            adv_metrics = lavig_metrics.compute_metrics(
                predictions=pred_q,
                targets=target_q,
                lpips_device=device,
                stylegan_src=quality_cfg.get("stylegan_src"),
                fvd_batch_size=fvd_batch_size,
                compute_fvd=compute_fvd_requested and not use_bootstrap_fvd,
                fvd_target_frames=fvd_target_frames,
            )

            if use_bootstrap_fvd:
                seed_offset = int(stage * 1000 + sum(ord(ch) for ch in model_name))
                try:
                    fvd_mean, fvd_std = _compute_fvd_with_bootstrap(
                        lavig_metrics,
                        pred_q,
                        target_q,
                        device=device,
                        stylegan_src=quality_cfg.get("stylegan_src"),
                        fvd_batch_size=fvd_batch_size,
                        fvd_target_frames=fvd_target_frames,
                        bootstrap_repeats=fvd_bootstrap_repeats,
                        bootstrap_sample_size=fvd_bootstrap_sample_size,
                        bootstrap_seed=fvd_bootstrap_seed + seed_offset,
                    )
                    adv_metrics["fvd"] = fvd_mean
                    adv_metrics["fvd_std"] = fvd_std
                except Exception as exc:
                    print(
                        f"[fvd-bootstrap] {model_name} stage {stage}: "
                        f"failed bootstrap ({exc}); falling back to single-run FVD."
                    )
                    fvd_value = lavig_metrics.calculate_fvd(
                        targets=target_q,
                        predictions=pred_q,
                        device=device,
                        stylegan_src=quality_cfg.get("stylegan_src"),
                        batch_size=fvd_batch_size,
                        target_frames=fvd_target_frames,
                    )
                    adv_metrics["fvd"] = float(fvd_value)
                    adv_metrics["fvd_std"] = 0.0

            summary["quality"]["ssim"] = {"mean": adv_metrics["ssim"], "std": adv_metrics["ssim_std"]}
            summary["quality"]["psnr"] = {"mean": adv_metrics["psnr"], "std": adv_metrics["psnr_std"]}
            summary["quality"]["lpips"] = {"mean": adv_metrics["lpips"], "std": adv_metrics["lpips_std"]}
            if "fvd" in adv_metrics:
                summary["quality"]["fvd"] = {"mean": adv_metrics["fvd"], "std": adv_metrics.get("fvd_std", 0.0)}

        results[stage] = summary

    return results


def fmt_pm(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} \\pm {std:.6f}"


def render_tables(model_name: str, results: Dict[int, Dict[str, Dict[str, Dict[str, float]]]]) -> str:
    lines = []

    # Basic metrics table
    lines.append(f"% Autoregressive metrics for {model_name}")
    lines.append("\\pgfplotstableread{")
    lines.append("stage predframes gas_mse_pm gas_mae_pm gas_rmse_pm gas_psnr_pm pressure_mse_pm pressure_mae_pm pressure_rmse_pm pressure_psnr_pm")
    for stage, data in sorted(results.items()):
        predframes = data.get("predicted_frames", {}).get("count", "")
        gas_basic = data.get("gas", {}).get("basic", {})
        pressure_basic = data.get("pressure", {}).get("basic", {})
        row = [
            str(stage),
            str(predframes),
            "{" + fmt_pm(gas_basic.get("mse", {}).get("mean"), gas_basic.get("mse", {}).get("std")) + "}",
            "{" + fmt_pm(gas_basic.get("mae", {}).get("mean"), gas_basic.get("mae", {}).get("std")) + "}",
            "{" + fmt_pm(gas_basic.get("rmse", {}).get("mean"), gas_basic.get("rmse", {}).get("std")) + "}",
            "{" + fmt_pm(gas_basic.get("psnr", {}).get("mean"), gas_basic.get("psnr", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_basic.get("mse", {}).get("mean"), pressure_basic.get("mse", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_basic.get("mae", {}).get("mean"), pressure_basic.get("mae", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_basic.get("rmse", {}).get("mean"), pressure_basic.get("rmse", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_basic.get("psnr", {}).get("mean"), pressure_basic.get("psnr", {}).get("std")) + "}",
        ]
        lines.append(" ".join(row))
    lines.append(f"}}\\AutoregMetrics{model_name.replace('-', '').replace(' ', '')}")
    lines.append("")

    # Quality metrics table (if present)
    lines.append(f"% Quality metrics for {model_name}")
    lines.append("\\pgfplotstableread{")
    lines.append("stage predframes gas_ssim_pm gas_psnr_pm gas_lpips_pm gas_fvd_pm pressure_ssim_pm pressure_psnr_pm pressure_lpips_pm pressure_fvd_pm")
    for stage, data in sorted(results.items()):
        predframes = data.get("predicted_frames", {}).get("count", "")
        gas_quality = data.get("gas", {}).get("quality", {})
        pressure_quality = data.get("pressure", {}).get("quality", {})
        row = [
            str(stage),
            str(predframes),
            "{" + fmt_pm(gas_quality.get("ssim", {}).get("mean"), gas_quality.get("ssim", {}).get("std")) + "}",
            "{" + fmt_pm(gas_quality.get("psnr", {}).get("mean"), gas_quality.get("psnr", {}).get("std")) + "}",
            "{" + fmt_pm(gas_quality.get("lpips", {}).get("mean"), gas_quality.get("lpips", {}).get("std")) + "}",
            "{" + fmt_pm(gas_quality.get("fvd", {}).get("mean"), gas_quality.get("fvd", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_quality.get("ssim", {}).get("mean"), pressure_quality.get("ssim", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_quality.get("psnr", {}).get("mean"), pressure_quality.get("psnr", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_quality.get("lpips", {}).get("mean"), pressure_quality.get("lpips", {}).get("std")) + "}",
            "{" + fmt_pm(pressure_quality.get("fvd", {}).get("mean"), pressure_quality.get("fvd", {}).get("std")) + "}",
        ]
        lines.append(" ".join(row))
    lines.append(f"}}\\QualityMetrics{model_name.replace('-', '').replace(' ', '')}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline models against LAViG-FLOW metrics")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-json", default=None, help="Path to write JSON results")
    parser.add_argument("--output-tex", default=None, help="Path to write LaTeX tables")
    parser.add_argument("--lavig-root", default=None, help="Path to LAViG-FLOW (for quality metrics)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    metrics_cfg = cfg.get("metrics", {})

    device = torch.device(data_cfg.get("device", "cpu"))

    stages = [int(s) for s in data_cfg.get("stages", [1, 2, 3, 4])]
    context_frames = int(data_cfg.get("context_frames", 15))
    predict_frames = int(data_cfg.get("predict_frames", 2))
    batch_size = int(data_cfg.get("batch_size", 4))
    max_samples = data_cfg.get("max_samples")
    max_samples_quality = data_cfg.get("max_samples_quality")
    targets_source = str(data_cfg.get("targets_source", "lavig")).lower()

    compute_quality = bool(metrics_cfg.get("compute_quality", True))
    clamp_quality = bool(metrics_cfg.get("clamp_quality", False))
    align_to_lavig_space = bool(metrics_cfg.get("align_to_lavig_space", True))

    gas_denorm_fn, gas_denorm_scale = _resolve_basic_denorm(metrics_cfg, "gas")
    dp_denorm_fn, dp_denorm_scale = _resolve_basic_denorm(metrics_cfg, "pressure")

    lavig_root = _as_path(args.lavig_root) if args.lavig_root else None
    if compute_quality and lavig_root is None:
        lavig_root = DEFAULT_REPO_ROOT
    if compute_quality and (lavig_root is None or not lavig_root.exists()):
        raise ValueError("--lavig-root is required when compute_quality=true")

    # Load shared inputs/targets
    gas_inputs, gas_targets = load_inputs_targets(cfg, "gas")
    dp_inputs, dp_targets = load_inputs_targets(cfg, "pressure")
    use_baseline_targets = targets_source == "baseline"

    gas_metric_ref_min = None
    gas_metric_ref_max = None
    dp_metric_ref_min = None
    dp_metric_ref_max = None
    gas_pred_metric_pre_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    dp_pred_metric_pre_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    gas_target_metric_pre_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    dp_target_metric_pre_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    gas_normalize_target_with_ref = False
    dp_normalize_target_with_ref = False

    if align_to_lavig_space:
        baseline_cfg = deepcopy(cfg)
        baseline_cfg.setdefault("data", {})
        baseline_cfg["data"]["targets_source"] = "baseline"

        # Build reference per-(sample, frame) min/max from baseline targets.
        # This mirrors the LAViG local target normalization convention.
        try:
            _, gas_baseline_targets = load_inputs_targets(baseline_cfg, "gas")
            _, dp_baseline_targets = load_inputs_targets(baseline_cfg, "pressure")
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                "metrics.align_to_lavig_space=true requires baseline validation targets "
                "(co2_data/.../sg_val_outputs.pt and dP_data/.../dP_val_outputs.pt). "
                "Set data.inputs_root (or inputs_co2_root/inputs_dp_root) and data.targets_root "
                "correctly, or disable alignment."
            ) from exc

        pressure_to_physical_fn, dp_scale, dp_shift, dp_clamp_min = _resolve_pressure_physical_transform(metrics_cfg)
        gas_metric_ref_min, gas_metric_ref_max = _compute_reference_minmax(gas_baseline_targets)
        dp_metric_ref_min, dp_metric_ref_max = _compute_reference_minmax(
            dp_baseline_targets, preprocess_fn=pressure_to_physical_fn
        )

        # Baseline model outputs are in baseline domains; map to LAViG metric domain.
        gas_pred_metric_pre_fn = None
        dp_pred_metric_pre_fn = pressure_to_physical_fn

        # If targets come from baseline files, map them the same way as predictions.
        gas_target_metric_pre_fn = None
        dp_target_metric_pre_fn = pressure_to_physical_fn if use_baseline_targets else None
        gas_normalize_target_with_ref = use_baseline_targets
        dp_normalize_target_with_ref = use_baseline_targets

        # In aligned mode, metrics are evaluated in [-1, 1].
        gas_denorm_fn = None
        dp_denorm_fn = None
        gas_data_range = 2.0
        dp_data_range = 2.0

        clamp_msg = "none" if dp_clamp_min is None else f"{dp_clamp_min:.6f}"
        print(
            "[align] Enabled baseline->LAViG metric-space mapping "
            "(per-sample/per-frame min-max from baseline targets)."
        )
        print(
            f"[align] Pressure standardized->physical uses scale={dp_scale:.12f}, "
            f"shift={dp_shift:.12f}, clamp_min={clamp_msg}."
        )
    else:
        gas_data_range = _resolve_basic_data_range(
            metrics_cfg, "gas", gas_denorm_scale, gas_targets, denorm_fn=gas_denorm_fn
        )
        dp_data_range = _resolve_basic_data_range(
            metrics_cfg, "pressure", dp_denorm_scale, dp_targets, denorm_fn=dp_denorm_fn
        )

    results: Dict[str, Dict[str, Dict[int, Dict[str, Dict[str, Dict[str, float]]]]]] = {}

    models_cfg = cfg.get("models", {})

    for model_name, spec in models_cfg.items():
        gas_ckpt = _as_path(spec.get("gas"))
        pressure_ckpt = _as_path(spec.get("pressure"))

        if gas_ckpt is None or pressure_ckpt is None:
            print(f"Skipping {model_name}: missing gas or pressure checkpoint")
            continue

        model_dir = MODEL_DIRS.get(model_name)
        if model_dir is None:
            print(f"Skipping {model_name}: unknown model dir")
            continue

        print(f"Evaluating {model_name} ...")

        if model_name in {"conv_fno", "fno", "ufno"}:
            gas_model = load_torch_model(gas_ckpt, device)
            dp_model = load_torch_model(pressure_ckpt, device)

            def gas_pred_fn(inputs_stage, targets_stage, total_frames):
                return predict_torch_volume(gas_model, inputs_stage, batch_size, device)

            def dp_pred_fn(inputs_stage, targets_stage, total_frames):
                return predict_torch_volume(dp_model, inputs_stage, batch_size, device)

        elif model_name == "vanilla_mionet":
            gas_predictor = make_mionet_vanilla_predictor(model_dir, gas_ckpt, device)
            dp_predictor = make_mionet_vanilla_predictor(model_dir, pressure_ckpt, device)

            def gas_pred_fn(inputs_stage, targets_stage, total_frames):
                return gas_predictor(inputs_stage, targets_stage, total_frames, batch_size)

            def dp_pred_fn(inputs_stage, targets_stage, total_frames):
                return dp_predictor(inputs_stage, targets_stage, total_frames, batch_size)

        elif model_name == "fourier_mionet":
            gas_predictor = make_mionet_fourier_predictor(model_dir, gas_ckpt, device, modality="gas")
            dp_predictor = make_mionet_fourier_predictor(model_dir, pressure_ckpt, device, modality="pressure")

            def gas_pred_fn(inputs_stage, targets_stage, total_frames):
                return gas_predictor(inputs_stage, targets_stage, total_frames, batch_size)

            def dp_pred_fn(inputs_stage, targets_stage, total_frames):
                return dp_predictor(inputs_stage, targets_stage, total_frames, batch_size)

        else:
            print(f"Skipping {model_name}: unsupported")
            continue

        gas_results = evaluate_model(
            model_name,
            gas_pred_fn,
            gas_inputs,
            gas_targets,
            stages,
            context_frames,
            predict_frames,
            batch_size,
            device,
            compute_quality,
            metrics_cfg,
            lavig_root,
            max_samples,
            max_samples_quality,
            basic_denorm_fn=gas_denorm_fn,
            basic_data_range=gas_data_range,
            clamp_quality=clamp_quality,
            pred_metric_pre_fn=gas_pred_metric_pre_fn,
            target_metric_pre_fn=gas_target_metric_pre_fn,
            metric_ref_min=gas_metric_ref_min,
            metric_ref_max=gas_metric_ref_max,
            normalize_pred_with_ref=align_to_lavig_space,
            normalize_target_with_ref=gas_normalize_target_with_ref,
        )

        dp_results = evaluate_model(
            model_name,
            dp_pred_fn,
            dp_inputs,
            dp_targets,
            stages,
            context_frames,
            predict_frames,
            batch_size,
            device,
            compute_quality,
            metrics_cfg,
            lavig_root,
            max_samples,
            max_samples_quality,
            basic_denorm_fn=dp_denorm_fn,
            basic_data_range=dp_data_range,
            clamp_quality=clamp_quality,
            pred_metric_pre_fn=dp_pred_metric_pre_fn,
            target_metric_pre_fn=dp_target_metric_pre_fn,
            metric_ref_min=dp_metric_ref_min,
            metric_ref_max=dp_metric_ref_max,
            normalize_pred_with_ref=align_to_lavig_space,
            normalize_target_with_ref=dp_normalize_target_with_ref,
        )

        # Merge gas + pressure into a single dict per stage
        merged: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for stage in stages:
            merged[stage] = {
                "gas": {
                    "basic": gas_results[stage].get("basic", {}),
                    "quality": gas_results[stage].get("quality", {}),
                },
                "pressure": {
                    "basic": dp_results[stage].get("basic", {}),
                    "quality": dp_results[stage].get("quality", {}),
                },
                "evaluated_samples": gas_results[stage]["evaluated_samples"],
                "predicted_frames": gas_results[stage]["predicted_frames"],
            }
        results[model_name] = merged

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved JSON to {out_path}")

    # Optionally merge LAViG-FLOW metrics JSONs (from LAViG eval scripts)
    lavig_cfg = cfg.get("lavig_flow", {}) or {}
    metrics_dir = _as_path(lavig_cfg.get("metrics_dir"))
    quality_dir = _as_path(lavig_cfg.get("quality_dir"))
    if metrics_dir is None:
        metrics_dir = DEFAULT_COMPARE_ROOT / "results" / "lavig_flow"
    if quality_dir is None:
        quality_dir = metrics_dir
    lavig_results = _merge_lavig_metrics(stages, metrics_dir, quality_dir)
    if lavig_results:
        results["lavig_flow"] = lavig_results

    if args.output_tex:
        out_path = Path(args.output_tex)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tex_parts: List[str] = []
        for model_name, model_results in results.items():
            tex_parts.append(render_tables(model_name, model_results))
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(tex_parts))
        print(f"Saved LaTeX tables to {out_path}")


if __name__ == "__main__":
    main()
