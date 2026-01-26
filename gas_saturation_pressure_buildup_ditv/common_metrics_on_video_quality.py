###############################################################################
# Common Video Quality Metrics Runner (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Evaluate CO₂ gas/pressure predictions with FVD, SSIM, LPIPS, PSNR, etc.
#   The metric implementations are adapted from the public StyleGAN-V release
#   (https://github.com/universome/stylegan-v/tree/master) and rely on the
#   pretrained detector weights distributed with that project. Pretrained
#   I3D/LPIPS detectors are downloaded into `metric_detectors/` using the
#   original StyleGAN-V URLs.
###############################################################################

import argparse
import importlib
import json
import math
import os
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paths import JOINT_DITV_ROOT
from joint_evaluation_dataset import JointEvaluationDataset
from evaluate_autoregressive import (
    aggregate_metrics,
    build_model,
    build_scheduler,
    compute_batch_metrics,
    decode_joint_latents,
    generate_joint_predictions,
    load_config,
    pad_latents,
    set_seed,
)
from multi_gpu_train_ditv import _encode_gas, _encode_pressure
from utils import build_autoencoders, compute_latent_hw, resolve_path


# Detector weights follow the StyleGAN-V release (see repo above).
FVD_DETECTOR_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
FVD_DETECTOR_FILENAME = "i3d_torchscript.pt"
MIN_FVD_FRAMES = 16
LOCAL_DETECTORS_DIR = Path(__file__).resolve().parent / "metric_detectors"

LOCAL_DETECTORS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# StyleGAN-V metric integration
# --------------------------------------------------

# External StyleGAN-V repo location (set via env or CLI when not adjacent).
STYLEGAN_ENV_VAR = "STYLEGAN_V_SRC"

_STYLEGAN_METRIC_UTILS = None
_STYLEGAN_SEARCHED_PATHS: set = set()


def _import_stylegan_metric_utils(stylegan_src: Optional[str] = None):
    """Locate and import StyleGAN-V metric utilities for LPIPS/FVD helpers."""
    global _STYLEGAN_METRIC_UTILS
    if _STYLEGAN_METRIC_UTILS is not None:
        return _STYLEGAN_METRIC_UTILS

    def _canonicalize(path: pathlib.Path) -> pathlib.Path:
        return path.expanduser().resolve(strict=False)

    search_roots: List[pathlib.Path] = []
    seen_root_keys: set = set()

    def _queue_root(path: pathlib.Path) -> None:
        canon = _canonicalize(path)
        key = str(canon)
        if key not in seen_root_keys:
            search_roots.append(canon)
            seen_root_keys.add(key)

    if stylegan_src:
        _queue_root(pathlib.Path(stylegan_src))

    env_path = os.environ.get(STYLEGAN_ENV_VAR)
    if env_path:
        _queue_root(pathlib.Path(env_path))

    script_path = pathlib.Path(__file__).resolve()
    for parent in script_path.parents:
        candidate_root = parent / "stylegan-v-main"
        _queue_root(candidate_root)
        _queue_root(candidate_root / "src")

    for root in search_roots:
        candidates = [root]
        if root.name != "src":
            candidates.append(_canonicalize(root / "src"))

        for path in candidates:
            if not path.is_dir():
                continue
            key = str(path)
            if key in _STYLEGAN_SEARCHED_PATHS:
                continue
            _STYLEGAN_SEARCHED_PATHS.add(key)
            if key not in sys.path:
                sys.path.insert(0, key)
            try:
                module = importlib.import_module("metrics.metric_utils")
            except ModuleNotFoundError:
                continue
            _STYLEGAN_METRIC_UTILS = module
            return module

    raise RuntimeError(
        "Unable to import stylegan-v metric utilities. Provide --stylegan-src, set STYLEGAN_V_SRC, "
        "or ensure the repository is located at ../../stylegan-v-main relative to this script."
    )


def _resolve_detector_resource(filename: str, default_url: str) -> str:
    """Resolve detector weights shipped with StyleGAN-V (download on demand)."""
    local_path = LOCAL_DETECTORS_DIR / filename
    if local_path.is_file():
        return str(local_path.resolve())
    return default_url


# --------------------------------------------------
# SSIM / perceptual helper utilities
# --------------------------------------------------

def _ensure_window_size(height: int, width: int, preferred: int = 11) -> int:
    size = min(preferred, height, width)
    if size % 2 == 0:
        size = max(size - 1, 1)
    return max(size, 3)


def _create_gaussian_window(
    channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    size = _ensure_window_size(height, width)
    sigma = size / 6.0

    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    window_2d = (gauss[:, None] * gauss[None, :]).to(dtype=dtype, device=device)
    window_2d = window_2d / window_2d.sum()

    window = window_2d.view(1, 1, size, size)
    window = window.expand(channels, 1, size, size)
    return window


def _aggregate_metric(values: List[float]) -> Dict[str, List[float]]:
    if not values:
        raise ValueError("No values collected for metric computation.")

    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    return {"value": [mean], "value_std": [std]}


def _flatten_frames(tensor: torch.Tensor, only_final: bool) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected tensor with 5 dims [B, T, C, H, W], got {tensor.shape}.")
    if only_final:
        tensor = tensor[:, -1:, ...]
    b, t, c, h, w = tensor.shape
    return tensor.reshape(b * t, c, h, w)


def calculate_ssim(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    only_final: bool = True,
    data_range: float = 1.0,
) -> Dict[str, List[float]]:
    preds = _flatten_frames(predictions, only_final)
    gts = _flatten_frames(targets, only_final)

    values: List[float] = []
    for pred, gt in zip(preds, gts):
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        _, channels, height, width = pred.shape
        window = _create_gaussian_window(
            channels=channels,
            height=height,
            width=width,
            dtype=pred.dtype,
            device=pred.device,
        )
        padding = window.shape[-1] // 2

        mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
        mu_gt = F.conv2d(gt, window, padding=padding, groups=channels)

        mu_pred_sq = mu_pred.pow(2)
        mu_gt_sq = mu_gt.pow(2)
        mu_pred_gt = mu_pred * mu_gt

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
        sigma_gt_sq = F.conv2d(gt * gt, window, padding=padding, groups=channels) - mu_gt_sq
        sigma_pred_gt = F.conv2d(pred * gt, window, padding=padding, groups=channels) - mu_pred_gt

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        numerator = (2 * mu_pred_gt + c1) * (2 * sigma_pred_gt + c2)
        denominator = (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
        ssim_map = numerator / denominator

        values.append(float(ssim_map.mean().item()))

    return _aggregate_metric(values)


def calculate_psnr(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    only_final: bool = True,
    data_range: float = 1.0,
) -> Dict[str, List[float]]:
    preds = _flatten_frames(predictions, only_final)
    gts = _flatten_frames(targets, only_final)

    mse = torch.mean((preds - gts) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)

    values = [float(val.item()) for val in psnr]
    return _aggregate_metric(values)


_LPIPS_MODELS: Dict[str, torch.nn.Module] = {}


def _get_lpips_model(device: torch.device):
    key = str(device)
    model = _LPIPS_MODELS.get(key)
    if model is None:
        try:
            import lpips  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LPIPS metric requested but the 'lpips' package is not installed. "
                "Install it with `pip install lpips`."
            ) from exc

        model = lpips.LPIPS(net="vgg").to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return model


def calculate_lpips(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    device: torch.device,
    only_final: bool = True,
) -> Dict[str, List[float]]:
    preds = _flatten_frames(predictions, only_final=only_final).to(device=device, dtype=torch.float32)
    gts = _flatten_frames(targets, only_final=only_final).to(device=device, dtype=torch.float32)

    lpips_model = _get_lpips_model(device)

    preds = preds * 2.0 - 1.0
    gts = gts * 2.0 - 1.0

    values: List[float] = []
    with torch.no_grad():
        for pred, gt in zip(preds, gts):
            distance = lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).mean()
            values.append(float(distance.item()))

    return _aggregate_metric(values)


def _to_uint8_image(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 255.0).round().clamp_(0.0, 255.0).to(torch.uint8)


def _ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 3:
        return tensor
    if tensor.shape[1] == 1:
        repeat_dims = [1] * tensor.ndim
        repeat_dims[1] = 3
        return tensor.repeat(*repeat_dims)
    raise ValueError(f"Expected tensor with 1 or 3 channels, got {tensor.shape}.")


def _calculate_frechet_distance(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    eps: float = 1e-6,
) -> float:
    if mu_pred.shape != mu_real.shape:
        raise ValueError("Mean vectors have different shapes.")
    if sigma_pred.shape != sigma_real.shape:
        raise ValueError("Covariance matrices have different shapes.")

    diff = mu_pred - mu_real
    cov_prod = sigma_pred @ sigma_real
    covmean, _ = scipy.linalg.sqrtm(cov_prod, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_pred.shape[0], dtype=np.float64) * eps
        covmean, _ = scipy.linalg.sqrtm((sigma_pred + offset) @ (sigma_real + offset), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_pred + sigma_real - 2.0 * covmean))
    return fid


def _collect_feature_stats(
    tensor: torch.Tensor,
    detector: torch.nn.Module,
    detector_kwargs: Dict[str, object],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    total = int(tensor.shape[0])
    if total == 0:
        raise ValueError("No samples provided for feature statistics.")

    sum_vec: Optional[np.ndarray] = None
    sum_cov: Optional[np.ndarray] = None
    count = 0

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = tensor[start:stop]
        batch = _ensure_min_temporal_frames(batch)
        batch = _ensure_three_channels(batch)
        with torch.no_grad():
            features = detector(batch.to(device), **detector_kwargs)
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("Detector returned an empty sequence.")
            features = features[0]
        if not isinstance(features, torch.Tensor):
            raise TypeError("Detector output is not a tensor.")
        features = features.view(features.shape[0], -1)
        batch_np = features.detach().to(dtype=torch.float64, device="cpu").numpy()

        if sum_vec is None:
            feat_dim = batch_np.shape[1]
            sum_vec = np.zeros(feat_dim, dtype=np.float64)
            sum_cov = np.zeros((feat_dim, feat_dim), dtype=np.float64)

        sum_vec += batch_np.sum(axis=0)
        sum_cov += batch_np.T @ batch_np
        count += batch_np.shape[0]

    if count == 0 or sum_vec is None or sum_cov is None:
        raise ValueError("Failed to accumulate detector features.")

    mean = sum_vec / count
    cov = sum_cov / count - np.outer(mean, mean)
    return mean, cov


def _match_temporal_frames(videos: torch.Tensor, target_frames: int) -> torch.Tensor:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive.")
    current = videos.shape[2]
    if current == target_frames:
        return videos
    indices = torch.linspace(0, current - 1, target_frames, dtype=torch.float32, device=videos.device)
    indices = indices.round().to(dtype=torch.long)
    indices = torch.clamp(indices, 0, current - 1)
    return videos.index_select(2, indices)


def _ensure_min_temporal_frames(videos: torch.Tensor, min_frames: int = MIN_FVD_FRAMES) -> torch.Tensor:
    frames = videos.shape[2]
    if frames >= min_frames:
        return videos
    if frames == 0:
        raise ValueError("Videos tensor must contain at least one frame for FVD computation.")
    repeat = math.ceil(min_frames / frames)
    videos = videos.repeat(1, 1, repeat, 1, 1)
    return videos[:, :, :min_frames].contiguous()


def _prepare_videos_for_fvd(tensor: torch.Tensor, target_frames: Optional[int]) -> torch.Tensor:
    videos = tensor.to(dtype=torch.float32)
    videos = _to_uint8_image(videos)
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # [N, C, T, H, W]
    min_required_frames = MIN_FVD_FRAMES
    if target_frames is not None:
        target_frames = max(target_frames, min_required_frames)
        videos = _match_temporal_frames(videos, target_frames)
    videos = _ensure_min_temporal_frames(videos, min_frames=min_required_frames)
    return videos


def calculate_fvd(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    device: torch.device,
    stylegan_src: Optional[str],
    batch_size: int,
    target_frames: Optional[int] = None,
) -> float:
    if targets.shape != predictions.shape:
        raise ValueError("Predictions and targets must have the same shape for FVD computation.")

    metric_utils = _import_stylegan_metric_utils(stylegan_src)
    detector_resource = _resolve_detector_resource(FVD_DETECTOR_FILENAME, FVD_DETECTOR_URL)
    detector = metric_utils.get_feature_detector(
        url=detector_resource,
        device=device,
        num_gpus=1,
        rank=0,
        verbose=False,
    )
    detector_kwargs = {"rescale": True, "resize": True, "return_features": True}

    preds_unit = to_unit_interval(predictions.clone()).cpu()
    gts_unit = to_unit_interval(targets.clone()).cpu()

    pred_videos = _prepare_videos_for_fvd(preds_unit, target_frames)
    target_videos = _prepare_videos_for_fvd(gts_unit, target_frames)

    if pred_videos.shape[2] != target_videos.shape[2]:
        raise ValueError(
            "Predictions and targets must share the same number of frames after alignment for FVD."
        )

    mu_pred, sigma_pred = _collect_feature_stats(
        pred_videos, detector, detector_kwargs, batch_size, device
    )
    mu_real, sigma_real = _collect_feature_stats(
        target_videos, detector, detector_kwargs, batch_size, device
    )

    return _calculate_frechet_distance(mu_pred, sigma_pred, mu_real, sigma_real)


def to_unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] latent-space frames to [0, 1] for metric functions."""
    return tensor.add(1.0).mul_(0.5).clamp_(0.0, 1.0)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lpips_device: torch.device,
    *,
    stylegan_src: Optional[str] = None,
    fvd_batch_size: int = 16,
    compute_fvd: bool = True,
    fvd_target_frames: Optional[int] = None,
) -> Dict[str, float]:
    """
    predictions / targets: [B, T, C, H, W] in [-1, 1]
    Returns mean/std dict for SSIM/PSNR/LPIPS and optionally FVD.
    """
    predictions = predictions.detach()
    targets = targets.detach()

    preds_unit = to_unit_interval(predictions.clone()).cpu()
    gts_unit = to_unit_interval(targets.clone()).cpu()

    ssim_stats = calculate_ssim(gts_unit, preds_unit, only_final=True)
    psnr_stats = calculate_psnr(gts_unit, preds_unit, only_final=True)
    lpips_stats = calculate_lpips(gts_unit, preds_unit, lpips_device, only_final=True)

    def extract(stats: Dict[str, List[float]], key: str) -> float:
        values = stats.get(key, [])
        if not values:
            raise ValueError(f"Metric helper returned empty {key}.")
        return float(values[0])

    metrics = {
        "ssim": extract(ssim_stats, "value"),
        "ssim_std": extract(ssim_stats, "value_std"),
        "psnr": extract(psnr_stats, "value"),
        "psnr_std": extract(psnr_stats, "value_std"),
        "lpips": extract(lpips_stats, "value"),
        "lpips_std": extract(lpips_stats, "value_std"),
    }

    if compute_fvd:
        fvd_value = calculate_fvd(
            targets=targets,
            predictions=predictions,
            device=lpips_device,
            stylegan_src=stylegan_src,
            batch_size=fvd_batch_size,
            target_frames=fvd_target_frames,
        )
        metrics["fvd"] = fvd_value
        metrics["fvd_std"] = 0.0

    return metrics


# --------------------------------------------------
# Autoregressive evaluation pipeline
# --------------------------------------------------

def evaluate_for_metrics(
    config: dict,
    config_dir: Path,
    chunks: Sequence[int],
    batch_size: int,
    max_samples: Optional[int],
    device: torch.device,
    *,
    stylegan_src: Optional[str],
    fvd_batch_size: int,
    compute_fvd: bool,
    fvd_target_frames: Optional[int],
) -> Dict[int, Dict[str, Dict[str, Dict[str, float]]]]:
    dataset_cfg = config["dataset_params"]
    autoreg_cfg = dataset_cfg["autoregressive"]
    context_frames = int(autoreg_cfg["context_frames"])
    predict_frames = int(autoreg_cfg["predict_frames"])

    file_map = dataset_cfg["file_map"]
    val_paths = file_map["val"]

    dataset = JointEvaluationDataset(
        gas_path=resolve_path(val_paths["gas"], config_dir),
        pressure_path=resolve_path(val_paths["pressure"], config_dir),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    gas_model, pressure_model, _, _ = build_autoencoders(config, device, config_dir)

    scheduler = build_scheduler(config["diffusion_params"], config.get("rf_params", {}))
    model, frame_height, frame_width, latent_h, latent_w = build_model(config, config_dir, device)

    results: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for chunk_count in chunks:
        total_frames = context_frames + predict_frames * chunk_count

        metric_accumulators: Dict[str, Dict[str, List[float]]] = {
            "gas": defaultdict(list),
            "pressure": defaultdict(list),
        }
        preds_storage: Dict[str, List[torch.Tensor]] = {"gas": [], "pressure": []}
        gts_storage: Dict[str, List[torch.Tensor]] = {"gas": [], "pressure": []}
        processed = 0

        for batch in loader:
            gas_seq = batch["gas"].to(device)
            pressure_seq = batch["pressure"].to(device)

            if gas_seq.shape[1] < total_frames or pressure_seq.shape[1] < total_frames:
                continue

            limit = gas_seq.shape[0]
            if max_samples is not None:
                remaining = max_samples - processed
                if remaining <= 0:
                    break
                limit = min(limit, remaining)
                gas_seq = gas_seq[:limit]
                pressure_seq = pressure_seq[:limit]

            context_gas = gas_seq[:, :context_frames]
            context_pressure = pressure_seq[:, :context_frames]
            gt_gas = gas_seq[:, context_frames: total_frames]
            gt_pressure = pressure_seq[:, context_frames: total_frames]

            with torch.no_grad():
                context_gas_latents = _encode_gas(context_gas, gas_model)
                context_pressure_latents = _encode_pressure(context_pressure, pressure_model)
                context_latents = torch.cat([context_gas_latents, context_pressure_latents], dim=2)
                context_latents = pad_latents(context_latents, frame_height, frame_width)

                generated_latents = generate_joint_predictions(
                    model=model,
                    scheduler=scheduler,
                    config=config,
                    frame_height=frame_height,
                    frame_width=frame_width,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    batch_size=limit,
                    num_chunks=chunk_count,
                    device=device,
                    context_latents=context_latents,
                )

                gas_pred, pressure_pred = decode_joint_latents(
                    generated_latents,
                    gas_model,
                    pressure_model,
                    config,
                    latent_h,
                    latent_w,
                )

                gas_pred_suffix = gas_pred[:, context_frames:context_frames + predict_frames * chunk_count]
                pressure_pred_suffix = pressure_pred[:, context_frames:context_frames + predict_frames * chunk_count]

            gas_pred_suffix = gas_pred_suffix.detach()
            pressure_pred_suffix = pressure_pred_suffix.detach()

            gas_metrics = compute_batch_metrics(gas_pred_suffix, gt_gas)
            pressure_metrics = compute_batch_metrics(pressure_pred_suffix, gt_pressure)

            for key in ("mse", "mae", "rmse", "psnr"):
                metric_accumulators["gas"][key].extend(gas_metrics[key].detach().cpu().tolist())
                metric_accumulators["pressure"][key].extend(pressure_metrics[key].detach().cpu().tolist())

            preds_storage["gas"].append(gas_pred_suffix.detach().cpu())
            gts_storage["gas"].append(gt_gas.detach().cpu())
            preds_storage["pressure"].append(pressure_pred_suffix.detach().cpu())
            gts_storage["pressure"].append(gt_pressure.detach().cpu())

            processed += limit

        if processed == 0:
            raise ValueError(
                f"No validation samples provided at least {total_frames} frames for chunk={chunk_count}."
            )

        chunk_summary: Dict[str, Dict[str, Dict[str, float]]] = {
            modality: aggregate_metrics(metrics)
            for modality, metrics in metric_accumulators.items()
        }

        for modality in ("gas", "pressure"):
            predictions = torch.cat(preds_storage[modality], dim=0)
            targets = torch.cat(gts_storage[modality], dim=0)
            adv_metrics = compute_metrics(
                predictions=predictions,
                targets=targets,
                lpips_device=device,
                stylegan_src=stylegan_src,
                fvd_batch_size=fvd_batch_size,
                compute_fvd=compute_fvd,
                fvd_target_frames=fvd_target_frames,
            )
            chunk_summary.setdefault(modality, {})
            chunk_summary[modality]["ssim"] = {"mean": adv_metrics["ssim"], "std": adv_metrics["ssim_std"]}
            chunk_summary[modality]["psnr"] = {"mean": adv_metrics["psnr"], "std": adv_metrics["psnr_std"]}
            chunk_summary[modality]["lpips"] = {"mean": adv_metrics["lpips"], "std": adv_metrics["lpips_std"]}
            if "fvd" in adv_metrics:
                chunk_summary[modality]["fvd"] = {
                    "mean": adv_metrics["fvd"],
                    "std": adv_metrics.get("fvd_std", 0.0),
                }

        chunk_summary["evaluated_samples"] = {"count": processed}
        chunk_summary["predicted_frames"] = {"count": predict_frames * chunk_count}

        results[chunk_count] = chunk_summary

    return results


# --------------------------------------------------
# CLI arguments
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute joint video quality metrics for gas saturation and pressure build-up."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(JOINT_DITV_ROOT / "joint_eval.yaml"),
        help="Evaluation configuration file.",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="1",
        help="Comma-separated list of chunk counts to evaluate (e.g. '1,2,3').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for metric computation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of validation sequences to evaluate (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string (e.g. 'cuda:0', 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--stylegan-src",
        type=str,
        default=None,
        help=(
            "Optional path to a stylegan-v 'src' directory providing metric implementations. "
            "Defaults to STYLEGAN_V_SRC or ../../stylegan-v-main/src."
        ),
    )
    parser.add_argument(
        "--fvd-batch-size",
        type=int,
        default=16,
        help="Batch size to use when extracting I3D features for FVD.",
    )
    parser.add_argument(
        "--skip-fvd",
        action="store_true",
        help="Disable FVD computation.",
    )
    parser.add_argument(
        "--fvd-num-frames",
        type=int,
        default=None,
        help="Temporally resample videos to this many frames before computing FVD (default: keep original length).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    return parser.parse_args()


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------

def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    config = load_config(str(config_path))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed))

    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]
    if not chunks:
        raise ValueError("No valid chunk counts provided.")

    latent_h, latent_w = compute_latent_hw(
        config["dataset_params"],
        config["autoencoder_params"]["gas"],
        config["autoencoder_params"]["pressure"],
    )
    _ = latent_h, latent_w  # kept for parity (explicit outputs computed in evaluate_for_metrics)

    results = evaluate_for_metrics(
        config=config,
        config_dir=config_dir,
        chunks=chunks,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=device,
        stylegan_src=args.stylegan_src,
        fvd_batch_size=args.fvd_batch_size,
        compute_fvd=not args.skip_fvd,
        fvd_target_frames=args.fvd_num_frames,
    )

    for chunk in sorted(results.keys()):
        chunk_results = results[chunk]
        predicted_frames = chunk_results.get("predicted_frames", {}).get("count")
        samples = chunk_results.get("evaluated_samples", {}).get("count")
        for modality in ("gas", "pressure"):
            modality_metrics = chunk_results.get(modality, {})
            if not modality_metrics:
                continue

            parts: List[str] = []
            for key, label in (("ssim", "SSIM"), ("psnr", "PSNR"), ("lpips", "LPIPS"), ("fvd", "FVD")):
                stats = modality_metrics.get(key)
                if stats is None:
                    continue
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is None:
                    continue
                if std is None:
                    parts.append(f"{label}={mean:.6f}")
                else:
                    parts.append(f"{label}={mean:.6f}±{std:.6f}")

            if parts:
                frame_text = (
                    f"predicted_frames={predicted_frames}"
                    if predicted_frames is not None
                    else "predicted_frames=?"
                )
                sample_text = f"samples={samples}" if samples is not None else "samples=?"
                print(
                    f"Chunk {chunk} | modality={modality} | {frame_text} | {sample_text} | "
                    + " | ".join(parts)
                )

    print(json.dumps(results, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()