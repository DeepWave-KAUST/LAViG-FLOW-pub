###############################################################################
# Autoregressive Evaluation (CO₂ gas saturation + pressure build-up) – 2025
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Loads trained autoencoders + DiTV diffusion models, performs chunked
#   autoregressive rollouts, and reports reconstruction metrics. Uses the same
#   architecture stack as the training scripts.
###############################################################################

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from joint_evaluation_dataset import JointEvaluationDataset
from paths import JOINT_DITV_ROOT
from multi_gpu_train_ditv import _encode_gas, _encode_pressure
from rf_scheduler import RFlowScheduler
from utils import (
    DITVideo,
    build_autoencoders,
    compute_latent_hw,
    resolve_path,
)


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python/torch RNGs for reproducible evaluation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def build_scheduler(diffusion_cfg: dict, rf_cfg: dict) -> RFlowScheduler:
    """Instantiate the reverse-flow scheduler from config dictionaries."""
    return RFlowScheduler(
        num_timesteps=diffusion_cfg["num_timesteps"],
        num_sampling_steps=rf_cfg.get("num_sampling_steps", 30),
        sample_method=rf_cfg.get("sample_method", "uniform"),
        use_discrete_timesteps=rf_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=rf_cfg.get("use_timestep_transform", False),
        transform_scale=rf_cfg.get("transform_scale", 1.0),
        pa_vdm=rf_cfg.get("pa_vdm", False),
        noise_pattern=rf_cfg.get("noise_pattern", "linear"),
        linear_variance_scale=rf_cfg.get("linear_variance_scale", 0.1),
        linear_shift_scale=rf_cfg.get("linear_shift_scale", 0.3),
        latent_chunk_size=rf_cfg.get("latent_chunk_size", 1),
        keep_x0=rf_cfg.get("keep_x0", False),
        variable_length=rf_cfg.get("variable_length", False),
    )


def build_model(
    config: dict,
    config_dir: Path,
    device: torch.device,
) -> Tuple[DITVideo, int, int, int, int]:
    """Load DiTV weights and return the model plus latent sizes."""
    dataset_cfg = config["dataset_params"]
    auto_cfg_gas = config["autoencoder_params"]["gas"]
    auto_cfg_pressure = config["autoencoder_params"]["pressure"]
    dit_cfg = config["ditv_params"]
    train_cfg = config["train_params"]

    latent_h, latent_w = compute_latent_hw(dataset_cfg, auto_cfg_gas, auto_cfg_pressure)
    patch = dit_cfg["patch_size"]
    frame_height = math.ceil(latent_h / patch) * patch
    frame_width = math.ceil(latent_w / patch) * patch
    total_channels = auto_cfg_gas["z_channels"] + auto_cfg_pressure["z_channels"]

    model = DITVideo(
        frame_height=frame_height,
        frame_width=frame_width,
        im_channels=total_channels,
        num_frames=dataset_cfg["num_frames"],
        config=dit_cfg,
    ).to(device)

    ckpt_path = resolve_path(train_cfg["ditv_ckpt_name"], config_dir)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, frame_height, frame_width, latent_h, latent_w


def pad_latents(latents: torch.Tensor, frame_height: int, frame_width: int) -> torch.Tensor:
    """Zero-pad latent maps to match the DiTV spatial resolution."""
    pad_h = frame_height - latents.shape[-2]
    pad_w = frame_width - latents.shape[-1]
    if pad_h or pad_w:
        latents = F.pad(latents, (0, pad_w, 0, pad_h))
    return latents


def generate_joint_predictions(
    model: DITVideo,
    scheduler: RFlowScheduler,
    config: dict,
    frame_height: int,
    frame_width: int,
    latent_h: int,
    latent_w: int,
    batch_size: int,
    num_chunks: int,
    device: torch.device,
    context_latents: Optional[torch.Tensor],
) -> torch.Tensor:
    """Sample autoregressive latent trajectories chunk by chunk."""
    dataset_cfg = config["dataset_params"]
    dit_cfg = config["ditv_params"]
    auto_cfg_gas = config["autoencoder_params"]["gas"]
    auto_cfg_pressure = config["autoencoder_params"]["pressure"]

    autoreg_cfg = dataset_cfg["autoregressive"]
    context_frames = int(autoreg_cfg["context_frames"])

    total_chunk_frames = dataset_cfg["num_frames"]
    total_channels = auto_cfg_gas["z_channels"] + auto_cfg_pressure["z_channels"]

    model_kwargs = {
        "height": torch.tensor([frame_height], device=device, dtype=torch.float32),
        "width": torch.tensor([frame_width], device=device, dtype=torch.float32),
        "num_frames": torch.tensor([total_chunk_frames], device=device, dtype=torch.float32),
    }

    def model_wrapper(x, timesteps, **kwargs):
        x_time_last = x.permute(0, 2, 1, 3, 4)
        out = model(
            x_time_last,
            timesteps,
            num_images=dataset_cfg.get("num_images_train", 0),
        )
        return out.permute(0, 2, 1, 3, 4)

    initial_context = pad_latents(context_latents, frame_height, frame_width) if context_latents is not None else None
    context_buffer = initial_context

    clamp_initial_context = True
    retain_context = True

    assembled_latents = None

    for chunk_idx in range(num_chunks):
        latents = torch.randn(
            (batch_size, total_chunk_frames, total_channels, frame_height, frame_width),
            device=device,
        )

        if context_buffer is not None:
            latents[:, :context_frames] = context_buffer

        timesteps = scheduler.prepare_inference_timesteps(
            batch_size,
            device=device,
            model_kwargs=model_kwargs,
        )

        for step_idx, t_b in enumerate(timesteps):
            if step_idx < len(timesteps) - 1:
                dt = (t_b - timesteps[step_idx + 1]) / config["diffusion_params"]["num_timesteps"]
            else:
                dt = t_b / config["diffusion_params"]["num_timesteps"]

            latents = scheduler.inference_step(
                model_wrapper,
                latents,
                t_b,
                dt,
                model_kwargs=model_kwargs,
                channel_last=True,
            )
            force_context = context_buffer is not None and (chunk_idx > 0 or clamp_initial_context)
            if force_context:
                latents[:, :context_frames] = context_buffer

        if assembled_latents is None:
            assembled_latents = latents
        else:
            assembled_latents = torch.cat([assembled_latents, latents[:, context_frames:]], dim=1)

        context_buffer = assembled_latents[:, -context_frames:].detach().clone()

    if assembled_latents is None:
        raise RuntimeError("Failed to generate latents.")

    if frame_height != latent_h or frame_width != latent_w:
        assembled_latents = assembled_latents[..., :latent_h, :latent_w]

    return assembled_latents


def decode_joint_latents(
    latents: torch.Tensor,
    gas_model,
    pressure_model,
    config: dict,
    latent_h: int,
    latent_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split joint latents and decode them back to pixel space."""
    auto_cfg_gas = config["autoencoder_params"]["gas"]
    auto_cfg_pressure = config["autoencoder_params"]["pressure"]
    dataset_cfg = config["dataset_params"]

    gas_channels = auto_cfg_gas["z_channels"]
    pressure_channels = auto_cfg_pressure["z_channels"]
    total_channels = gas_channels + pressure_channels
    if latents.shape[2] != total_channels:
        raise ValueError(
            f"Latent channels mismatch: expected {total_channels}, got {latents.shape[2]}"
        )

    gas_latents = latents[:, :, :gas_channels]
    pressure_latents = latents[:, :, gas_channels:]

    gas_decoded = gas_model.decode(
        gas_latents.reshape(-1, gas_channels, latent_h, latent_w)
    ).reshape(
        latents.shape[0],
        latents.shape[1],
        -1,
        dataset_cfg["frame_height"],
        dataset_cfg["frame_width"],
    )

    pressure_decoded = pressure_model.decode(
        pressure_latents.reshape(-1, pressure_channels, latent_h, latent_w)
    ).reshape(
        latents.shape[0],
        latents.shape[1],
        -1,
        dataset_cfg["frame_height"],
        dataset_cfg["frame_width"],
    )

    gas_decoded = torch.clamp(gas_decoded, -1.0, 1.0)
    pressure_decoded = torch.clamp(pressure_decoded, -1.0, 1.0)

    return gas_decoded, pressure_decoded


# --------------------------------------------------
# Metric computation helpers
# --------------------------------------------------

def compute_batch_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    data_range: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Return per-sample MSE/MAE/RMSE/PSNR tensors."""
    diff = predictions - targets
    mse = diff.pow(2).flatten(start_dim=1).mean(dim=1)
    mae = diff.abs().flatten(start_dim=1).mean(dim=1)
    rmse = torch.sqrt(mse)

    mse_clamped = torch.clamp(mse, min=1e-10)
    max_val = torch.tensor(data_range, device=predictions.device)
    psnr = 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse_clamped)
    psnr = torch.where(torch.isfinite(psnr), psnr, torch.full_like(psnr, 100.0))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
    }


def aggregate_metrics(metric_lists: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate running metric lists into mean/std summaries."""
    summary = {}
    for key, values in metric_lists.items():
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[key] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
        }
    return summary


# --------------------------------------------------
# Autoregressive evaluation loop
# --------------------------------------------------

def evaluate_chunks(
    config: dict,
    config_dir: Path,
    chunks: Sequence[int],
    max_samples: Optional[int],
    batch_size: int,
    device: torch.device,
    progress: bool = True,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Run evaluation for each chunk count and return aggregated metrics."""
    dataset_cfg = config["dataset_params"]
    autoreg_cfg = dataset_cfg.get("autoregressive")
    if not autoreg_cfg:
        raise ValueError("dataset_params.autoregressive section is required for this evaluation.")

    context_frames = int(autoreg_cfg["context_frames"])
    predict_frames = int(autoreg_cfg["predict_frames"])

    file_map = dataset_cfg["file_map"]
    val_paths = file_map.get("val")
    if not val_paths:
        raise ValueError("Evaluation requires dataset_params.file_map.val entries.")

    dataset = JointEvaluationDataset(
        gas_path=resolve_path(val_paths["gas"], config_dir),
        pressure_path=resolve_path(val_paths["pressure"], config_dir),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    gas_model, pressure_model, _, _ = build_autoencoders(config, device, config_dir)

    scheduler = build_scheduler(config["diffusion_params"], config.get("rf_params", {}))
    model, frame_height, frame_width, latent_h, latent_w = build_model(config, config_dir, device)

    results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for chunk_count in chunks:
        if chunk_count < 1:
            raise ValueError("Chunk counts must be >= 1.")

        total_frames = context_frames + predict_frames * chunk_count
        if total_frames > dataset.num_frames:
            raise ValueError(
                f"Requested total frames {total_frames} (context {context_frames}, predict {predict_frames}, "
                f"chunks {chunk_count}) but evaluation clips have only {dataset.num_frames} frames."
            )

        metric_accumulators_gas: Dict[str, List[float]] = defaultdict(list)
        metric_accumulators_pressure: Dict[str, List[float]] = defaultdict(list)
        processed = 0

        iterator = loader if not progress else tqdm(loader, desc=f"Chunks={chunk_count}", leave=False)
        for batch in iterator:
            gas_seq = batch["gas"].to(device)  # [B, T, C, H, W]
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

            for key, tensor in gas_metrics.items():
                metric_accumulators_gas[key].extend(tensor.detach().cpu().tolist())
            for key, tensor in pressure_metrics.items():
                metric_accumulators_pressure[key].extend(tensor.detach().cpu().tolist())

            processed += limit

        if processed == 0:
            raise ValueError(
                f"No validation samples provided at least {total_frames} frames for chunk={chunk_count}."
            )

        results[chunk_count] = {
            "gas": aggregate_metrics(metric_accumulators_gas),
            "pressure": aggregate_metrics(metric_accumulators_pressure),
            "evaluated_samples": {"count": processed},
            "predicted_frames": {"count": predict_frames * chunk_count},
        }

    return results


# --------------------------------------------------
# CLI plumbing
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Configure CLI arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Joint autoregressive evaluation for gas saturation + pressure build-up.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(JOINT_DITV_ROOT / "joint_eval.yaml"),
        help="Path to joint evaluation configuration file.",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="1",
        help="Comma-separated list of chunk counts to evaluate (e.g. '1,2,3').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of validation sequences to evaluate (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for autoregressive evaluation.",
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
    """Entry point for running the autoregressive evaluation workflow."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    config = load_config(str(config_path))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed))

    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]

    results = evaluate_chunks(
        config=config,
        config_dir=config_dir,
        chunks=chunks,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=device,
        progress=True,
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
            for key, label in (("mse", "MSE"), ("mae", "MAE"), ("rmse", "RMSE"), ("psnr", "PSNR")):
                stats = modality_metrics.get(key)
                if stats is None:
                    continue
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is None or std is None:
                    continue
                parts.append(f"{label}={mean:.6f}±{std:.6f}")

            if parts:
                frame_text = (
                    f"Metrics over these total predicted frames={predicted_frames}"
                    if predicted_frames is not None
                    else "Metrics"
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