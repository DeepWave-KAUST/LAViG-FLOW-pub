###############################################################################
# Autoregressive Sample Visualizer (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Rolls out DiTV autoregressive predictions and renders comparison grids +
#   videos for CO₂ gas saturation and pressure build-up. Inspired by visualization
#   tooling in Video Diffusion projects.
###############################################################################

# --------------------------------------------------
# Imports
# --------------------------------------------------

import argparse
import os
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

matplotlib.use("Agg")

from paths import JOINT_DITV_ROOT
from joint_evaluation_dataset import JointEvaluationDataset
from evaluate_autoregressive import (
    build_model,
    build_scheduler,
    decode_joint_latents,
    generate_joint_predictions,
    load_config,
    pad_latents,
    set_seed,
)
from multi_gpu_train_ditv import _encode_gas, _encode_pressure
from utils import build_autoencoders, resolve_path
from matplotlib import colors as mcolors


# --------------------------------------------------
# Denormalization helpers
# --------------------------------------------------

def denorm_field(x_norm: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x_norm)
    assert x.dim() == 5 and x.shape[2] == 1, f"Expected [B, T, 1, H, W], got {x.shape}"

    B, T, _, _, _ = x.shape
    mn = torch.as_tensor(mn, dtype=x.dtype, device=x.device)
    mx = torch.as_tensor(mx, dtype=x.dtype, device=x.device)

    if mn.dim() == 1:
        mn = mn.view(1, T, 1, 1, 1).expand(B, -1, -1, -1, -1)
        mx = mx.view(1, T, 1, 1, 1).expand(B, -1, -1, -1, -1)
    elif mn.dim() == 2:
        mn = mn.view(B, T, 1, 1, 1)
        mx = mx.view(B, T, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected min/max shape: {mn.shape}")

    rng = mx - mn
    safe_rng = torch.where(rng == 0, torch.ones_like(rng), rng)
    x_phys = ((x + 1.0) / 2.0) * safe_rng + mn
    x_phys = torch.where(rng == 0, mn, x_phys)
    return x_phys


# --------------------------------------------------
# CLI arguments
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize joint autoregressive predictions for gas saturation and pressure.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(JOINT_DITV_ROOT / "joint_eval.yaml"),
        help="Evaluation configuration file.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the validation sample to visualize.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="Number of autoregressive chunks to roll out.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda:0'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for figures. Defaults to config infer output dir.",
    )
    parser.add_argument(
        "--context-visuals",
        type=int,
        default=None,
        help="Number of context frames to display in the visualization (default: all).",
    )
    return parser.parse_args()


# --------------------------------------------------
# Visualization helpers
# --------------------------------------------------

def _extend_stats(stat_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    stat_tensor = torch.as_tensor(stat_tensor)
    if stat_tensor.shape[0] >= target_len:
        return stat_tensor[:target_len]
    pad = stat_tensor[-1:].expand(target_len - stat_tensor.shape[0], *stat_tensor.shape[1:])
    return torch.cat([stat_tensor, pad], dim=0)


def _render_comparison(
    context_phys: np.ndarray,
    predicted_phys: np.ndarray,
    ground_truth_phys: np.ndarray,
    out_path: Path,
    chunk_count: int,
    sample_idx: int,
    context_frames_to_show: Optional[int],
    colorbar_label: str,
    error_label: str,
    mask_fn: Callable[[np.ndarray], np.ndarray],
    err_vmax: Optional[float] = None,
) -> None:
    context_frames_to_show = (
        context_phys.shape[0]
        if context_frames_to_show is None
        else min(context_frames_to_show, context_phys.shape[0])
    )
    pred_frames = predicted_phys.shape[0]

    cols = context_frames_to_show + pred_frames
    rows = 4  # context, predicted, ground truth, absolute error

    dx = np.cumsum(3.5938 * np.power(1.035012, np.arange(context_phys.shape[-1]))) + 0.1
    X, Y_full = np.meshgrid(dx, np.linspace(0, 208, context_phys.shape[-2]))

    mask = mask_fn(context_phys[0])
    thickness = int(mask[:, 0].sum())
    if thickness <= 0:
        thickness = context_phys.shape[-2]
    Y = Y_full[:thickness, :]

    all_values = np.concatenate(
        [
            context_phys[:, :thickness, :].reshape(-1),
            predicted_phys[:, :thickness, :].reshape(-1),
            ground_truth_phys[:, :thickness, :].reshape(-1),
        ]
    )
    vmin = float(all_values.min())
    vmax = float(all_values.max())

    abs_err = np.abs(predicted_phys[:, :thickness, :] - ground_truth_phys[:, :thickness, :])
    if err_vmax is None:
        err_max = float(abs_err.max()) if abs_err.size else 0.0
        err_vmax = err_max if err_max > 0 else 1e-6
    err_vmin = 0.0

    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 7.0), constrained_layout=True)
    axes = np.atleast_2d(axes)

    pcm = None
    pcm_err = None
    for col in range(cols):
        ax_context = axes[0, col]
        if col < context_frames_to_show:
            frame_idx = col
            frame = context_phys[frame_idx, :thickness, :]
            pcm = ax_context.pcolor(
                X[:thickness, :],
                Y,
                np.flipud(frame),
                shading="auto",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
            )
            ax_context.set_title(f"Context {frame_idx + 1}", fontsize=10, fontweight="bold")
            ax_context.set_xlim([0, 1500])
            ax_context.axis("off")
        else:
            ax_context.axis("off")

        pred_idx = col - context_frames_to_show

        ax_pred = axes[1, col]
        ax_gt = axes[2, col]
        ax_err = axes[3, col]

        if 0 <= pred_idx < pred_frames:
            pred_frame = predicted_phys[pred_idx, :thickness, :]
            gt_frame = ground_truth_phys[pred_idx, :thickness, :]
            err_frame = np.abs(pred_frame - gt_frame)

            pcm = ax_pred.pcolor(
                X[:thickness, :],
                Y,
                np.flipud(pred_frame),
                shading="auto",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
            )
            ax_pred.set_title(f"Pred {pred_idx + 1}", fontsize=10, fontweight="bold")
            ax_pred.set_xlim([0, 1500])
            pcm = ax_gt.pcolor(
                X[:thickness, :],
                Y,
                np.flipud(gt_frame),
                shading="auto",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
            )
            ax_gt.set_title(f"GT {pred_idx + 1}", fontsize=10, fontweight="bold")
            ax_gt.set_xlim([0, 1500])
            ax_pred.axis("off")
            ax_gt.axis("off")

            pcm_err = ax_err.pcolor(
                X[:thickness, :],
                Y,
                np.flipud(err_frame),
                shading="auto",
                cmap="magma",
                vmin=err_vmin,
                vmax=err_vmax,
            )
            ax_err.set_title(f"|Err| {pred_idx + 1}", fontsize=10, fontweight="bold")
            ax_err.set_xlim([0, 1500])
            ax_err.axis("off")
        else:
            ax_pred.axis("off")
            ax_gt.axis("off")
            ax_err.axis("off")

    fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.18, wspace=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pcm is not None:
        colorbar_path = out_path.with_name(f"{out_path.stem}_colorbar.png")
        fig_cbar, ax_cbar = plt.subplots(figsize=(4, 0.5))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_new = fig_cbar.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="jet"),
            cax=ax_cbar,
            orientation="horizontal",
        )
        cbar_new.set_label(colorbar_label, fontsize=10, fontweight="bold")
        fig_cbar.savefig(colorbar_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_cbar)

    if pcm_err is not None:
        colorbar_err_path = out_path.with_name(f"{out_path.stem}_colorbar_error.png")
        fig_cbar_e, ax_cbar_e = plt.subplots(figsize=(4, 0.5))
        norm_e = mcolors.Normalize(vmin=err_vmin, vmax=err_vmax)
        cbar_err = fig_cbar_e.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm_e, cmap="magma"),
            cax=ax_cbar_e,
            orientation="horizontal",
        )
        cbar_err.set_label(error_label, fontsize=10, fontweight="bold")
        cbar_err.ax.tick_params(labelsize=9)
        fig_cbar_e.savefig(colorbar_err_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_cbar_e)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_config(str(config_path))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed))

    dataset_cfg = config["dataset_params"]
    autoreg_cfg = dataset_cfg["autoregressive"]
    context_frames = int(autoreg_cfg["context_frames"])
    predict_frames = int(autoreg_cfg["predict_frames"])

    val_paths = dataset_cfg["file_map"]["val"]
    dataset = JointEvaluationDataset(
        gas_path=resolve_path(val_paths["gas"], config_dir),
        pressure_path=resolve_path(val_paths["pressure"], config_dir),
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"Sample index {args.sample_index} out of range (0, {len(dataset)-1}).")

    record = dataset[args.sample_index]
    gas_seq = record["gas"].unsqueeze(0).to(device)
    pressure_seq = record["pressure"].unsqueeze(0).to(device)

    total_frames = context_frames + predict_frames * args.chunks
    if gas_seq.shape[1] < total_frames or pressure_seq.shape[1] < total_frames:
        raise ValueError(
            f"Sample has only {gas_seq.shape[1]} frames; need at least {total_frames} for chunks={args.chunks}."
        )

    gas_model, pressure_model, _, _ = build_autoencoders(config, device, config_dir)
    scheduler = build_scheduler(config["diffusion_params"], config.get("rf_params", {}))
    model, frame_height, frame_width, latent_h, latent_w = build_model(config, config_dir, device)

    context_gas = gas_seq[:, :context_frames]
    context_pressure = pressure_seq[:, :context_frames]
    gt_gas = gas_seq[:, :total_frames]
    gt_pressure = pressure_seq[:, :total_frames]

    generation_start = None
    with torch.inference_mode():
        context_gas_latents = _encode_gas(context_gas, gas_model)
        context_pressure_latents = _encode_pressure(context_pressure, pressure_model)
        context_latents = torch.cat([context_gas_latents, context_pressure_latents], dim=2)
        context_latents = pad_latents(context_latents, frame_height, frame_width)

        generation_start = time.perf_counter()

        generated_latents = generate_joint_predictions(
            model=model,
            scheduler=scheduler,
            config=config,
            frame_height=frame_height,
            frame_width=frame_width,
            latent_h=latent_h,
            latent_w=latent_w,
            batch_size=1,
            num_chunks=args.chunks,
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

    generation_time = time.perf_counter() - generation_start
    print(
        f"Chunk {args.chunks}: generation completed in {generation_time:.2f}s (sampling + decode, excludes plotting)."
    )

    gas_pred = gas_pred[:, :total_frames]
    pressure_pred = pressure_pred[:, :total_frames]

    gas_stats = torch.load(resolve_path(dataset_cfg["stats_paths"]["gas"], config_dir), map_location="cpu")
    pressure_stats = torch.load(resolve_path(dataset_cfg["stats_paths"]["pressure"], config_dir), map_location="cpu")

    gas_global_min = torch.as_tensor(gas_stats["global_min"], device=device, dtype=gas_pred.dtype)
    gas_global_max = torch.as_tensor(gas_stats["global_max"], device=device, dtype=gas_pred.dtype)
    pressure_global_min = torch.as_tensor(pressure_stats["global_min"], device=device, dtype=pressure_pred.dtype)
    pressure_global_max = torch.as_tensor(pressure_stats["global_max"], device=device, dtype=pressure_pred.dtype)

    gas_total_min = _extend_stats(gas_global_min, total_frames)
    gas_total_max = _extend_stats(gas_global_max, total_frames)
    pressure_total_min = _extend_stats(pressure_global_min, total_frames)
    pressure_total_max = _extend_stats(pressure_global_max, total_frames)

    pred_frames = predict_frames * args.chunks

    gas_context_min = gas_total_min[:context_frames]
    gas_context_max = gas_total_max[:context_frames]
    gas_suffix_min = gas_total_min[context_frames : context_frames + pred_frames]
    gas_suffix_max = gas_total_max[context_frames : context_frames + pred_frames]

    pressure_context_min = pressure_total_min[:context_frames]
    pressure_context_max = pressure_total_max[:context_frames]
    pressure_suffix_min = pressure_total_min[context_frames : context_frames + pred_frames]
    pressure_suffix_max = pressure_total_max[context_frames : context_frames + pred_frames]

    context_gas_phys = denorm_field(context_gas[:, :context_frames], gas_context_min, gas_context_max)[0, :, 0].detach().cpu().numpy()
    gas_pred_suffix = gas_pred[:, context_frames : context_frames + pred_frames]
    gas_gt_suffix = gt_gas[:, context_frames : context_frames + pred_frames]
    gas_pred_phys = denorm_field(gas_pred_suffix, gas_suffix_min, gas_suffix_max)[0, :, 0].detach().cpu().numpy()
    gas_gt_phys = denorm_field(gas_gt_suffix, gas_suffix_min, gas_suffix_max)[0, :, 0].detach().cpu().numpy()

    context_pressure_phys = denorm_field(context_pressure[:, :context_frames], pressure_context_min, pressure_context_max)[0, :, 0].detach().cpu().numpy()
    pressure_pred_suffix = pressure_pred[:, context_frames : context_frames + pred_frames]
    pressure_gt_suffix = gt_pressure[:, context_frames : context_frames + pred_frames]
    pressure_pred_phys = denorm_field(pressure_pred_suffix, pressure_suffix_min, pressure_suffix_max)[0, :, 0].detach().cpu().numpy()
    pressure_gt_phys = denorm_field(pressure_gt_suffix, pressure_suffix_min, pressure_suffix_max)[0, :, 0].detach().cpu().numpy()

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        config.get("infer_params", {}).get("output_dir", ".")
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gas_out_dir = output_dir / "gas" / f"chunk_{args.chunks}"
    gas_path = gas_out_dir / f"sample_{args.sample_index}.png"
    _render_comparison(
        context_phys=context_gas_phys,
        predicted_phys=gas_pred_phys,
        ground_truth_phys=gas_gt_phys,
        out_path=gas_path,
        chunk_count=args.chunks,
        sample_idx=args.sample_index,
        context_frames_to_show=args.context_visuals,
        colorbar_label="CO₂ Gas Saturation (-)",
        error_label="|Pred − GT| (-)",
        mask_fn=lambda frame: frame != 0,
        err_vmax=0.25,
    )

    pressure_out_dir = output_dir / "pressure" / f"chunk_{args.chunks}"
    pressure_path = pressure_out_dir / f"sample_{args.sample_index}.png"
    _render_comparison(
        context_phys=context_pressure_phys,
        predicted_phys=pressure_pred_phys,
        ground_truth_phys=pressure_gt_phys,
        out_path=pressure_path,
        chunk_count=args.chunks,
        sample_idx=args.sample_index,
        context_frames_to_show=args.context_visuals,
        colorbar_label="Pressure Build-Up (bar)",
        error_label="|Pred − GT| (bar)",
        mask_fn=lambda frame: frame > 20,
        err_vmax=None,
    )

    print(f"Saved visualizations:\n - {gas_path}\n - {pressure_path}")


if __name__ == "__main__":
    main()