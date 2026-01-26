###############################################################################
# DiTV Sampling & Visualization Utility (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Loads trained DiTV checkpoints, generates autoregressive samples, and
#   produces both video clips and high-resolution grids for CO₂ gas saturation
#   and pressure build-up. Inspired by visualization utilities in Video Diffusion.
###############################################################################

import warnings

warnings.filterwarnings("ignore")

import argparse
import math
import os
import time
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm

from paths import JOINT_DITV_ROOT
from joint_dataset import JointTargetsDataset  
from utils import (
    DITVideo,
    RFlowScheduler,
    build_autoencoders,
    compute_latent_hw,
    init_device,
    resolve_path,
)


PNG_DPI = 600
VIDEO_DPI = 200


# --------------------------------------------------
# Timing helpers
# --------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _now():
    return time.perf_counter()


# --------------------------------------------------
# Denormalization helpers
# --------------------------------------------------

def denorm_gas(x_norm, mn, mx):
    x = torch.as_tensor(x_norm)
    assert x.dim() in (4, 5), f"x_norm must be 4D/5D, got {x.shape}"

    if x.dim() == 4:
        T, C, _, _ = x.shape
        assert C == 1
        mn = torch.as_tensor(mn, dtype=x.dtype, device=x.device).view(T, 1, 1, 1)
        mx = torch.as_tensor(mx, dtype=x.dtype, device=x.device).view(T, 1, 1, 1)
    else:
        B, T, C, _, _ = x.shape
        assert C == 1
        mn = torch.as_tensor(mn, dtype=x.dtype, device=x.device)
        mx = torch.as_tensor(mx, dtype=x.dtype, device=x.device)
        if mn.dim() == 1:
            mn = mn.view(1, T, 1, 1, 1).expand(B, -1, -1, -1, -1)
            mx = mx.view(1, T, 1, 1, 1).expand(B, -1, -1, -1, -1)
        else:
            mn = mn.view(B, T, 1, 1, 1)
            mx = mx.view(B, T, 1, 1, 1)

    rng = mx - mn
    safe_rng = torch.where(rng == 0, torch.ones_like(rng), rng)
    x_phys = ((x + 1.0) / 2.0) * safe_rng + mn
    x_phys = torch.where(rng == 0, mn, x_phys)
    return x_phys


def denorm_pressure(x_norm, mn, mx):
    x = torch.as_tensor(x_norm)
    assert x.dim() in (4, 5), f"x_norm must be 4D/5D, got {x.shape}"

    if x.dim() == 4:
        T, C, _, _ = x.shape
        assert C == 1
        mn = torch.as_tensor(mn, dtype=x.dtype, device=x.device).view(T, 1, 1, 1)
        mx = torch.as_tensor(mx, dtype=x.dtype, device=x.device).view(T, 1, 1, 1)
    else:
        B, T, C, _, _ = x.shape
        assert C == 1
        mn = torch.as_tensor(mn, dtype=x.dtype, device=x.device)
        mx = torch.as_tensor(mx, dtype=x.dtype, device=x.device)
        if mn.dim() == 1:
            mn = mn.view(1, T, 1, 1, 1).expand(B, -1, -1, -1, -1)
            mx = mx.view(1, T, 1, 1, 1).expand(B, -1, -1, -1, -1)
        else:
            mn = mn.view(B, T, 1, 1, 1)
            mx = mx.view(B, T, 1, 1, 1)

    rng = mx - mn
    safe_rng = torch.where(rng == 0, torch.ones_like(rng), rng)
    x_phys = ((x + 1.0) / 2.0) * safe_rng + mn
    x_phys = torch.where(rng == 0, mn, x_phys)
    return x_phys


def extend_stats(stat_tensor, target_len):
    stat_tensor = torch.as_tensor(stat_tensor)
    if stat_tensor.shape[0] >= target_len:
        return stat_tensor[:target_len]
    pad = stat_tensor[-1:].repeat(target_len - stat_tensor.shape[0], *([1] * (stat_tensor.dim() - 1)))
    return torch.cat([stat_tensor, pad], dim=0)


# --------------------------------------------------
# Rendering helpers
# --------------------------------------------------

def _render_strip(frames, out_path):
    imgs = [np.array(Image.open(fp).convert("RGB")) for fp in frames]
    h_min = min(im.shape[0] for im in imgs)
    imgs = [
        im if im.shape[0] == h_min
        else np.array(Image.fromarray(im).resize((im.shape[1], h_min), resample=Image.BILINEAR))
        for im in imgs
    ]
    c = imgs[0].shape[2]
    spacing = 3
    spacer = 255 * np.ones((h_min, spacing, c), dtype=np.uint8)
    row_parts = []
    for idx, im in enumerate(imgs):
        row_parts.append(im)
        if idx < len(imgs) - 1:
            row_parts.append(spacer)
    strip = np.concatenate(row_parts, axis=1)
    Image.fromarray(strip).save(out_path, dpi=(PNG_DPI, PNG_DPI))
    return strip


def _render_grid(strips, out_path):
    if not strips:
        return
    grid_w = max(s.shape[1] for s in strips)
    c = strips[0].shape[2]
    padded = []
    for s in strips:
        h, w, _ = s.shape
        if w < grid_w:
            pad = 255 * np.ones((h, grid_w - w, c), dtype=np.uint8)
            s = np.concatenate([s, pad], axis=1)
        padded.append(s)
    row_spacing = 3
    row_spacer = 255 * np.ones((row_spacing, grid_w, c), dtype=np.uint8)
    parts = []
    for idx, strip in enumerate(padded):
        parts.append(strip)
        if idx < len(padded) - 1:
            parts.append(row_spacer)
    grid = np.concatenate(parts, axis=0)
    Image.fromarray(grid).save(out_path, dpi=(PNG_DPI, PNG_DPI))


def _render_video_mosaic(video_frames, out_path, fps=8, grid_shape=None, grid_cols=None, max_frames=None):
    video_frames = [frames for frames in video_frames if frames]
    if not video_frames:
        return

    if grid_shape is not None:
        rows, cols = grid_shape
    else:
        if grid_cols is not None and grid_cols > 0:
            cols = min(grid_cols, len(video_frames))
        else:
            cols = max(1, math.ceil(math.sqrt(len(video_frames))))
        rows = math.ceil(len(video_frames) / cols)

    base_h, base_w, c = video_frames[0][0].shape
    spacing = 3
    col_spacer = np.full((base_h, spacing, c), 255, dtype=np.uint8)
    row_width = cols * base_w + spacing * max(cols - 1, 0)
    row_spacer = np.full((spacing, row_width, c), 255, dtype=np.uint8)
    blank_frame = np.full((base_h, base_w, c), 255, dtype=np.uint8)
    slots = rows * cols
    padded_sources = video_frames + [None] * (slots - len(video_frames))
    frame_cap = min(len(frames) for frames in video_frames)
    if max_frames is not None:
        frame_cap = min(frame_cap, max_frames)

    mosaic_frames = []
    for t in range(frame_cap):
        row_blocks = []
        for r in range(rows):
            row_parts = []
            for col_idx in range(cols):
                idx = r * cols + col_idx
                src = padded_sources[idx]
                if src is None or t >= len(src):
                    frame = blank_frame
                else:
                    frame = src[t]
                row_parts.append(frame)
                if col_idx < cols - 1:
                    row_parts.append(col_spacer)
            row_blocks.append(np.concatenate(row_parts, axis=1))
            if r < rows - 1:
                row_blocks.append(row_spacer)
        mosaic_frames.append(np.concatenate(row_blocks, axis=0))

    iio.imwrite(out_path, mosaic_frames, fps=fps, codec="libx264", quality=9)


def render_gas_outputs(videos, stats, base_dir,thicknesses):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    row_strips = []
    mosaic_sources = []

    disp_min = stats["global_min"][:, None, None, None]
    disp_max = stats["global_max"][:, None, None, None]
    disp_rng = torch.clamp(disp_max - disp_min, min=1e-12)

    for idx, video in enumerate(videos):
        ims = torch.clamp(video.cpu(), -1.0, 1.0)
        ims = denorm_gas(ims, stats["global_min"], stats["global_max"])
        ims_disp = torch.clamp((ims - disp_min) / disp_rng, 0, 1)
        gray = ims_disp.squeeze(1).numpy()

        dx = np.cumsum(3.5938 * np.power(1.035012, range(gray.shape[2]))) + 0.1
        X, Y = np.meshgrid(dx, np.linspace(0, 208, gray.shape[1]))

        mask = ims[0, 0].numpy() != 0
        thickness = int(mask[:, 0].sum())
        gray_masked = gray[:, :thickness, :]

        clean_dir = base_dir / f"sg_pngs_and_videos_no_labels_{idx}"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_video_frames = []

        for t in range(gray_masked.shape[0]):
            fig, ax = plt.subplots(figsize=(3.5, 2.0))
            ax.axis("off")
            ax.pcolor(
                X[:thickness, :],
                Y[:thickness, :],
                np.flipud(gray_masked[t]),
                shading="auto",
                cmap="jet",
                vmin=0,
                vmax=1,
            )
            ax.set_xlim([0, 1500])
            frame_path = clean_dir / f"frame_{t:03d}.png"
            plt.savefig(frame_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            img = Image.open(frame_path).convert("RGB")
            if PNG_DPI != VIDEO_DPI:
                scale = VIDEO_DPI / PNG_DPI
                new_w = max(1, int(round(img.width * scale)))
                new_h = max(1, int(round(img.height * scale)))
                if new_w != img.width or new_h != img.height:
                    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            clean_video_frames.append(np.array(img))

        video_path = clean_dir / f"sg_video_{idx}.mp4"
        iio.imwrite(video_path, clean_video_frames, fps=8, codec="libx264", quality=9)
        mosaic_sources.append(clean_video_frames)

        labeled_dir = base_dir / f"sg_pngs_and_videos_labels_{idx}"
        labeled_dir.mkdir(parents=True, exist_ok=True)
        labeled_video_frames = []

        sample_cbar_path = labeled_dir / f"sg_colorbar_sample_{idx}.png"
        if not sample_cbar_path.exists():
            fig_cb, ax_cb = plt.subplots(figsize=(4, 0.5))
            norm = colors.Normalize(vmin=0.0, vmax=1.0)
            cb = fig_cb.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), cax=ax_cb, orientation="horizontal")
            cb.set_label("CO₂ Gas Saturation (-)", fontsize=10, fontweight="bold")
            cb.ax.tick_params(labelsize=9)
            fig_cb.savefig(sample_cbar_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig_cb)

        for t in range(gray.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 4))
            imc = ax.pcolor(
                X[:thickness, :],
                Y[:thickness, :],
                np.flipud(gray[t, :thickness, :]),
                shading="auto",
                cmap="jet",
                vmin=0,
                vmax=1,
            )
            ax.set_title(f"Video {idx}, Frame {t}", fontsize=16, fontweight="bold")
            ax.set_xlabel("Radial Distance (m)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Reservoir Thickness (m)", fontsize=14, fontweight="bold")
            ax.set_xlim([0, 1500])
            cbar = fig.colorbar(imc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)
            frame_path = labeled_dir / f"frame_{t:03d}.png"
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(frame_path, dpi=PNG_DPI, bbox_inches="tight")
            plt.close(fig)
            img = Image.open(frame_path).convert("RGB")
            if PNG_DPI != VIDEO_DPI:
                scale = VIDEO_DPI / PNG_DPI
                new_w = max(1, int(round(img.width * scale)))
                new_h = max(1, int(round(img.height * scale)))
                if new_w != img.width or new_h != img.height:
                    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            labeled_video_frames.append(np.array(img))

        labeled_video_path = labeled_dir / f"sg_video_{idx}.mp4"
        iio.imwrite(labeled_video_path, labeled_video_frames, fps=8, codec="libx264", quality=9)

        frame_files = [clean_dir / f"frame_{t:03d}.png" for t in range(gray_masked.shape[0])]
        strip_path = clean_dir / f"sg_strip_no_labels_{idx}.png"
        strip = _render_strip(frame_files, strip_path)
        row_strips.append(strip)

    _render_grid(row_strips, base_dir / "sg_stacked_images.png")
    mosaic_path = base_dir / "sg_mosaic_no_labels.mp4"
    _render_video_mosaic(mosaic_sources, mosaic_path, fps=8, grid_cols=12, max_frames=48)


def render_pressure_outputs(videos, stats, base_dir,thicknesses):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    row_strips = []
    mosaic_sources = []
    global_min = stats["global_min"].cpu().numpy()
    global_max = stats["global_max"].cpu().numpy()

    for idx, video in enumerate(videos):
        ims = torch.clamp(video.cpu(), -1.0, 1.0)
        ims = denorm_pressure(ims, stats["global_min"], stats["global_max"])
        gray = ims.squeeze(1).numpy()

        dx = np.cumsum(3.5938 * np.power(1.035012, range(gray.shape[2]))) + 0.1
        X, Y = np.meshgrid(dx, np.linspace(0, 208, gray.shape[1]))

        mask = ims[0, 0].numpy() > 25 # Adjust this value as you prefer 
        thickness = int(mask[:, 0].sum())
        gray_masked = gray[:, :thickness, :]

        vmins = global_min[:gray_masked.shape[0]]
        vmaxs = global_max[:gray_masked.shape[0]]

        clean_dir = base_dir / f"dP_pngs_and_videos_no_labels_{idx}"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_video_frames = []

        for t in range(gray_masked.shape[0]):
            fig, ax = plt.subplots(figsize=(3.5, 2.0))
            ax.axis("off")
            ax.pcolor(
                X[:thickness, :],
                Y[:thickness, :],
                np.flipud(gray_masked[t]),
                shading="auto",
                cmap="jet",
                vmin=float(vmins[t]),
                vmax=float(vmaxs[t]),
            )
            ax.set_xlim([0, 1500])
            frame_path = clean_dir / f"frame_{t:03d}.png"
            plt.savefig(frame_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            img = Image.open(frame_path).convert("RGB")
            if PNG_DPI != VIDEO_DPI:
                scale = VIDEO_DPI / PNG_DPI
                new_w = max(1, int(round(img.width * scale)))
                new_h = max(1, int(round(img.height * scale)))
                if new_w != img.width or new_h != img.height:
                    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            clean_video_frames.append(np.array(img))

        video_path = clean_dir / f"dP_video_{idx}.mp4"
        iio.imwrite(video_path, clean_video_frames, fps=8, codec="libx264", quality=9)
        mosaic_sources.append(clean_video_frames)

        labeled_dir = base_dir / f"dP_pngs_and_videos_labels_{idx}"
        labeled_dir.mkdir(parents=True, exist_ok=True)
        labeled_video_frames = []

        sample_vmin = float(np.nanmin(vmins))
        sample_vmax = float(np.nanmax(vmaxs))
        sample_cbar_path = labeled_dir / f"dP_colorbar_sample_{idx}.png"
        if not sample_cbar_path.exists():
            fig_cb, ax_cb = plt.subplots(figsize=(4, 0.5))
            norm = colors.Normalize(vmin=sample_vmin, vmax=sample_vmax)
            cb = fig_cb.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), cax=ax_cb, orientation="horizontal")
            cb.set_label("Pressure Build-Up (bar)", fontsize=10, fontweight="bold")
            cb.ax.tick_params(labelsize=9)
            fig_cb.savefig(sample_cbar_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig_cb)

        for t in range(gray.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 4))
            imc = ax.pcolor(
                X[:thickness, :],
                Y[:thickness, :],
                np.flipud(gray[t, :thickness, :]),
                shading="auto",
                cmap="jet",
                vmin=float(vmins[t]),
                vmax=float(vmaxs[t]),
            )
            ax.set_title(f"Video {idx}, Frame {t}", fontsize=16, fontweight="bold")
            ax.set_xlabel("Radial Distance (m)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Reservoir Thickness (m)", fontsize=14, fontweight="bold")
            ax.set_xlim([0, 1500])
            cbar = fig.colorbar(imc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)
            frame_path = labeled_dir / f"frame_{t:03d}.png"
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.savefig(frame_path, dpi=PNG_DPI, bbox_inches="tight")
            plt.close(fig)
            img = Image.open(frame_path).convert("RGB")
            if PNG_DPI != VIDEO_DPI:
                scale = VIDEO_DPI / PNG_DPI
                new_w = max(1, int(round(img.width * scale)))
                new_h = max(1, int(round(img.height * scale)))
                if new_w != img.width or new_h != img.height:
                    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            labeled_video_frames.append(np.array(img))

        labeled_video_path = labeled_dir / f"dP_video_{idx}.mp4"
        iio.imwrite(labeled_video_path, labeled_video_frames, fps=8, codec="libx264", quality=9)

        frame_files = [clean_dir / f"frame_{t:03d}.png" for t in range(gray_masked.shape[0])]
        strip_path = clean_dir / f"dP_strip_no_labels_{idx}.png"
        strip = _render_strip(frame_files, strip_path)
        row_strips.append(strip)

    _render_grid(row_strips, base_dir / "dP_stacked_images.png")
    mosaic_path = base_dir / "dP_mosaic_no_labels.mp4"
    _render_video_mosaic(mosaic_sources, mosaic_path, fps=8, grid_cols=12, max_frames=48)


def sample(args):
    config_path = Path(args.config_path).resolve()
    config_dir = config_path.parent

    with open(config_path, "r") as fh:
        try:
            config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Failed to parse config: {exc}")

    diffusion_cfg = config["diffusion_params"]
    dataset_cfg = config["dataset_params"]
    dit_cfg = config["ditv_params"]
    auto_cfg = config["autoencoder_params"]
    train_cfg = config["train_params"]
    infer_cfg = config["infer_params"]
    autoreg_cfg = dataset_cfg.get("autoregressive")
    rf_cfg = config.get("rf_params", {})

    device = init_device()

    gas_model, pressure_model, gas_channels, pressure_channels = build_autoencoders(
        config,
        device,
        config_dir,
    )

    latent_h, latent_w = compute_latent_hw(
        dataset_cfg,
        auto_cfg["gas"],
        auto_cfg["pressure"],
    )

    patch = dit_cfg["patch_size"]
    frame_height = math.ceil(latent_h / patch) * patch
    frame_width = math.ceil(latent_w / patch) * patch
    pad_h = frame_height - latent_h
    pad_w = frame_width - latent_w
    total_channels = gas_channels + pressure_channels

    num_samples = args.num_samples or train_cfg.get("num_samples", 8)
    num_images = dataset_cfg.get("num_images_train", 0)
    total_chunk_frames = dataset_cfg["num_frames"]
    context_frames = autoreg_cfg["context_frames"] if autoreg_cfg else 0
    predict_frames = autoreg_cfg["predict_frames"] if autoreg_cfg else total_chunk_frames

    num_chunks = args.num_chunks if args.num_chunks is not None else infer_cfg.get("num_chunks", 1)
    if autoreg_cfg is None:
        num_chunks = 1
    elif num_chunks < 1:
        num_chunks = 1

    scheduler = RFlowScheduler(
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

    model = DITVideo(
        frame_height=frame_height,
        frame_width=frame_width,
        im_channels=total_channels,
        num_frames=total_chunk_frames,
        config=dit_cfg,
    ).to(device)
    model.eval()

    ckpt_path = resolve_path(args.checkpoint or train_cfg["ditv_ckpt_name"], config_dir)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Missing joint diffusion checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    gas_stats = torch.load(resolve_path(dataset_cfg["stats_paths"]["gas"], config_dir), map_location="cpu")
    pressure_stats = torch.load(resolve_path(dataset_cfg["stats_paths"]["pressure"], config_dir), map_location="cpu")

    output_dir = Path(resolve_path(infer_cfg["output_dir"], config_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "height": torch.tensor([frame_height], device=device, dtype=torch.float32),
        "width": torch.tensor([frame_width], device=device, dtype=torch.float32),
        "num_frames": torch.tensor([total_chunk_frames], device=device, dtype=torch.float32),
    }

    def model_wrapper(x, timesteps, **_):
        x_time_last = x.permute(0, 2, 1, 3, 4)
        out = model(x_time_last, timesteps, num_images=num_images)
        return out.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        assembled = None
        context_latents = None
        timings = {"chunks": [], "steps": []}

        for chunk_idx in range(num_chunks):
            latents = torch.randn(
                (num_samples, total_chunk_frames, total_channels, frame_height, frame_width),
                device=device,
            )

            if autoreg_cfg and chunk_idx > 0:
                latents[:, :context_frames] = context_latents

            timesteps = scheduler.prepare_inference_timesteps(
                num_samples,
                device=device,
                model_kwargs=model_kwargs,
            )

            _sync()
            t_chunk_start = _now()

            for step_idx, t_b in enumerate(timesteps):
                _sync()
                t_step_start = _now()
                if step_idx < len(timesteps) - 1:
                    dt = (t_b - timesteps[step_idx + 1]) / diffusion_cfg["num_timesteps"]
                else:
                    dt = t_b / diffusion_cfg["num_timesteps"]

                latents = scheduler.inference_step(
                    model_wrapper,
                    latents,
                    t_b,
                    dt,
                    model_kwargs=model_kwargs,
                    channel_last=True,
                )

                if autoreg_cfg and context_latents is not None:
                    latents[:, :context_frames] = context_latents

                _sync()
                timings["steps"].append(_now() - t_step_start)

            _sync()
            timings["chunks"].append(_now() - t_chunk_start)

            if assembled is None:
                assembled = latents
            else:
                assembled = torch.cat([assembled, latents[:, context_frames:]], dim=1)

            if autoreg_cfg:
                context_latents = assembled[:, -context_frames:].detach().clone()

        if timings["chunks"]:
            total_diff = sum(timings["chunks"])
            mean_chunk = total_diff / len(timings["chunks"])
            mean_step = sum(timings["steps"]) / len(timings["steps"])
            steps_per_s = 1.0 / mean_step if mean_step > 0 else float("nan")
            print(
                f"[TIMING] Diffusion only: total={total_diff:.3f}s | per-chunk≈{mean_chunk:.3f}s | "
                f"per-step≈{mean_step:.4f}s ({steps_per_s:.2f} steps/s)",
                flush=True,
            )

        if pad_h or pad_w:
            assembled = assembled[..., :latent_h, :latent_w]

        gas_latents = assembled[:, :, :gas_channels]
        pressure_latents = assembled[:, :, gas_channels:]

        gas_flat = gas_latents.reshape(-1, gas_channels, latent_h, latent_w)
        pressure_flat = pressure_latents.reshape(-1, pressure_channels, latent_h, latent_w)

        gas_decoded = gas_model.decode(gas_flat)
        pressure_decoded = pressure_model.decode(pressure_flat)

        gas_decoded = gas_decoded.reshape(num_samples, -1, gas_decoded.shape[1], gas_decoded.shape[2], gas_decoded.shape[3])
        pressure_decoded = pressure_decoded.reshape(
            num_samples, -1, pressure_decoded.shape[1], pressure_decoded.shape[2], pressure_decoded.shape[3]
        )

        target_h = dataset_cfg["frame_height"]
        target_w = dataset_cfg["frame_width"]
        gas_decoded = gas_decoded[..., :target_h, :target_w]
        pressure_decoded = pressure_decoded[..., :target_h, :target_w]

        gas_decoded = torch.clamp(gas_decoded, -1.0, 1.0)
        pressure_decoded = torch.clamp(pressure_decoded, -1.0, 1.0)

    eps = 1e-4
    gas_masks = (gas_decoded.abs() > eps)[:, :, 0]
    row_activity = gas_masks.any(dim=3)  # collapse width
    row_activity = row_activity.any(dim=1)  # collapse time
    thicknesses = row_activity.sum(dim=1)
    thicknesses = torch.where(
        thicknesses > 0,
        thicknesses,
        torch.full_like(thicknesses, gas_decoded.shape[-2]),
    ).cpu()

    total_frames = gas_decoded.shape[1]
    gas_stats["global_min"] = extend_stats(gas_stats["global_min"], total_frames)
    gas_stats["global_max"] = extend_stats(gas_stats["global_max"], total_frames)
    pressure_stats["global_min"] = extend_stats(pressure_stats["global_min"], total_frames)
    pressure_stats["global_max"] = extend_stats(pressure_stats["global_max"], total_frames)

    torch.save(
        {
            "gas_latent": gas_latents.cpu(),
            "pressure_latent": pressure_latents.cpu(),
            "gas_recon": gas_decoded.cpu(),
            "pressure_recon": pressure_decoded.cpu(),
            "thicknesses": thicknesses,
        },
        output_dir / "joint_samples.pt",
    )

    render_gas_outputs(
        gas_decoded,
        gas_stats,
        output_dir / "gas_saturation_outputs",
        thicknesses.numpy(),
    )
    render_pressure_outputs(
        pressure_decoded,
        pressure_stats,
        output_dir / "pressure_buildup_outputs",
        thicknesses.numpy(),
    )

    print(f"Saved outputs under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint sampler for gas saturation and pressure build-up")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=str(JOINT_DITV_ROOT / "joint.yaml"),
        type=str,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a specific diffusion checkpoint.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of generated videos.")
    parser.add_argument("--num-chunks", type=int, default=None, help="Override autoregressive chunks.")
    sample(parser.parse_args())