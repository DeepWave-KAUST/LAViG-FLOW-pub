###############################################################################
# VQ-VAE training CLI (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Stand-alone training script for the CO₂ gas saturation VQ-VAE.
#   Mirrors the original notebook workflow, but externalises configuration
#   via YAML files and adds multi-GPU support through Hugging Face Accelerate.
#
# Source inspiration:
#   StableDiffusion-PyTorch training utility
#   https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_vqvae.py
###############################################################################

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib

# Use a non-interactive backend for headless environments (e.g., SLURM nodes)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from co2_dataset import load_co2_dataset
from lpips import LPIPS
from utils import set_seed
from vqvae import VQVAE


# --------------------------------------------------
# CLI: argument parsing
# --------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQ-VAE for CO₂ gas saturation.")
    parser.add_argument(
        "--config",
        type=str,
        default="co2.yaml",
        help="Path to YAML config file (default: co2.yaml).",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="JSON string with dot.notation overrides (e.g. '{\"training.num_epochs\":80}').",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from (e.g. checkpoints/last.pt).",
    )
    return parser.parse_args()


# --------------------------------------------------
# Config: load YAML and apply overrides
# --------------------------------------------------
def load_config(path: str, overrides: Optional[str]) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        override_dict = json.loads(overrides)
        for key, value in override_dict.items():
            keys = key.split(".")
            cfg = config
            for sub_key in keys[:-1]:
                if sub_key not in cfg or not isinstance(cfg[sub_key], dict):
                    cfg[sub_key] = {}
                cfg = cfg[sub_key]
            cfg[keys[-1]] = value
    return config


# --------------------------------------------------
# Accelerate: construct distributed accelerator
# --------------------------------------------------
def prepare_accelerator(cfg: Dict[str, Any]) -> Accelerator:
    acc_cfg = cfg.get("accelerator", {})
    kwargs: Dict[str, Any] = {}
    if "split_batches" in acc_cfg:
        kwargs["split_batches"] = acc_cfg["split_batches"]
    if "gradient_accumulation_steps" in acc_cfg:
        kwargs["gradient_accumulation_steps"] = acc_cfg["gradient_accumulation_steps"]
    mixed_precision = acc_cfg.get("mixed_precision")
    if mixed_precision:
        kwargs["mixed_precision"] = mixed_precision
    return Accelerator(**kwargs)


# --------------------------------------------------
# Tensors: ensure grayscale tensors mimic RGB
# --------------------------------------------------
def ensure_three_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 4 or tensor.size(1) != 1:
        return tensor
    return tensor.repeat(1, 3, 1, 1)


# --------------------------------------------------
# IO: resolve relative paths against base dir
# --------------------------------------------------
def to_path(base_dir: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


# --------------------------------------------------
# Training utils: toggle gradient flow
# --------------------------------------------------
def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


# --------------------------------------------------
# Visuals: persist loss curve plot
# --------------------------------------------------
def plot_loss_curves(
    generator_losses: Iterable[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(generator_losses) + 1))

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, generator_losses, "-", color="red", linewidth=3, label="Training Loss")
    plt.title("Loss Curve Over Epochs", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    plt.xticks(
        epochs[:: max(1, len(epochs) // 10)] if epochs else [1],
        fontsize=12,
    )
    plt.semilogy()
    plt.legend(fontsize=14, loc="upper right", frameon=True, framealpha=0.8, facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close()


# --------------------------------------------------
# IO: read historical loss values
# --------------------------------------------------
def read_loss_history(summary_path: Path) -> list[float]:
    history_g: list[float] = []
    if not summary_path.exists():
        return history_g
    with open(summary_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("epoch"):
                continue
            parts = stripped.replace("\t", " ").split()
            if len(parts) >= 2:
                try:
                    history_g.append(float(parts[1]))
                except ValueError:
                    continue
    return history_g


# --------------------------------------------------
# Checkpointing: persist training state
# --------------------------------------------------
def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_g_loss: float,
    epochs_since_improvement: int,
    ckpt_dir: Path,
    tag: str = "last",
) -> None:
    if not accelerator.is_main_process:
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_g_loss": best_g_loss,
        "epochs_since_improvement": epochs_since_improvement,
        "model": accelerator.get_state_dict(model),
        "optimizer_g": optimizer_g.state_dict(),
    }
    torch.save(state, ckpt_dir / f"{tag}.pt")
    torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pt")


# --------------------------------------------------
# Training: end-to-end VQ-VAE optimisation loop
# --------------------------------------------------
def train(config: Dict[str, Any], resume_path: Optional[str] = None) -> None:
    accelerator = prepare_accelerator(config)

    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))

    config_path_str = config.get("_config_path")
    config_path = Path(config_path_str) if config_path_str else None
    config_dir = config_path.parent if config_path else Path.cwd()

    output_cfg = config.get("output", {})
    training_cfg = config.get("training", {})

    resume_target = resume_path or training_cfg.get("resume_checkpoint")
    auto_resume = training_cfg.get("auto_resume", False)
    checkpoints_cfg_path = output_cfg.get("checkpoints_dir")
    checkpoints_base_path: Optional[Path] = None
    if checkpoints_cfg_path:
        checkpoints_base_path = Path(checkpoints_cfg_path)
        if not checkpoints_base_path.is_absolute():
            checkpoints_base_path = (config_dir / checkpoints_base_path).resolve()
    if resume_target is None and auto_resume and checkpoints_base_path:
        candidate = checkpoints_base_path / "last.pt"
        if candidate.exists():
            resume_target = str(candidate)

    resume_ckpt: Optional[Path] = None
    if resume_target:
        resume_ckpt = Path(resume_target)
        if not resume_ckpt.is_absolute():
            resume_ckpt = (config_dir / resume_ckpt).resolve()

    start_epoch = 0
    global_step = 0
    best_g_loss = math.inf
    epochs_since_improvement = 0

    data_cfg = config.get("data", {})
    train_dataset = load_co2_dataset(data_cfg["train"])
    loader_cfg = data_cfg.get("loader", {})
    batch_size = loader_cfg.get("batch_size", 32)
    shuffle = loader_cfg.get("shuffle", True)
    drop_last = loader_cfg.get("drop_last", True)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", True)
    persistent_workers = loader_cfg.get("persistent_workers", False) if num_workers > 0 else False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model_cfg = config["model"]
    autoencoder_cfg = deepcopy(model_cfg["autoencoder"])
    im_channels = model_cfg.get("im_channels", 1)
    model = VQVAE(im_channels=im_channels, autoencoder_model_config=autoencoder_cfg)

    lpips_cfg = config.get("lpips", {})
    lpips_enabled = lpips_cfg.get("enable", True)
    lpips_model = LPIPS().eval() if lpips_enabled else None

    optimizer_cfg = config.get("optimizer", {})
    opt_gen_cfg = optimizer_cfg.get("generator", {})
    optimizer_g = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_gen_cfg.get("lr", 2e-4)),
        betas=tuple(opt_gen_cfg.get("betas", [0.5, 0.999])),
    )

    resume_active = False
    if resume_ckpt is not None:
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt}")
        accelerator.print(f"Resuming training from checkpoint: {resume_ckpt}")
        state = torch.load(resume_ckpt, map_location="cpu")
        if "model" not in state:
            raise KeyError("Checkpoint missing 'model' state_dict.")
        model.load_state_dict(state["model"])
        if "optimizer_g" in state:
            optimizer_g.load_state_dict(state["optimizer_g"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_g_loss = float(state.get("best_g_loss", math.inf))
        epochs_since_improvement = int(state.get("epochs_since_improvement", 0))
        resume_active = True

    loss_cfg = config.get("loss", {})
    loss_weights = loss_cfg.get("weights", {})

    num_epochs = int(training_cfg.get("num_epochs", 1))
    early_stopping_patience = training_cfg.get("early_stopping_patience")
    save_loss_curves = bool(training_cfg.get("save_loss_curves", True))
    tensorboard_enabled = bool(training_cfg.get("tensorboard", True))

    if resume_active and start_epoch >= num_epochs:
        accelerator.print(
            f"Checkpoint epoch {start_epoch} >= configured num_epochs {num_epochs}. Nothing to train."
        )
        return

    model_dir = Path(output_cfg.get("model_dir", "trained_models/vqvae"))
    if not model_dir.is_absolute():
        model_dir = (config_dir / model_dir).resolve()
    results_dir = Path(output_cfg.get("results_dir", "results/vqvae"))
    if not results_dir.is_absolute():
        results_dir = (config_dir / results_dir).resolve()
    checkpoints_dir = output_cfg.get("checkpoints_dir")
    if checkpoints_dir:
        checkpoints_dir = checkpoints_base_path if checkpoints_base_path else Path(checkpoints_dir)
        if not isinstance(checkpoints_dir, Path):
            checkpoints_dir = Path(checkpoints_dir)
        if not checkpoints_dir.is_absolute():
            checkpoints_dir = (config_dir / checkpoints_dir).resolve()
    base_dir = model_dir
    summary_path = to_path(base_dir, output_cfg.get("summary_filename"))
    if summary_path is None:
        summary_path = model_dir / "vqvae_training_summary.txt"
    loss_curve_path = (
        to_path(results_dir, output_cfg.get("loss_curve_filename"))
        if save_loss_curves
        else None
    )
    tensorboard_subdir = output_cfg.get("tensorboard_subdir", "tensorboard_logs")

    if accelerator.is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        if checkpoints_dir:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if output_cfg.get("save_config_copy", True):
            with open(model_dir / "used_train_config.yaml", "w") as cfg_out:
                config_export = deepcopy(config)
                config_export.pop("_config_path", None)
                yaml.safe_dump(config_export, cfg_out)

    accelerator.wait_for_everyone()

    recon_criterion = torch.nn.MSELoss()

    model, optimizer_g, train_dataloader = accelerator.prepare(
        model, optimizer_g, train_dataloader
    )
    if lpips_model is not None:
        lpips_model = lpips_model.to(accelerator.device)

    history_g = read_loss_history(summary_path) if resume_active else []

    summary_file = None
    writer = None
    if accelerator.is_main_process:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_mode = "a" if resume_active and summary_path.exists() else "w"
        summary_file = open(summary_path, summary_mode)
        header_line = "Epoch\tTraining Loss\n"
        if summary_mode == "w":
            summary_file.write(header_line)
        elif summary_path.stat().st_size == 0:
            summary_file.write(header_line)
        if tensorboard_enabled:
            log_dir = results_dir / tensorboard_subdir
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))

    if not resume_active:
        best_g_loss = math.inf
        epochs_since_improvement = 0
        global_step = 0

    accelerator.print(
        f"{'Resuming' if resume_active else 'Starting'} training -> "
        f"total epochs: {num_epochs} | starting epoch: {start_epoch + 1} | "
        f"batch size (per process): {batch_size} | world size: {accelerator.num_processes}"
    )

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()

        g_losses: list[float] = []
        progress_bar = tqdm(
            train_dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}/{num_epochs}",
        )

        for batch in progress_bar:
            data = batch[0].to(accelerator.device)
            global_step += 1

            optimizer_g.zero_grad(set_to_none=True)

            with accelerator.autocast():
                output, _, quantize_losses = model(data)
                recon_loss = recon_criterion(output, data)
                g_loss = (
                    recon_loss * float(loss_weights.get("recon", 1.0))
                    + quantize_losses["codebook_loss"] * float(loss_weights.get("codebook", 0.0))
                    + quantize_losses["commitment_loss"] * float(loss_weights.get("commitment", 0.0))
                )

                if lpips_model is not None and float(loss_weights.get("lpips", 0.0)) > 0:
                    lpips_weight = float(loss_weights["lpips"])
                    lpips_in0 = ensure_three_channel(output)
                    lpips_in1 = ensure_three_channel(data)
                    lpips_loss = lpips_model(lpips_in0, lpips_in1).mean()
                    g_loss = g_loss + lpips_weight * lpips_loss

            accelerator.backward(g_loss)
            optimizer_g.step()
            g_losses.append(g_loss.detach().float().item())

        accelerator.wait_for_everyone()

        g_epoch_loss_tensor = torch.tensor(
            [sum(g_losses), len(g_losses)], device=accelerator.device
        )
        accelerator.reduce(g_epoch_loss_tensor, reduction="sum")

        g_epoch_loss = (g_epoch_loss_tensor[0] / max(g_epoch_loss_tensor[1], 1)).item()

        history_g.append(g_epoch_loss)

        accelerator.print(
            f"[Epoch {epoch:03d}/{num_epochs}] Training Loss: {g_epoch_loss:.4f}"
        )

        if accelerator.is_main_process and summary_file:
            summary_file.write(f"{epoch}\t{g_epoch_loss:.4f}\n")
            summary_file.flush()
            if writer:
                writer.add_scalar("VQ-VAE/Training_Loss", g_epoch_loss, epoch)

        improved = g_epoch_loss < best_g_loss
        best_g_loss = min(best_g_loss, g_epoch_loss)
        epochs_since_improvement = 0 if improved else epochs_since_improvement + 1

        if checkpoints_dir:
            save_checkpoint(
                accelerator,
                model,
                optimizer_g,
                epoch,
                global_step,
                best_g_loss,
                epochs_since_improvement,
                checkpoints_dir,
            )

        if early_stopping_patience is not None and epochs_since_improvement >= int(
            early_stopping_patience
        ):
            accelerator.print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        gen_state = accelerator.unwrap_model(model).state_dict()
        torch.save(
            {
                "autoencoder_config": autoencoder_cfg,
                "state_dict": gen_state,
            },
            model_dir / "vqvae_model.pth",
        )

        final_summary_path = model_dir / "vqvae_training_summary_full.txt"
        if summary_path != final_summary_path:
            final_summary_path.write_text(summary_path.read_text())

        if save_loss_curves and loss_curve_path is not None:
            plot_loss_curves(history_g, loss_curve_path)

        if writer:
            writer.close()
        if summary_file:
            summary_file.close()

    accelerator.print("Training completed successfully.")


# --------------------------------------------------
# CLI: script entry point
# --------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    cfg["_config_path"] = str(Path(args.config).resolve())
    train(cfg, args.resume)


if __name__ == "__main__":
    main()