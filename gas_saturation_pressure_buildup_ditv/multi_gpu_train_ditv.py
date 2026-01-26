###############################################################################
# Multi-GPU DiTV Training (CO₂ Gas + Pressure Build-Up) (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Distributed training loop for the joint DiTV model, inspired by the open-source
#   Video Diffusion reference implementation (https://github.com/explainingai-code/VideoGeneration-PyTorch)
#   and adapted to the CO₂ gas saturation + pressure build-up workflow. Handles dataset
#   preparation, latent encoding, scheduler integration, checkpointing, and TensorBoard
#   logging across multiple GPUs via HuggingFace Accelerate.
###############################################################################

import warnings

warnings.filterwarnings("ignore")

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make parent modules importable (mirrors the Video Diffusion repo layout)
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from joint_dataset import JointTargetsDataset
from utils import (
    DITVideo,
    RFlowScheduler,
    build_autoencoders,
    compute_latent_hw,
    resolve_path,
)
from paths import JOINT_DITV_ROOT


# --------------------------------------------------
# Checkpoint utilities
# --------------------------------------------------


def save_checkpoint(model, optimizer, epoch, global_step, path, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, path, accelerator, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return start_epoch, global_step


def _encode_gas(latents, model):
    b, f, c, h, w = latents.shape
    latents = latents.reshape(-1, c, h, w)
    latents, _ = model.encode(latents)
    latents = rearrange(latents, "(b f) c h w -> b f c h w", b=b, f=f)
    return latents


def _encode_pressure(latents, model):
    b, f, c, h, w = latents.shape
    latents = latents.reshape(-1, c, h, w)
    _, enc = model.encode(latents)
    mu, _ = torch.chunk(enc, 2, dim=1)
    latents = rearrange(mu, "(b f) c h w -> b f c h w", b=b, f=f)
    return latents


def prepare_joint_inputs(
    batch,
    dataset,
    device,
    gas_model,
    pressure_model,
    patch,
    detach_context: bool = False,
):
    """Move batch to device, encode modalities, and pad spatial dimensions."""
    autoreg = getattr(dataset, "autoregressive_cfg", None) is not None

    with torch.no_grad():
        if autoreg:
            context = batch["context"]
            target = batch["target"]

            gas_context = context["gas"].float().to(device, non_blocking=True)
            gas_target = target["gas"].float().to(device, non_blocking=True)
            pressure_context = context["pressure"].float().to(device, non_blocking=True)
            pressure_target = target["pressure"].float().to(device, non_blocking=True)

            gas_latents_context = _encode_gas(gas_context, gas_model)
            gas_latents_target = _encode_gas(gas_target, gas_model)
            pressure_latents_context = _encode_pressure(pressure_context, pressure_model)
            pressure_latents_target = _encode_pressure(pressure_target, pressure_model)

            latents_context = torch.cat([gas_latents_context, pressure_latents_context], dim=2)
            latents_target = torch.cat([gas_latents_target, pressure_latents_target], dim=2)

            latents = torch.cat([latents_context, latents_target], dim=1)
            context_frames = gas_context.shape[1]
            if detach_context:
                latents[:, :context_frames] = latents[:, :context_frames].detach()
        else:
            gas = batch["gas"].float().to(device, non_blocking=True)
            pressure = batch["pressure"].float().to(device, non_blocking=True)

            gas_latents = _encode_gas(gas, gas_model)
            pressure_latents = _encode_pressure(pressure, pressure_model)

            latents = torch.cat([gas_latents, pressure_latents], dim=2)
            context_frames = 0

    pad_h = (-latents.shape[-2]) % patch
    pad_w = (-latents.shape[-1]) % patch
    if pad_h or pad_w:
        latents = torch.nn.functional.pad(latents, (0, pad_w, 0, pad_h))

    return latents, context_frames


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------

@torch.no_grad()
def evaluate_epoch(
    accelerator,
    model,
    scheduler,
    dataloader,
    dataset,
    gas_model,
    pressure_model,
    patch,
    diffusion_config,
    autoregressive_params=None,
    num_images=0,
):
    if dataloader is None:
        return None

    model.eval()
    losses = []

    mask_context_loss = False
    noise_context = True
    if autoregressive_params and getattr(dataset, "autoregressive_cfg", None):
        mask_context_loss = bool(autoregressive_params.get("mask_context_loss", False))
        noise_context = bool(autoregressive_params.get("noise_context", True))
    detach_context = (
        bool(autoregressive_params.get("detach_context_from_grad", False))
        if autoregressive_params and getattr(dataset, "autoregressive_cfg", None)
        else False
    )

    for batch in dataloader:
        latents, context_frames = prepare_joint_inputs(
            batch,
            dataset,
            accelerator.device,
            gas_model,
            pressure_model,
            patch,
            detach_context=detach_context,
        )
        latents_cf = latents.permute(0, 2, 1, 3, 4)
        b, _, f, _, _ = latents_cf.shape

        noise_mask = None
        loss_mask = None
        if context_frames:
            if not noise_context:
                noise_mask = torch.ones((b, f), device=accelerator.device)
                noise_mask[:, :context_frames] = 0
            if mask_context_loss:
                loss_mask = torch.ones((b, f), device=accelerator.device)
                loss_mask[:, :context_frames] = 0

        model_kwargs = {
            "height": torch.tensor([latents.shape[-2]], device=accelerator.device, dtype=torch.float32),
            "width": torch.tensor([latents.shape[-1]], device=accelerator.device, dtype=torch.float32),
            "num_frames": torch.tensor([latents.shape[1]], device=accelerator.device, dtype=torch.float32),
        }

        def model_wrapper(x, timesteps, **_):
            x_time_last = x.permute(0, 2, 1, 3, 4)
            out = model(x_time_last, timesteps, num_images=num_images)
            return out.permute(0, 2, 1, 3, 4)

        loss_vals, _ = scheduler.training_losses(
            model_wrapper,
            latents_cf,
            model_kwargs=model_kwargs,
            mask=noise_mask,
            loss_mask=loss_mask,
        )
        loss = loss_vals.mean()

        gathered = accelerator.gather_for_metrics(loss.detach())
        losses.extend(gathered.tolist())

    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------
# Training orchestration
# --------------------------------------------------

def train(args):
    set_seed(42)

    config_path = Path(args.config_path).resolve()
    config_dir = config_path.parent

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    diffusion_config = config["diffusion_params"]
    rf_config = config.get("rf_params", {})
    dataset_config = config["dataset_params"]
    ditv_config = config["ditv_params"]
    autoencoder_config = config["autoencoder_params"]
    train_config = config["train_params"]
    autoregressive_cfg = dataset_config.get("autoregressive")
    autoregressive_params = config.get("autoregressive_params", {}) if autoregressive_cfg else {}

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config["ditv_acc_steps"],
        mixed_precision="fp16",
    )

    if accelerator.is_main_process:
        print(config)
        print("===================================")
        print("Distributed Training Environment")
        print("World size:", accelerator.num_processes)
        print("Device:", accelerator.device)
        print("===================================")

    accelerator.print(f"[Rank {accelerator.process_index}] using device={accelerator.device}")

    scheduler = RFlowScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        num_sampling_steps=rf_config.get("num_sampling_steps", 30),
        sample_method=rf_config.get("sample_method", "uniform"),
        use_discrete_timesteps=rf_config.get("use_discrete_timesteps", False),
        use_timestep_transform=rf_config.get("use_timestep_transform", False),
        transform_scale=rf_config.get("transform_scale", 1.0),
        pa_vdm=rf_config.get("pa_vdm", False),
        noise_pattern=rf_config.get("noise_pattern", "linear"),
        linear_variance_scale=rf_config.get("linear_variance_scale", 0.1),
        linear_shift_scale=rf_config.get("linear_shift_scale", 0.3),
        latent_chunk_size=rf_config.get("latent_chunk_size", 1),
        keep_x0=rf_config.get("keep_x0", False),
        variable_length=rf_config.get("variable_length", False),
    )

    file_map = dataset_config["file_map"]

    def build_dataset(split_key, drop_incomplete=True):
        if split_key not in file_map or file_map[split_key] is None:
            return None
        paths = file_map[split_key]
        gas_path = resolve_path(paths["gas"], config_dir)
        pressure_path = resolve_path(paths["pressure"], config_dir)
        ar_cfg = None
        if autoregressive_cfg:
            ar_cfg = dict(autoregressive_cfg)
            ar_cfg["drop_incomplete"] = drop_incomplete
        return JointTargetsDataset(
            gas_path=gas_path,
            pressure_path=pressure_path,
            time_slice=dataset_config.get("time_slice"),
            autoregressive=ar_cfg,
        )

    train_dataset = build_dataset("train", drop_incomplete=True)
    val_dataset = build_dataset("val", drop_incomplete=False)
    test_dataset = build_dataset("test", drop_incomplete=False)

    assert train_dataset is not None, "Training dataset is required."

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["ditv_batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    def build_loader(dataset):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=train_config["ditv_batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    val_loader = build_loader(val_dataset)
    test_loader = build_loader(test_dataset)

    gas_model, pressure_model, gas_channels, pressure_channels = build_autoencoders(
        config,
        accelerator.device,
        config_dir,
    )

    latent_h, latent_w = compute_latent_hw(
        dataset_config,
        autoencoder_config["gas"],
        autoencoder_config["pressure"],
    )

    patch = ditv_config["patch_size"]
    frame_height = math.ceil(latent_h / patch) * patch
    frame_width = math.ceil(latent_w / patch) * patch
    num_frames = getattr(train_dataset, "num_frames", dataset_config["num_frames"])
    im_channels = gas_channels + pressure_channels

    model = DITVideo(
        frame_height=frame_height,
        frame_width=frame_width,
        im_channels=im_channels,
        num_frames=num_frames,
        config=ditv_config,
    )

    total_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        run_type = "autoregressive" if autoregressive_cfg else "standard"
        print(f"Total DiTVideo parameters ({run_type} run): {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=float(train_config["ditv_lr"]), weight_decay=0.0)

    prepared = [model, optimizer, train_loader]
    if val_loader is not None:
        prepared.append(val_loader)
    if test_loader is not None:
        prepared.append(test_loader)
    prepared = accelerator.prepare(*prepared)

    model = prepared[0]
    optimizer = prepared[1]
    train_loader = prepared[2]
    idx = 3
    if val_loader is not None:
        val_loader = prepared[idx]
        idx += 1
    if test_loader is not None:
        test_loader = prepared[idx]

    ckpt_path = resolve_path(train_config["ditv_ckpt_name"], config_dir)
    full_ckpt_path = resolve_path(train_config["ditv_ckpt_complete_name"], config_dir)
    base_ckpt_path = None
    if train_config.get("base_ditv_ckpt"):
        base_ckpt_path = resolve_path(train_config["base_ditv_ckpt"], config_dir)

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        log_dir = resolve_path(train_config["log_dir"], config_dir)
        loss_dir = resolve_path(train_config["loss_dir"], config_dir)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        train_csv = Path(loss_dir) / "loss_train.csv"
        val_csv = Path(loss_dir) / "loss_val.csv"
        test_csv = Path(loss_dir) / "loss_test.csv"
        if not train_csv.exists():
            train_csv.write_text("epoch,loss\n")
        if val_loader is not None and not val_csv.exists():
            val_csv.write_text("epoch,loss\n")
        if test_loader is not None and not test_csv.exists():
            test_csv.write_text("epoch,loss\n")
    else:
        writer = None
        loss_dir = train_csv = val_csv = test_csv = None

    accelerator.wait_for_everyone()

    start_epoch = 0
    global_step = 0

    if os.path.exists(full_ckpt_path):
        accelerator.print(f"Resuming training state from {full_ckpt_path}")
        start_epoch, global_step = load_checkpoint(
            model,
            optimizer,
            full_ckpt_path,
            accelerator,
            map_location=accelerator.device,
        )
    else:
        target_path = None
        tag = None
        base_missing = False

        if os.path.exists(ckpt_path):
            target_path = ckpt_path
            tag = "Loaded diffusion weights"
        elif base_ckpt_path:
            if os.path.exists(base_ckpt_path):
                target_path = base_ckpt_path
                tag = "Initializing diffusion weights from base checkpoint"
            else:
                base_missing = True

        if target_path is not None:
            if accelerator.is_main_process:
                print(f"{tag}: {target_path}")

            state = torch.load(target_path, map_location="cpu")
            if isinstance(state, dict):
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]

            if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}

            accelerator.unwrap_model(model).load_state_dict(state, strict=True)
        else:
            if base_missing:
                accelerator.print(f"Base checkpoint not found: {base_ckpt_path}")
            accelerator.print(f"No diffusion checkpoint found at: {ckpt_path}")

    num_epochs = train_config["ditv_epochs"]
    mask_context_loss = bool(autoregressive_params.get("mask_context_loss", True)) if autoregressive_cfg else False
    noise_context = bool(autoregressive_params.get("noise_context", True)) if autoregressive_cfg else True
    detach_context = bool(autoregressive_params.get("detach_context_from_grad", False)) if autoregressive_cfg else False

    for epoch_idx in range(start_epoch, num_epochs):
        if accelerator.is_main_process:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch_idx+1}/{num_epochs}", dynamic_ncols=True)
        else:
            pbar = None

        running_losses = []

        for batch in train_loader:
            latents, context_frames = prepare_joint_inputs(
                batch,
                train_dataset,
                accelerator.device,
                gas_model,
                pressure_model,
                patch,
                detach_context=detach_context,
            )
            latents_cf = latents.permute(0, 2, 1, 3, 4)
            b, _, f, _, _ = latents_cf.shape

            noise_mask = None
            loss_mask = None
            if context_frames:
                if not noise_context:
                    noise_mask = torch.ones((b, f), device=accelerator.device)
                    noise_mask[:, :context_frames] = 0
                if mask_context_loss:
                    loss_mask = torch.ones((b, f), device=accelerator.device)
                    loss_mask[:, :context_frames] = 0

            model_kwargs = {
                "height": torch.tensor([latents.shape[-2]], device=accelerator.device, dtype=torch.float32),
                "width": torch.tensor([latents.shape[-1]], device=accelerator.device, dtype=torch.float32),
                "num_frames": torch.tensor([latents.shape[1]], device=accelerator.device, dtype=torch.float32),
            }

            with accelerator.accumulate(model):
                def model_wrapper(x, timesteps, **_):
                    x_time_last = x.permute(0, 2, 1, 3, 4)
                    out = model(x_time_last, timesteps, num_images=dataset_config.get("num_images_train", 0))
                    return out.permute(0, 2, 1, 3, 4)

                loss_vals, _ = scheduler.training_losses(
                    model_wrapper,
                    latents_cf,
                    model_kwargs=model_kwargs,
                    mask=noise_mask,
                    loss_mask=loss_mask,
                )
                loss = loss_vals.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients and train_config.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1

            loss_value = accelerator.reduce(loss.detach(), reduction="mean")
            running_losses.append(loss_value.item())

            if pbar is not None and accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_value.item():.4f}")

            if accelerator.is_main_process and writer is not None:
                writer.add_scalar("train/loss_step", loss_value.item(), global_step)

        if pbar is not None:
            pbar.close()

        accelerator.wait_for_everyone()

        val_mean = evaluate_epoch(
            accelerator,
            model,
            scheduler,
            val_loader,
            val_dataset,
            gas_model,
            pressure_model,
            patch,
            diffusion_config,
            autoregressive_params if val_dataset else None,
            num_images=dataset_config.get("num_images_val", 0),
        ) if val_loader is not None else None

        test_mean = evaluate_epoch(
            accelerator,
            model,
            scheduler,
            test_loader,
            test_dataset,
            gas_model,
            pressure_model,
            patch,
            diffusion_config,
            autoregressive_params if test_dataset else None,
            num_images=dataset_config.get("num_images_test", 0),
        ) if (test_loader is not None and train_config.get("eval_test_each_epoch", False)) else None

        if accelerator.is_main_process:
            train_mean = float(np.mean(running_losses)) if running_losses else float("nan")

            if writer is not None:
                scalars = {"train": train_mean}
                if val_mean is not None:
                    scalars["val"] = val_mean
                if test_mean is not None:
                    scalars["test"] = test_mean
                writer.add_scalars("loss/epoch", scalars, epoch_idx + 1)

            msg = f"Epoch {epoch_idx+1}/{num_epochs} | train {train_mean:.4f}"
            if val_mean is not None:
                msg += f" | val {val_mean:.4f}"
            if test_mean is not None:
                msg += f" | test {test_mean:.4f}"
            print(msg)

            with open(train_csv, "a") as f:
                f.write(f"{epoch_idx + 1},{train_mean}\n")
            if val_mean is not None and val_csv is not None:
                with open(val_csv, "a") as f:
                    f.write(f"{epoch_idx + 1},{val_mean}\n")
            if test_mean is not None and test_csv is not None:
                with open(test_csv, "a") as f:
                    f.write(f"{epoch_idx + 1},{test_mean}\n")

            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            save_checkpoint(model, optimizer, epoch_idx + 1, global_step, full_ckpt_path, accelerator)

        accelerator.wait_for_everyone()

    if writer is not None:
        writer.close()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_test = None
        if test_loader is not None and not train_config.get("eval_test_each_epoch", False):
            final_test = evaluate_epoch(
                accelerator,
                model,
                scheduler,
                test_loader,
                test_dataset,
                gas_model,
                pressure_model,
                patch,
                diffusion_config,
                autoregressive_params if test_dataset else None,
                num_images=dataset_config.get("num_images_test", 0),
            )
            writer = SummaryWriter(log_dir=resolve_path(train_config["log_dir"], config_dir))
            writer.add_scalars("loss/epoch", {"test": final_test}, train_config["ditv_epochs"])
            with open(os.path.join(resolve_path(train_config["loss_dir"], config_dir), "loss_test.csv"), "a") as f:
                f.write(f"{train_config['ditv_epochs']},{final_test}\n")
            writer.close()
            print(f"Final Test Loss: {final_test:.4f}")

        print("Done Training ...")

        loss_dir_path = Path(resolve_path(train_config["loss_dir"], config_dir))
        series = []

        def read_series(path, label):
            if not path.exists():
                return
            xs, ys = [], []
            with path.open() as fh:
                next(fh, None)  # skip header
                for line in fh:
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    xs.append(int(parts[0]))
                    ys.append(float(parts[1]))
            if xs:
                series.append((label, xs, ys))

        read_series(loss_dir_path / "loss_train.csv", "train")
        read_series(loss_dir_path / "loss_val.csv", "val")
        read_series(loss_dir_path / "loss_test.csv", "test")

        if series:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            colors = {"train": "blue", "val": "green", "test": "red"}
            plt.figure(figsize=(7, 4))
            for label, xs, ys in series:
                plt.semilogy(xs, ys, linestyle="-", label=label, color=colors.get(label, "black"))
            plt.xlabel("Epoch", fontweight="bold")
            plt.ylabel("Loss", fontweight="bold")
            plt.title("Training - Validation - Test Curves", fontweight="bold")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            out_png = loss_dir_path / "loss_curves.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=600, bbox_inches="tight")
            plt.close()
            print(f"Saved loss curves to: {out_png}")


# --------------------------------------------------
# CLI entry
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU joint diffusion training with autoregressive support")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=str(JOINT_DITV_ROOT / "joint.yaml"),
        type=str,
    )
    train(parser.parse_args())