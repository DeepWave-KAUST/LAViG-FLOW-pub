###############################################################################
# Joint Workflow Utilities (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Helper functions shared across the joint DiTV scripts: path resolution,
#   device setup, autoencoder loading, scheduler/model construction, etc.
###############################################################################

import os
import sys
from pathlib import Path
from typing import Tuple

import torch

# Extend sys.path so we can reuse the single-task modules without converting them
# into packages. Order matters: gas_saturation is appended first so shared module
# names (e.g. transformer) resolve to that implementation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "gas_saturation"))
sys.path.append(str(ROOT / "pressure_buildup"))

from rf_scheduler import RFlowScheduler 
from transformer import DITVideo 
from vqvae import VQVAE 
from vae import VAE  


# --------------------------------------------------
# Path/device helpers
# --------------------------------------------------

def resolve_path(path: str, base_dir: Path) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return str((base_dir / path).resolve())


def init_device() -> torch.device:
    device = torch.device("cpu")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print("Number of CUDA devices:", num_devices)
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        current = torch.cuda.current_device()
        print("Current CUDA device:", current, "-", torch.cuda.get_device_name(current))
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS backend")

    return device


# --------------------------------------------------
# Autoencoder / model builders
# --------------------------------------------------


def build_autoencoders(config, device: torch.device, config_dir: Path):
    gas_cfg = config["autoencoder_params"]["gas"]
    pressure_cfg = config["autoencoder_params"]["pressure"]

    gas_model = VQVAE(
        im_channels=config["dataset_params"]["frame_channels"],
        autoencoder_model_config={
            key: value for key, value in gas_cfg.items() if key not in {"type", "ckpt_path"}
        },
    ).to(device)
    gas_model.eval()

    pressure_model = VAE(
        im_channels=config["dataset_params"]["frame_channels"],
        autoencoder_model_config={
            key: value for key, value in pressure_cfg.items() if key not in {"type", "ckpt_path"}
        },
    ).to(device)
    pressure_model.eval()

    gas_ckpt = resolve_path(gas_cfg["ckpt_path"], config_dir)
    pressure_ckpt = resolve_path(pressure_cfg["ckpt_path"], config_dir)

    assert os.path.exists(gas_ckpt), f"Missing gas saturation checkpoint: {gas_ckpt}"
    assert os.path.exists(pressure_ckpt), f"Missing pressure build-up checkpoint: {pressure_ckpt}"

    gas_state = torch.load(gas_ckpt, map_location=device)
    if isinstance(gas_state, dict) and "state_dict" in gas_state:
        gas_state = gas_state["state_dict"]
    gas_model.load_state_dict(gas_state)

    pressure_state = torch.load(pressure_ckpt, map_location=device)
    if isinstance(pressure_state, dict) and "state_dict" in pressure_state:
        pressure_state = pressure_state["state_dict"]
    pressure_model.load_state_dict(pressure_state)

    for param in gas_model.parameters():
        param.requires_grad = False
    for param in pressure_model.parameters():
        param.requires_grad = False

    gas_channels = gas_cfg["z_channels"]
    pressure_channels = pressure_cfg["z_channels"]

    return gas_model, pressure_model, gas_channels, pressure_channels


def compute_latent_hw(dataset_cfg, gas_cfg, pressure_cfg) -> Tuple[int, int]:
    down_gas = sum(bool(d) for d in gas_cfg["down_sample"])
    down_pressure = sum(bool(d) for d in pressure_cfg["down_sample"])

    latent_h_gas = dataset_cfg["frame_height"] // (2**down_gas)
    latent_h_pressure = dataset_cfg["frame_height"] // (2**down_pressure)
    latent_w_gas = dataset_cfg["frame_width"] // (2**down_gas)
    latent_w_pressure = dataset_cfg["frame_width"] // (2**down_pressure)

    if latent_h_gas != latent_h_pressure or latent_w_gas != latent_w_pressure:
        raise ValueError(
            f"Latent resolutions mismatch: gas=({latent_h_gas}, {latent_w_gas}), "
            f"pressure=({latent_h_pressure}, {latent_w_pressure})"
        )

    return latent_h_gas, latent_w_gas