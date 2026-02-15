"""
###############################################################################
# Benchmarking Python Module (2026)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description: conv-fno/train_ConvFNO_sg.py
#              Baseline benchmarking source file for training, modeling,
#              evaluation, or utilities in the benchmarking workflow.
###############################################################################
"""

import argparse
import os
import time

import numpy as np
import torch

from tqdm.auto import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except ImportError as exc:
    raise RuntimeError("accelerate is required: pip install accelerate") from exc

from conv_fno import ConvFNO3d
from lploss import LpLoss


def load_config(config_path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Train conv-FNO gas saturation model")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    return parser.parse_args()


def init_wandb(cfg, run_name, run_config):
    wb_cfg = cfg.get("wandb", {}) or {}
    if not wb_cfg.get("enable", False):
        return None, None
    os.environ.setdefault("WANDB_API_KEY", "cb961098fd8be776dbcd4831b8ccf64d61cfd1df")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required: pip install wandb") from exc
    project = wb_cfg.get("project", "benchmarking")
    entity = wb_cfg.get("entity")
    tags = wb_cfg.get("tags")
    group = wb_cfg.get("group")
    name = wb_cfg.get("name") or run_name
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        group=group,
        config=run_config,
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME", "allow"),
    )
    return wandb, wandb_run


def main():
    args = parse_args()
    default_config = os.path.join(os.path.dirname(__file__), "train_sg.yaml")
    config_path = args.config or default_config
    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    accelerator = Accelerator()

    n_frames = int(cfg.get("n_frames", 17))
    data_dir = cfg.get("data_dir", "your/data/path")
    save_dir = cfg.get("save_dir", "saved_models")

    print("loading train_a...", flush=True)
    _load_start = time.time()

    train_a = torch.load(f"{data_dir}/co2_data/training_dataset_gas_saturation/sg_train_inputs.pt")
    print(f"loaded train_a in {time.time() - _load_start:.2f}s", flush=True)
    print("loading train_u...", flush=True)
    _load_start = time.time()
    train_u = torch.load(f"{data_dir}/co2_data/training_dataset_gas_saturation/sg_train_outputs.pt")
    print(f"loaded train_u in {time.time() - _load_start:.2f}s", flush=True)
    train_a = train_a[:, :, :, :n_frames, :]
    train_u = train_u[:, :, :, :n_frames]
    print(train_a.shape)
    print(train_u.shape)

    mode1 = int(cfg.get("mode1", 10))
    mode2 = int(cfg.get("mode2", 10))
    mode3 = int(cfg.get("mode3", 10))
    width = int(cfg.get("width", 36))
    device = accelerator.device
    model = ConvFNO3d(mode1, mode2, mode3, width).to(device)
    num_params = model.count_params() if hasattr(model, "count_params") else sum(p.numel() for p in model.parameters())
    print(f"num params: {num_params}")

    _time_grid = np.cumsum(np.power(1.421245, range(n_frames)))
    _time_grid /= np.max(_time_grid)

    grid_x = train_a[0, 0, :, 0, -3]
    grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2
    grid_dx = grid_dx[None, None, :, None].to(device)

    epochs = int(cfg.get("epochs", 100))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    scheduler_step = int(cfg.get("scheduler_step", 2))
    scheduler_gamma = float(cfg.get("scheduler_gamma", 0.9))
    batch_size = int(cfg.get("batch_size", 4))
    save_every = int(cfg.get("save_every", 2))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    myloss = LpLoss(size_average=False)

    os.makedirs(save_dir, exist_ok=True)

    run_name = f"convfno_sg_{time.strftime('%Y%m%d-%H%M%S')}"
    run_config = {
        "model": "conv-fno",
        "target": "sg",
        "num_params": num_params,
        "n_frames": n_frames,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "scheduler_step": scheduler_step,
        "scheduler_gamma": scheduler_gamma,
        "weight_decay": weight_decay,
        "mode1": mode1,
        "mode2": mode2,
        "mode3": mode3,
        "width": width,
    }
    wandb = None
    wandb_run = None
    if accelerator.is_main_process:
        wandb, wandb_run = init_wandb(cfg, run_name, run_config)
    accelerator.wait_for_everyone()
    train_start_time = time.time()
    train_l2 = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        train_l2 = 0
        counter = 0
        progress_bar = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {ep}/{epochs}",
        )
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, n_frames)
            pred = model(x).view(-1, 96, 200, n_frames)

            dy = (y[:, :, 2:, :] - y[:, :, :-2, :]) / grid_dx
            dy_pred = (pred[:, :, 2:, :] - pred[:, :, :-2, :]) / grid_dx

            ori_loss = 0
            der_loss = 0

            # original loss
            for i in range(batch_size):
                ori_loss += myloss(
                    pred[i, ...][mask[i, ...]].reshape(1, -1),
                    y[i, ...][mask[i, ...]].reshape(1, -1),
                )

            # first-derivative loss
            mask_dy = mask[:, :, :198, :]
            for i in range(batch_size):
                der_loss += myloss(
                    dy_pred[i, ...][mask_dy[i, ...]].reshape(1, -1),
                    dy[i, ...][mask_dy[i, ...]].reshape(1, -1),
                )

            loss = ori_loss + 0.5 * der_loss

            accelerator.backward(loss)
            optimizer.step()
            train_l2 += loss.item()

            counter += 1
            if counter % 100 == 0 and accelerator.is_local_main_process:
                progress_bar.set_postfix(train_loss=f"{loss.item() / batch_size:.4f}")

        scheduler.step()

        train_l2_total = accelerator.reduce(torch.tensor(train_l2, device=device), reduction="sum").item()
        accelerator.print(f"Epoch {ep}/{epochs} | train {train_l2_total/train_a.shape[0]:.4f}")

        lr_ = optimizer.param_groups[0]["lr"]
        if wandb_run is not None:
            wandb.log({"train/loss": train_l2_total / train_a.shape[0], "lr": lr_}, step=ep)
        if accelerator.is_main_process and ep % save_every == 0:
            path = f"{save_dir}/SG_ConvFNO_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{train_a.shape[0]}train_{lr_:.2e}lr"
            accelerator.save(accelerator.unwrap_model(model), path)

    total_time_s = time.time() - train_start_time
    accelerator.print(f"total training time (s): {total_time_s:.2f}")
    if wandb_run is not None:
        wandb.log({"train/total_time_s": total_time_s})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
