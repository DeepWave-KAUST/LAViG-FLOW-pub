"""
###############################################################################
# Benchmarking Python Module (2026)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description: Fourier_MIONet/train_Fourier_MIONet_dp.py
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

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde

from model import branch1, branch2, decoder


FIELD_INDICES = [0, 1, 2, 9, 10]
MIO_INDICES = [4, 5, 6, 7, 8, 9, 10]


def load_config(config_path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fourier-MIONet pressure buildup model")
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

class ProgressCallback(dde.callbacks.Callback):
    def __init__(self, total_epochs, accelerator, desc):
        super().__init__()
        self.total_epochs = total_epochs
        self.accelerator = accelerator
        self.desc = desc
        self.pbar = None

    def on_train_begin(self):
        if self.accelerator.is_main_process:
            self.pbar = tqdm(total=self.total_epochs, desc=self.desc, leave=True)

    def on_epoch_end(self):
        if self.pbar is None:
            return
        train_state = self.model.train_state
        postfix = {}
        if train_state is not None:
            if train_state.loss_train is not None:
                postfix["train"] = float(np.sum(train_state.loss_train))
            if train_state.loss_test is not None:
                postfix["test"] = float(np.sum(train_state.loss_test))
        if postfix:
            self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def on_train_end(self):
        if self.pbar is not None:
            self.pbar.close()

def build_xrt(n_frames):
    t = np.linspace(0, 1, n_frames).astype(np.float32)
    return np.array([[c] for c in t], dtype=np.float32)


def load_split(data_dir, split, n_frames, n_samples=None):
    if split == "train":
        input_path = os.path.join(
            data_dir,
            "dP_data",
            "training_dataset_pressure_buildup",
            "dP_train_inputs.pt",
        )
        output_path = os.path.join(
            data_dir,
            "dP_data",
            "training_dataset_pressure_buildup",
            "dP_train_outputs.pt",
        )
    else:
        input_path = os.path.join(
            data_dir,
            "dP_data",
            "validation_dataset_pressure_buildup",
            "dP_val_inputs.pt",
        )
        output_path = os.path.join(
            data_dir,
            "dP_data",
            "validation_dataset_pressure_buildup",
            "dP_val_outputs.pt",
        )

    inputs = torch.load(input_path)
    outputs = torch.load(output_path)

    inputs = inputs[:, :, :, :n_frames, :]
    outputs = outputs[:, :, :, :n_frames]

    if n_samples is not None:
        n_samples = min(int(n_samples), inputs.shape[0])
        inputs = inputs[:n_samples]
        outputs = outputs[:n_samples]

    return inputs, outputs


def build_inputs(inputs, n_frames):
    x_field = inputs[:, :, :, 0, FIELD_INDICES]
    x_mio = inputs[..., MIO_INDICES].mean(dim=(1, 2, 3))
    xrt = build_xrt(n_frames)
    return x_field.cpu().numpy(), x_mio.cpu().numpy(), xrt


def build_outputs(outputs):
    return outputs.permute(0, 3, 1, 2).reshape(outputs.shape[0], -1).cpu().numpy()


def main():
    args = parse_args()
    default_config = os.path.join(os.path.dirname(__file__), "train_dp.yaml")
    cfg = load_config(args.config or default_config)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    accelerator = Accelerator()
    device = accelerator.device

    n_frames = int(cfg.get("n_frames", 17))
    ntrain = cfg.get("ntrain")
    ntest = cfg.get("ntest")
    data_dir = cfg.get("data_dir", ".")
    save_dir = cfg.get("save_dir", "saved_models")

    train_a, train_u = load_split(data_dir, "train", n_frames, ntrain)
    test_a, test_u = load_split(data_dir, "val", n_frames, ntest)

    x_train = build_inputs(train_a, n_frames)
    y_train = build_outputs(train_u)
    x_test = build_inputs(test_a, n_frames)
    y_test = build_outputs(test_u)

    data = dde.data.QuadrupleCartesianProd(x_train, y_train, x_test, y_test)

    modes1 = int(cfg.get("modes1", 10))
    modes2 = int(cfg.get("modes2", 10))
    width = int(cfg.get("width", 36))
    width2 = int(cfg.get("width2", 128))

    gelu = torch.nn.GELU()

    net = dde.nn.pytorch.mionet.MIONetCartesianProd(
        layer_sizes_branch1=[y_train.shape[1] * 3, branch1(width)],
        layer_sizes_branch2=[7 * width, branch2(width)],
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
        layer_sizes_output_merger=[width, decoder(modes1, modes2, width, width2)],
    ).to(device)
    num_params = net.num_trainable_parameters() if hasattr(net, "num_trainable_parameters") else sum(p.numel() for p in net.parameters())
    accelerator.print(f"num params: {num_params}")

    grid_x = train_a[0, 0, :, 0, -3].cpu().numpy()
    grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2
    grid_dx = torch.as_tensor(grid_dx, device=device).reshape(1, 1, 1, 198)

    pre_mask_train = np.isclose(
        y_train.reshape(y_train.shape[0], n_frames, 96, 200)[:, -1, :, 0], -0.22228621
    )
    pre_mask_test = np.isclose(
        y_test.reshape(y_test.shape[0], n_frames, 96, 200)[:, -1, :, 0], -0.22228621
    )
    pre_mask_train = torch.as_tensor(pre_mask_train, device=device)
    pre_mask_test = torch.as_tensor(pre_mask_test, device=device)

    def loss_fnc(y_true, y_pred, train_indices, istrain):
        size = y_true.shape[0]
        timesize = int(y_true.shape[1] / 200 / 96)
        y_true = y_true.reshape(size, timesize, 96, 200)
        y_pred = y_pred.reshape(size, timesize, 96, 200)
        if istrain:
            mask = pre_mask_train[train_indices]
        else:
            mask = pre_mask_test[train_indices]
        mask = 1 - mask.to(torch.float32)
        mask = mask.reshape(size, 1, 96, 1)
        y_true = y_true * mask
        y_pred = y_pred * mask
        dydx_true_x = (y_true[:, :, :, 2:] - y_true[:, :, :, :-2]) / grid_dx
        dydx_pred_x = (y_pred[:, :, :, 2:] - y_pred[:, :, :, :-2]) / grid_dx
        y_true = y_true.reshape(size, timesize * 96 * 200)
        y_pred = y_pred.reshape(size, timesize * 96 * 200)
        dydx_true_x = dydx_true_x.reshape(size, timesize * 96 * 198)
        dydx_pred_x = dydx_pred_x.reshape(size, timesize * 96 * 198)
        ori_loss = torch.mean(torch.norm(y_true - y_pred, 2, dim=1) / torch.norm(y_true, 2, dim=1))
        der_loss_x = torch.mean(
            torch.norm(dydx_true_x - dydx_pred_x, 2, dim=1) / torch.norm(dydx_true_x, 2, dim=1)
        )
        return [ori_loss, der_loss_x]

    def rsquare_plume_together(y_true, y_pred):
        size = y_true.shape[0]
        y_true = y_true.reshape(size, n_frames, 96, 200)
        y_pred = y_pred.reshape(size, n_frames, 96, 200)
        r2 = 0.0
        for i in range(size):
            z_axis = y_true[i, -1, :, 0]
            mask = np.isclose(z_axis, -0.22228621)
            mask = 1 - mask
            mask = mask.astype(bool)
            y_true_i = y_true[i][:, mask, :]
            y_pred_i = y_pred[i][:, mask, :]
            sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
            sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
            r2 += 1 - sse / sst
        return r2 / size

    def rsquare_plume(y_true, y_pred):
        size = y_true.shape[0]
        y_true = y_true.reshape(size, n_frames, 96, 200)
        y_pred = y_pred.reshape(size, n_frames, 96, 200)
        r2 = 0.0
        for i in range(size):
            z_axis = y_true[i, -1, :, 0]
            mask = np.isclose(z_axis, -0.22228621)
            mask = 1 - mask
            mask = mask.astype(bool)
            for j in range(n_frames):
                y_true_i = y_true[i, j, mask, :]
                y_pred_i = y_pred[i, j, mask, :]
                sse = np.sum(np.square(y_true_i.flatten() - y_pred_i.flatten()))
                sst = np.sum(np.square(y_true_i.flatten() - np.mean(y_true_i.flatten())))
                r2 += 1 - sse / sst
        return r2 / n_frames / size

    os.makedirs(save_dir, exist_ok=True)

    run_name = f"fourier_mionet_dp_{time.strftime('%Y%m%d-%H%M%S')}"
    run_config = {
        "model": "fourier-mionet",
        "target": "dP",
        "n_frames": n_frames,
        "iterations": int(cfg.get("iterations", cfg.get("epochs", 168750))),
        "batch_size": int(cfg.get("batch_size", 4)),
        "learning_rate": float(cfg.get("learning_rate", 1e-3)),
    }

    wandb = None
    wandb_run = None
    if accelerator.is_main_process:
        wandb, wandb_run = init_wandb(cfg, run_name, run_config)

    class WandbCallback(dde.callbacks.Callback):
        def __init__(self, run, accelerator):
            super().__init__()
            self.run = run
            self.accelerator = accelerator

        def on_epoch_end(self):
            if self.run is None or not self.accelerator.is_main_process:
                return
            train_state = self.model.train_state
            if train_state is None:
                return
            data = {}
            if train_state.loss_train is not None:
                data["train/loss"] = float(np.sum(train_state.loss_train))
            if train_state.loss_test is not None:
                data["test/loss"] = float(np.sum(train_state.loss_test))
            if train_state.metrics_test is not None:
                for idx, metric in enumerate(train_state.metrics_test):
                    data[f"test/metric_{idx}"] = float(metric)
            if data:
                self.run.log(data, step=train_state.epoch)

    model = dde.Model(data, net)
    model.compile(
        cfg.get("optimizer", "rmsprop"),
        loss=loss_fnc,
        loss_weights=cfg.get("loss_weights", [1.0, 0.5]),
        lr=float(cfg.get("learning_rate", 1e-3)),
        decay=("step", int(cfg.get("decay_step", 3375)), float(cfg.get("decay_gamma", 0.9))),
        metrics=[rsquare_plume_together, rsquare_plume],
    )

    checkpoint_prefix = os.path.join(save_dir, "Fourier_MIONet_dP")
    checker = dde.callbacks.ModelCheckpoint(
        checkpoint_prefix,
        save_better_only=bool(cfg.get("save_better_only", True)),
        period=int(cfg.get("save_every", 3375)),
        monitor="test loss",
    )

    callbacks = [checker]
    callbacks.append(ProgressCallback(int(cfg.get("iterations", cfg.get("epochs", 1))), accelerator, "train"))
    if wandb_run is not None:
        callbacks.append(WandbCallback(wandb_run, accelerator))

    start = time.time()
    model.train(
        iterations=int(cfg.get("iterations", cfg.get("epochs", 168750))),
        batch_size=int(cfg.get("batch_size", 4)),
        timestep_batch_size=int(cfg.get("timestep_batch_size", 8)),
        training_time_size=n_frames,
        display_every=int(cfg.get("display_every", 500)),
        init_test=True,
        callbacks=callbacks,
    )

    end = time.time()
    accelerator.print(f"running time: {end - start:.2f}s")
    accelerator.print(f"num of parameters: {model.net.num_trainable_parameters()}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
