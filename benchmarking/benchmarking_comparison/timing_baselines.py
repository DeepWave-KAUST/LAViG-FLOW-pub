"""
###############################################################################
# Benchmarking Python Module (2026)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description: benchmark_comparison/timing_baselines.py
#              Baseline benchmarking source file for training, modeling,
#              evaluation, or utilities in the benchmarking workflow.
###############################################################################
"""

import argparse
import importlib.util
import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

from eval_baselines import (
    DEFAULT_EVAL_ROOT,
    MODEL_DIRS,
    _as_path,
    load_yaml,
    load_inputs_targets,
    load_torch_model,
    make_mionet_vanilla_predictor,
    make_mionet_fourier_predictor,
)


def _set_cpu_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


def _time_fn(fn, runs: int, warmup: int, device: torch.device) -> float:
    # warmup
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times))


def _load_lavig_eval_module(lavig_root: Path):
    # Add both the DiTV folder and LAViG-FLOW root (for paths.py).
    sys.path.insert(0, str(lavig_root))
    if lavig_root.parent.exists():
        sys.path.insert(0, str(lavig_root.parent))
    module_path = lavig_root / "evaluate_autoregressive.py"
    spec = importlib.util.spec_from_file_location("lavig_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load LAViG eval module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_lavig_model(cfg: dict, config_dir: Path, device: torch.device, lavig_eval):
    dataset_cfg = cfg["dataset_params"]
    auto_cfg_gas = cfg["autoencoder_params"]["gas"]
    auto_cfg_pressure = cfg["autoencoder_params"]["pressure"]
    dit_cfg = cfg["ditv_params"]
    train_cfg = cfg["train_params"]

    latent_h, latent_w = lavig_eval.compute_latent_hw(dataset_cfg, auto_cfg_gas, auto_cfg_pressure)
    patch = dit_cfg["patch_size"]
    frame_height = math.ceil(latent_h / patch) * patch
    frame_width = math.ceil(latent_w / patch) * patch
    total_channels = auto_cfg_gas["z_channels"] + auto_cfg_pressure["z_channels"]

    model = lavig_eval.DITVideo(
        frame_height=frame_height,
        frame_width=frame_width,
        im_channels=total_channels,
        num_frames=dataset_cfg["num_frames"],
        config=dit_cfg,
    ).to(device)

    ckpt_path = lavig_eval.resolve_path(train_cfg["ditv_ckpt_name"], config_dir)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("patch_embed_layer.patch_embed.0.") for k in state):
        state = {
            k.replace("patch_embed_layer.patch_embed.0.", "patch_embed_layer.patch_embed."): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, frame_height, frame_width, latent_h, latent_w


def _time_lavig_flow(
    lavig_cfg_path: Path,
    stages: List[int],
    device: torch.device,
    runs: int,
    warmup: int,
    batch_size: int,
    include_decode: bool,
    include_encode: bool,
) -> Dict[int, Dict[str, float]]:
    cfg = load_yaml(str(lavig_cfg_path))
    config_dir = lavig_cfg_path.parent

    lavig_eval = _load_lavig_eval_module(config_dir)

    dataset_cfg = cfg.get("dataset_params", {})
    autoreg_cfg = dataset_cfg.get("autoregressive", {}) or {}
    context_frames = int(autoreg_cfg.get("context_frames", 0))
    predict_frames = int(autoreg_cfg.get("predict_frames", dataset_cfg.get("num_frames", 0)))

    val_paths = dataset_cfg.get("file_map", {}).get("val", {})
    if not val_paths:
        raise ValueError("LAViG config missing dataset_params.file_map.val entries.")

    dataset = lavig_eval.JointEvaluationDataset(
        gas_path=lavig_eval.resolve_path(val_paths["gas"], config_dir),
        pressure_path=lavig_eval.resolve_path(val_paths["pressure"], config_dir),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    gas_seq = batch["gas"].to(device)
    pressure_seq = batch["pressure"].to(device)

    gas_model, pressure_model, _, _ = lavig_eval.build_autoencoders(cfg, device, config_dir)
    scheduler = lavig_eval.build_scheduler(cfg["diffusion_params"], cfg.get("rf_params", {}))
    model, frame_height, frame_width, latent_h, latent_w = _build_lavig_model(cfg, config_dir, device, lavig_eval)

    if context_frames > 0:
        context_gas = gas_seq[:, :context_frames]
        context_pressure = pressure_seq[:, :context_frames]
    else:
        context_gas = None
        context_pressure = None

    def _encode_context():
        if context_frames <= 0:
            return None
        context_gas_latents = lavig_eval._encode_gas(context_gas, gas_model)
        context_pressure_latents = lavig_eval._encode_pressure(context_pressure, pressure_model)
        return torch.cat([context_gas_latents, context_pressure_latents], dim=2)

    context_latents = None if include_encode else _encode_context()

    results: Dict[int, Dict[str, float]] = {}
    for stage in stages:
        num_chunks = int(stage)
        total_frames = context_frames + predict_frames * num_chunks

        def lavig_fn():
            with torch.no_grad():
                ctx = _encode_context() if include_encode else context_latents
                generated_latents = lavig_eval.generate_joint_predictions(
                    model=model,
                    scheduler=scheduler,
                    config=cfg,
                    frame_height=frame_height,
                    frame_width=frame_width,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    batch_size=batch_size,
                    num_chunks=num_chunks,
                    device=device,
                    context_latents=ctx,
                )
                if include_decode:
                    lavig_eval.decode_joint_latents(
                        generated_latents,
                        gas_model,
                        pressure_model,
                        cfg,
                        latent_h,
                        latent_w,
                    )

        total_time = _time_fn(lavig_fn, runs, warmup, device)
        results[stage] = {
            "predicted_frames": int(predict_frames * num_chunks),
            "total_frames": int(total_frames),
            "total_seconds_per_video": total_time / batch_size,
            "includes_encode": bool(include_encode),
            "includes_decode": bool(include_decode),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure baseline inference time per video")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-json", default=None, help="Path to write JSON timing results")
    parser.add_argument("--cpu-threads", type=int, default=None, help="Force CPU thread count (e.g. 1 or 4)")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=10, help="Timing iterations")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})

    requested = (args.device or data_cfg.get("device", "cpu")).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[timing] CUDA requested but not available; falling back to CPU.", flush=True)
        requested = "cpu"
    device = torch.device(requested)
    if args.cpu_threads is not None and device.type == "cpu":
        _set_cpu_threads(args.cpu_threads)

    stages = [int(s) for s in data_cfg.get("stages", [1, 2, 3, 4])]
    context_frames = int(data_cfg.get("context_frames", 15))
    predict_frames = int(data_cfg.get("predict_frames", 2))
    batch_size = int(data_cfg.get("batch_size", 1))

    # Load a single sample for timing
    gas_inputs, gas_targets = load_inputs_targets(cfg, "gas")
    dp_inputs, dp_targets = load_inputs_targets(cfg, "pressure")

    gas_inputs = gas_inputs[:batch_size]
    dp_inputs = dp_inputs[:batch_size]

    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for model_name, spec in cfg.get("models", {}).items():
        gas_ckpt = _as_path(spec.get("gas"))
        pressure_ckpt = _as_path(spec.get("pressure"))
        if gas_ckpt is None or pressure_ckpt is None:
            print(f"Skipping {model_name}: missing gas or pressure checkpoint")
            continue
        model_dir = MODEL_DIRS.get(model_name)
        if model_dir is None:
            print(f"Skipping {model_name}: unknown model dir")
            continue

        print(f"Timing {model_name} ...")
        results[model_name] = {}

        if model_name in {"conv_fno", "fno", "ufno"}:
            gas_model = load_torch_model(gas_ckpt, device)
            dp_model = load_torch_model(pressure_ckpt, device)

            for stage in stages:
                total_frames = context_frames + predict_frames * stage
                x_gas = gas_inputs[..., :total_frames, :].to(device)
                x_dp = dp_inputs[..., :total_frames, :].to(device)

                def gas_fn():
                    with torch.no_grad():
                        gas_model(x_gas)

                def dp_fn():
                    with torch.no_grad():
                        dp_model(x_dp)

                gas_time = _time_fn(gas_fn, args.runs, args.warmup, device)
                dp_time = _time_fn(dp_fn, args.runs, args.warmup, device)
                results[model_name][stage] = {
                    "predicted_frames": int(predict_frames * stage),
                    "total_frames": int(total_frames),
                    "gas_seconds_per_video": gas_time / batch_size,
                    "pressure_seconds_per_video": dp_time / batch_size,
                    "total_seconds_per_video": (gas_time + dp_time) / batch_size,
                }

        elif model_name == "vanilla_mionet":
            gas_predictor = make_mionet_vanilla_predictor(model_dir, gas_ckpt, device)
            dp_predictor = make_mionet_vanilla_predictor(model_dir, pressure_ckpt, device)
            for stage in stages:
                total_frames = context_frames + predict_frames * stage
                x_gas = gas_inputs[..., :total_frames, :]
                y_gas = gas_targets[:, :total_frames]
                x_dp = dp_inputs[..., :total_frames, :]
                y_dp = dp_targets[:, :total_frames]

                def gas_fn():
                    gas_predictor(x_gas, y_gas, total_frames, batch_size)

                def dp_fn():
                    dp_predictor(x_dp, y_dp, total_frames, batch_size)

                gas_time = _time_fn(gas_fn, args.runs, args.warmup, device)
                dp_time = _time_fn(dp_fn, args.runs, args.warmup, device)
                results[model_name][stage] = {
                    "predicted_frames": int(predict_frames * stage),
                    "total_frames": int(total_frames),
                    "gas_seconds_per_video": gas_time / batch_size,
                    "pressure_seconds_per_video": dp_time / batch_size,
                    "total_seconds_per_video": (gas_time + dp_time) / batch_size,
                }

        elif model_name == "fourier_mionet":
            gas_predictor = make_mionet_fourier_predictor(model_dir, gas_ckpt, device, modality="gas")
            dp_predictor = make_mionet_fourier_predictor(model_dir, pressure_ckpt, device, modality="pressure")
            for stage in stages:
                total_frames = context_frames + predict_frames * stage
                x_gas = gas_inputs[..., :total_frames, :]
                y_gas = gas_targets[:, :total_frames]
                x_dp = dp_inputs[..., :total_frames, :]
                y_dp = dp_targets[:, :total_frames]

                def gas_fn():
                    gas_predictor(x_gas, y_gas, total_frames, batch_size)

                def dp_fn():
                    dp_predictor(x_dp, y_dp, total_frames, batch_size)

                gas_time = _time_fn(gas_fn, args.runs, args.warmup, device)
                dp_time = _time_fn(dp_fn, args.runs, args.warmup, device)
                results[model_name][stage] = {
                    "predicted_frames": int(predict_frames * stage),
                    "total_frames": int(total_frames),
                    "gas_seconds_per_video": gas_time / batch_size,
                    "pressure_seconds_per_video": dp_time / batch_size,
                    "total_seconds_per_video": (gas_time + dp_time) / batch_size,
                }

        else:
            print(f"Skipping {model_name}: unsupported")
            continue

    lavig_cfg = cfg.get("lavig_flow", {}) or {}
    if lavig_cfg.get("enabled", False):
        lavig_cfg_path = _as_path(lavig_cfg.get("config"))
        if lavig_cfg_path is None:
            lavig_cfg_path = DEFAULT_EVAL_ROOT / "joint_autoreg.yaml"
        if not lavig_cfg_path.exists():
            raise ValueError(
                f"LAViG-FLOW config not found: {lavig_cfg_path}. "
                "Set lavig_flow.config in benchmarking/benchmark_comparison/config.yaml."
            )
        print("Timing lavig_flow ...")
        lavig_req = str(lavig_cfg.get("device") or device).lower()
        if lavig_req.startswith("cuda") and not torch.cuda.is_available():
            print("[timing] LAViG CUDA requested but not available; falling back to CPU.", flush=True)
            lavig_req = "cpu"
        lavig_device = torch.device(lavig_req)
        lavig_batch_size = int(lavig_cfg.get("batch_size", batch_size))
        include_decode = bool(lavig_cfg.get("include_decode", True))
        include_encode = bool(lavig_cfg.get("include_encode", False))
        results["lavig_flow"] = _time_lavig_flow(
            lavig_cfg_path,
            stages,
            lavig_device,
            args.runs,
            args.warmup,
            lavig_batch_size,
            include_decode=include_decode,
            include_encode=include_encode,
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved timing JSON to {out_path}")


if __name__ == "__main__":
    main()
