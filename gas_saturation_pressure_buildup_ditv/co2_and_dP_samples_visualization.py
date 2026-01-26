###############################################################################
# CO₂ & ΔP Samples Visualisation CLI (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Visualise CO₂ gas saturation and pressure build-up samples. The script
#   loads tensor datasets, denormalises the static/temporal fields, and saves
#   high-resolution figures for property maps, scalar parameters, and the
#   generated time series.
###############################################################################

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from paths import CO2_DATA_ROOT, DP_DATA_ROOT, JOINT_DITV_ROOT


DATA_DIRS: Dict[str, Dict[str, Path]] = {
    "sg": {
        "train": CO2_DATA_ROOT / "training_dataset_gas_saturation",
        "validation": CO2_DATA_ROOT / "validation_dataset_gas_saturation",
        "test": CO2_DATA_ROOT / "test_dataset_gas_saturation",
    },
    "dP": {
        "train": DP_DATA_ROOT / "training_dataset_pressure_buildup",
        "validation": DP_DATA_ROOT / "validation_dataset_pressure_buildup",
        "test": DP_DATA_ROOT / "test_dataset_pressure_buildup",
    },
}

OUTPUT_DIRS: Dict[str, Dict[str, Path]] = {
    "sg": {
        "train": JOINT_DITV_ROOT / "samples_training_dataset_gas_saturation",
        "validation": JOINT_DITV_ROOT / "samples_validation_dataset_gas_saturation",
        "test": JOINT_DITV_ROOT / "samples_test_dataset_gas_saturation",
    },
    "dP": {
        "train": JOINT_DITV_ROOT / "samples_training_dataset_pressure_buildup",
        "validation": JOINT_DITV_ROOT / "samples_validation_dataset_pressure_buildup",
        "test": JOINT_DITV_ROOT / "samples_test_dataset_pressure_buildup",
    },
}


DX = np.cumsum(3.5938 * np.power(1.035012, np.arange(200))) + 0.1
X_FULL, Y_FULL = np.meshgrid(DX, np.linspace(0, 200, num=96))


# --------------------------------------------------
# Utility: denormalisation helpers
# --------------------------------------------------

def dnorm_inj(a: np.ndarray) -> np.ndarray:
    """Denormalize the injection rate tensor to MT/yr."""
    return (a * (3e6 - 3e5) + 3e5) / (1e6 / 365.0 * 1000.0 / 1.862)


def dnorm_temp(a: np.ndarray) -> np.ndarray:
    """Denormalize the reservoir temperature tensor to °C."""
    return a * (180.0 - 30.0) + 30.0


def dnorm_pressure(a: np.ndarray) -> np.ndarray:
    """Denormalize the initial reservoir pressure to bar."""
    return a * (300.0 - 100.0) + 100.0


def dnorm_lambda(a: np.ndarray) -> np.ndarray:
    """Denormalize the Van Genuchten scaling factor."""
    return a * 0.4 + 0.3


def dnorm_swi(a: np.ndarray) -> np.ndarray:
    """Denormalize the irreducible water saturation."""
    return a * 0.2 + 0.1


def dnorm_dp(field: np.ndarray) -> np.ndarray:
    """Denormalize the pressure build-up field to bar."""
    return field * 18.772821433027488 + 4.172939172019009


# --------------------------------------------------
# Utility: plotting helpers
# --------------------------------------------------

def make_time_labels(num_steps: int) -> Iterable[str]:
    """Generate readable time stamps (days/years) for each frame."""
    times = np.cumsum(np.power(1.421245, np.arange(num_steps)))
    for value in times:
        if value < 365:
            yield f"{int(round(value))} d"
        else:
            yield f"{round(value / 365, 1)} y"


def plot_field(
    field: np.ndarray,
    thickness: int,
    outfile: Path,
    *,
    title: str | None = None,
    xlim: Tuple[float, float] | None = None,
    hide_axes: bool = False,
) -> None:
    """Render and save a single spatial field."""
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(
        X_FULL[:thickness, :],
        Y_FULL[:thickness, :],
        np.flipud(field),
        shading="auto",
        cmap="jet",
    )
    limits = xlim or (0.0, float(X_FULL[:thickness, :].max()))
    ax.set_xlim(limits)
    if hide_axes:
        ax.axis("off")
    else:
        ax.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.04)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0 if hide_axes else None,
    )
    plt.close(fig)


def save_scalar_card(
    outfile: Path,
    *,
    inj_rate: float,
    temperature: float,
    initial_pressure: float,
    swi: float,
    vg_lambda: float,
) -> None:
    """Create a text card summarizing the scalar inputs."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        (
            f"Injection Rate ($Q$) = {inj_rate:.2f} MT/yr\n"
            f"Isothermal Reservoir Temperature ($T$) = {temperature:.1f} °C\n"
            f"Initial Reservoir Pressure ($P_{{init}}$) = {initial_pressure:.1f} bar\n"
            f"Irreducible Water Saturation ($S_{{wi}}$) = {swi:.2f}\n"
            f"Van Genuchten Scaling Factor ($\\lambda$) = {vg_lambda:.2f}"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# Utility: tensor extraction helpers
# --------------------------------------------------

def extract_maps(sample_inputs: np.ndarray) -> Tuple[int, np.ndarray, Dict[str, np.ndarray]]:
    """Split static property maps (k_r/k_z/phi/perf) from model inputs."""
    grid = sample_inputs[:, :, 0, :]
    mask = grid[:, :, 0] != 0
    thickness = int(mask[:, 0].sum())
    if thickness == 0:
        mask = np.ones_like(mask, dtype=bool)
        thickness = grid.shape[0]

    maps = {
        "kr": np.exp(grid[:, :, 0][mask].reshape(thickness, -1) * 15.0),
        "kz": np.exp(grid[:, :, 1][mask].reshape(thickness, -1) * 15.0),
        "phi": grid[:, :, 2][mask].reshape(thickness, -1),
        "perf": grid[:, :, 3][mask].reshape(thickness, -1),
    }
    return thickness, mask, maps


def extract_scalars(sample_inputs: np.ndarray) -> Dict[str, float]:
    """Extract and denormalize scalar parameters from the input tensor."""
    scalars = sample_inputs[0, 0, 0, :]
    return {
        "inj_rate": float(dnorm_inj(scalars[4])),
        "temperature": float(dnorm_temp(scalars[5])),
        "initial_pressure": float(dnorm_pressure(scalars[6])),
        "swi": float(dnorm_swi(scalars[7])),
        "vg_lambda": float(dnorm_lambda(scalars[8])),
    }


def prepare_output_stack(sample_outputs: np.ndarray) -> np.ndarray:
    """Squeeze the network output to shape [H, W, T]."""
    stack = np.squeeze(sample_outputs, axis=0)
    if stack.ndim == 4 and stack.shape[-1] == 1:
        stack = np.squeeze(stack, axis=-1)
    if stack.ndim != 3:
        raise ValueError(f"Unexpected output tensor shape {sample_outputs.shape}")
    return stack


# --------------------------------------------------
# Pipeline: dataset traversal
# --------------------------------------------------

def process_dataset(
    parameter_type: str,
    dataset_type: str,
    max_samples: int,
    device: torch.device,
    *,
    unit: str,
    denorm_output: Callable[[np.ndarray], np.ndarray] | None = None,
    axial_xlim: Tuple[float, float] | None = None,
) -> None:
    """Iterate over a dataset split and save static maps and temporal fields."""
    data_dir = Path(DATA_DIRS[parameter_type][dataset_type])
    output_dir = Path(OUTPUT_DIRS[parameter_type][dataset_type])
    input_path = data_dir / f"{parameter_type}_{dataset_type}_inputs.pt"
    output_path = data_dir / f"{parameter_type}_{dataset_type}_outputs.pt"

    if not input_path.exists() or not output_path.exists():
        print(f"Skipping {parameter_type}-{dataset_type}: missing data tensors.")
        return

    print(f"Loading {parameter_type}-{dataset_type} from {data_dir}")
    inputs = torch.load(input_path, map_location=device)
    outputs = torch.load(output_path, map_location=device)
    print(f"  Inputs shape: {tuple(inputs.shape)}, Outputs shape: {tuple(outputs.shape)}")

    dataset = torch.utils.data.TensorDataset(inputs, outputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    time_labels = None

    for index, (sample_inputs, sample_outputs) in enumerate(loader):
        if max_samples >= 0 and index >= max_samples:
            break

        sample_inputs_np = sample_inputs.cpu().numpy()
        sample_outputs_np = sample_outputs.cpu().numpy()

        thickness, mask, parameter_maps = extract_maps(sample_inputs_np[0])
        scalar_values = extract_scalars(sample_inputs_np[0])
        output_stack = prepare_output_stack(sample_outputs_np)

        if time_labels is None:
            time_labels = list(make_time_labels(output_stack.shape[-1]))

        print(
            f"Processing sample {index + 1:03d} "
            f"(thickness={thickness}, unit={unit}, time steps={len(time_labels)})"
        )

        for name, (title, limit) in {
            "kr": ("$k_r$ (mD)", (0.0, 1e5)),
            "kz": ("$k_z$ (mD)", (0.0, 1e5)),
            "phi": ("$\\phi$ (-)", None),
            "perf": ("Perforation", (0.0, 2e3)),
        }.items():
            outfile = (
                output_dir
                / f"{dataset_type}_{parameter_type}_sample_{index + 1}_{name}.png"
            )
            plot_field(parameter_maps[name], thickness, outfile, title=title, xlim=limit)

        scalar_card = (
            output_dir
            / f"{dataset_type}_{parameter_type}_sample_{index + 1}_Scalar_Parameters.png"
        )
        save_scalar_card(scalar_card, **scalar_values)

        for t in range(output_stack.shape[-1]):
            field = output_stack[:, :, t]
            if denorm_output is not None:
                field = denorm_output(field)
            field = field[mask].reshape(thickness, -1)
            outfile = (
                output_dir
                / f"{dataset_type}_{parameter_type}_sample_{index + 1}_time_{time_labels[t]}.png"
            )
            plot_field(field, thickness, outfile, hide_axes=True, xlim=axial_xlim)

    print(f"Finished processing {parameter_type}-{dataset_type}; results in {output_dir}")


# --------------------------------------------------
# CLI plumbing
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Configure CLI options for dataset selection and plotting bounds."""
    parser = argparse.ArgumentParser(
        description="Generate sample visualisations for gas saturation and pressure build-up."
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        default=["train"],
        choices=["train", "validation", "test"],
        help="Datasets to visualise (default: train).",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=["sg", "dP"],
        choices=["sg", "dP"],
        help="Parameter types to visualise (default: sg dP).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of samples per dataset (default: 20). Use -1 for all samples.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading tensors (default: auto-detect).",
    )
    parser.add_argument(
        "--xlim-sg",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Optional x-axis limits for saturation plots (e.g. 0 3500).",
    )
    parser.add_argument(
        "--xlim-dp",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Optional x-axis limits for pressure plots (default: 0 3500).",
    )
    return parser.parse_args()


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------

def main() -> None:
    """Entry point that parses args and launches dataset processing."""
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    for parameter_type in args.parameters:
        for dataset_type in args.dataset_types:
            denorm = dnorm_dp if parameter_type == "dP" else None
            unit = "bar" if parameter_type == "dP" else "-"
            if parameter_type == "dP":
                cutoff = tuple(args.xlim_dp) if args.xlim_dp else (0.0, 1500.0)
            else:
                cutoff = tuple(args.xlim_sg) if args.xlim_sg else None
            process_dataset(
                parameter_type,
                dataset_type,
                args.max_samples,
                device,
                unit=unit,
                denorm_output=denorm,
                axial_xlim=cutoff,
            )


if __name__ == "__main__":
    main()