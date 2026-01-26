###############################################################################
# CO₂ Evaluation Dataset Preparation (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Builds the validation/evaluation tensors for the joint DiTV pipeline by
#   reusing the raw CO₂ gas saturation tensors produced inside
#   `gas_saturation_vqvae`. The script mirrors the train-time preprocessing
#   (temporal slicing, channel reshaping, local min-max normalisation) and
#   emits `val_sg_target.pt` plus the accompanying statistics underneath
#   `gas_saturation_pressure_buildup_ditv/co2_data_evaluation/`.
###############################################################################

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import torch

from paths import CO2_DATA_ROOT, JOINT_DITV_ROOT


SRC_VAL_DIR = CO2_DATA_ROOT / "validation_dataset_gas_saturation"
DST_ROOT = JOINT_DITV_ROOT / "co2_data_evaluation"
DST_STATS_DIR = DST_ROOT / "validation_dataset_gas_saturation"


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    print("VALIDATION DATASET PREPARATION\n")

    if not SRC_VAL_DIR.exists():
        raise FileNotFoundError(
            f"Missing CO₂ validation tensors under {SRC_VAL_DIR}. "
            "Run the single-task VQ-VAE preprocessing first."
        )

    ensure_dir(DST_ROOT)
    dst_stats = ensure_dir(DST_STATS_DIR)

    # Load validation tensors
    inputs_path = SRC_VAL_DIR / "sg_val_inputs.pt"
    outputs_path = SRC_VAL_DIR / "sg_val_outputs.pt"
    val_sg_inputs = torch.load(inputs_path)
    val_sg_outputs = torch.load(outputs_path)

    print(f"Loaded inputs:  {val_sg_inputs.shape}")
    print(f"Loaded targets: {val_sg_outputs.shape}")

    # Add channel dim to outputs and permute tensors to [B, T, C, H, W]
    val_sg_outputs = val_sg_outputs.unsqueeze(-1)
    val_sg_inputs = val_sg_inputs.permute(0, 3, 4, 1, 2)
    val_sg_outputs = val_sg_outputs.permute(0, 3, 4, 1, 2)

    print(f"New shape (inputs):  {val_sg_inputs.shape}")
    print(f"New shape (targets): {val_sg_outputs.shape}")

    # ------------------------------------------------------------------ #
    # Local min-max normalization of conditioning inputs (per sample/T/C)
    # ------------------------------------------------------------------ #
    B, T, Cin, H, W = val_sg_inputs.shape
    mins_inp = val_sg_inputs.amin(dim=(3, 4), keepdim=True)
    maxs_inp = val_sg_inputs.amax(dim=(3, 4), keepdim=True)

    before_norm_path = dst_stats / "sg_val_conditions_min_max_before_normalization.txt"
    with before_norm_path.open("w") as handle:
        handle.write(
            "Condition Min–Max Values BEFORE Local Normalization "
            "(per sample, per condition, per time frame)\n"
        )
        handle.write("=" * 100 + "\n")
        for c in range(Cin):
            handle.write(f"\n=== Condition {c + 1} ===\n")
            for t in range(T):
                handle.write(f"\n  Time frame {t + 1}:\n" + "-" * 50 + "\n")
                cmins = mins_inp[:, t, c, 0, 0]
                cmaxs = maxs_inp[:, t, c, 0, 0]
                for i in range(B):
                    handle.write(
                        f"Sample {i + 1}: Min={cmins[i].item():.4f}, "
                        f"Max={cmaxs[i].item():.4f}\n"
                    )
    print(f"[inputs] BEFORE norm log saved -> {before_norm_path}")

    ranges = maxs_inp - mins_inp
    zero_mask = ranges == 0
    ranges_safe = ranges.masked_fill(zero_mask, 1.0)
    norm_inputs = 2.0 * (val_sg_inputs - mins_inp) / ranges_safe - 1.0
    norm_inputs = torch.where(zero_mask, val_sg_inputs, norm_inputs)
    val_sg_inputs = norm_inputs

    after_norm_path = dst_stats / "sg_val_conditions_min_max_after_normalization.txt"
    with after_norm_path.open("w") as handle:
        handle.write(
            "Condition Min–Max Values AFTER Local Normalization "
            "(per sample, per condition, per time frame)\n"
        )
        handle.write("=" * 100 + "\n")
        for c in range(Cin):
            handle.write(f"\n=== Condition {c + 1} ===\n")
            for t in range(T):
                handle.write(f"\n  Time frame {t + 1}:\n" + "-" * 60 + "\n")
                mins_post = val_sg_inputs[:, t, c].amin(dim=(1, 2))
                maxs_post = val_sg_inputs[:, t, c].amax(dim=(1, 2))
                for i in range(B):
                    handle.write(
                        f"Sample {i + 1}: Min={mins_post[i].item():.4f}, "
                        f"Max={maxs_post[i].item():.4f}\n"
                    )
    print(f"[inputs] AFTER norm log saved -> {after_norm_path}")

    # ------------------------------------------------------------------ #
    # Target stats (per sample/time + global per frame)
    # ------------------------------------------------------------------ #
    B, T, Cout, H, W = val_sg_outputs.shape
    assert Cout == 1, f"Expected single-channel targets, got {Cout}"

    per_bt_min = val_sg_outputs.amin(dim=(3, 4)).squeeze(2)
    per_bt_max = val_sg_outputs.amax(dim=(3, 4)).squeeze(2)
    global_min = val_sg_outputs.amin(dim=(0, 2, 3, 4))
    global_max = val_sg_outputs.amax(dim=(0, 2, 3, 4))

    torch.save(
        {
            "per_bt_min": per_bt_min.cpu(),
            "per_bt_max": per_bt_max.cpu(),
            "global_min": global_min.cpu(),
            "global_max": global_max.cpu(),
        },
        dst_stats / "sg_val_min_max.pt",
    )

    before_targets_path = dst_stats / "sg_val_min_max_before_normalization.txt"
    with before_targets_path.open("w") as handle:
        handle.write("Gas Saturation Min–Max BEFORE Normalization\n")
        handle.write("=" * 80 + "\n")
        for t in range(T):
            handle.write(f"\n=== Time frame {t + 1} ===\n")
            handle.write("-" * 40 + "\n")
            mins = val_sg_outputs[:, t].amin(dim=(1, 2, 3))
            maxs = val_sg_outputs[:, t].amax(dim=(1, 2, 3))
            for i in range(B):
                handle.write(
                    f"Sample {i + 1}: Min={mins[i].item():.4f}, "
                    f"Max={maxs[i].item():.4f}\n"
                )

    after_targets_path = dst_stats / "sg_val_min_max_after_normalization.txt"
    with after_targets_path.open("w") as handle:
        handle.write(
            "Gas Saturation Min–Max AFTER Normalization (per sample, per time frame)\n"
        )
        handle.write("=" * 80 + "\n")
        for t in range(T):
            handle.write(f"\n=== Time frame {t + 1} ===\n")
            handle.write("-" * 40 + "\n")
            for i in range(B):
                arr = val_sg_outputs[i, t]
                mn = arr.min()
                mx = arr.max()
                norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
                val_sg_outputs[i, t] = norm
                handle.write(
                    f"Sample {i + 1}: Min={norm.min().item():.4f}, "
                    f"Max={norm.max().item():.4f}\n"
                )

    print(f"Target stats saved under {dst_stats}")

    # ------------------------------------------------------------------ #
    # Persist final tensors
    # ------------------------------------------------------------------ #
    val_sg_target = val_sg_outputs
    target_path = DST_ROOT / "val_sg_target.pt"
    torch.save(val_sg_target, target_path)
    print(f"Saved gas evaluation targets -> {target_path}")

    print("Gas Saturation Evaluation Dataset Prepared!\n")


if __name__ == "__main__":
    main()