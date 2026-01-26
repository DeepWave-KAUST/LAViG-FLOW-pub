###############################################################################
# Pressure Build-Up dataset preparation helpers (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import os
import torch
from paths import DP_DATA_ROOT

DP_VAE_ROOT = DP_DATA_ROOT

# ---------------------------
# Phys transform for pressure
# ---------------------------
def dnorm_dP(dP):
    # Apply only for TRAIN/VAL (not TEST)
    dP = dP * 18.772821433027488
    dP = dP + 4.172939172019009
    return dP

# --------------------------------------------------
# Logging: per-sample stats before normalisation
# --------------------------------------------------
def write_before_flat_log(dir_path, flat_outputs, tag):
    """
    Write sample-wise BEFORE-normalization min/max in flattened order (N=B*T):
    'Sample 1: Min Before=..., Max Before=...'
    """
    os.makedirs(dir_path, exist_ok=True)
    per_sample_min = flat_outputs.amin(dim=(1, 2, 3))
    per_sample_max = flat_outputs.amax(dim=(1, 2, 3))

    log_path = os.path.join(dir_path, f"{tag}_min_max_before_norm.txt")
    with open(log_path, "w") as f:
        f.write("Sample-wise Min–Max Values BEFORE Normalization (flattened N=B*T)\n")
        f.write("=" * 90 + "\n")
        for i in range(len(per_sample_min)):
            f.write(f"Sample {i+1}: Min Before={per_sample_min[i].item():.4f}, "
                    f"Max Before={per_sample_max[i].item():.4f}\n")
    print(f"[outputs] BEFORE norm (sample-wise) log saved -> {log_path}")


# --------------------------------------------------
# IO: persist flattened pre-normalisation stats
# --------------------------------------------------
def save_per_sample_stats(dir_path, flat_outputs, T, tag):
    """
    Save min/max for each flattened sample (N=B*T) with index map (B_idx, T_idx).
    flat_outputs: [N,1,H,W] BEFORE normalization
    """
    os.makedirs(dir_path, exist_ok=True)
    N = flat_outputs.shape[0]
    per_sample_min = flat_outputs.amin(dim=(1, 2, 3))   # [N]
    per_sample_max = flat_outputs.amax(dim=(1, 2, 3))   # [N]
    idxs = torch.arange(N)
    index_map = torch.stack([idxs // T, idxs % T], dim=1)  # [N,2]

    # Same naming style as your sg script
    pt_path = os.path.join(dir_path, f'{tag}_min_max.pt')
    torch.save({
        "per_sample_min": per_sample_min.cpu(),
        "per_sample_max": per_sample_max.cpu(),
        "index_map": index_map.cpu(),
        "frames_per_clip": int(T),
    }, pt_path)
    print(f"Saved flattened per-sample BEFORE stats -> {pt_path}")


# --------------------------------------------------
# Preprocessing: load, normalise, flatten split tensors
# --------------------------------------------------
def process_split(tag, input_path, output_path, save_dir, apply_denorm: bool):
    """
    tag: 'dP_train' | 'dP_val' | 'dP_test'
    apply_denorm: True for train/val, False for test
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load
    dP_inputs  = torch.load(input_path).float()   # [B,H,W,T,C]
    dP_outputs = torch.load(output_path).float()  # [B,H,W,T]

    # Slice to porosity + first 17 frames
    dP_inputs  = dP_inputs[:, :, :, :, 2][:, :, :, :17]   # [B,H,W,17]
    dP_outputs = dP_outputs[:, :, :, :17]                 # [B,H,W,17]

    # Move time next to batch
    dP_inputs  = dP_inputs.permute(0, 3, 1, 2)    # [B,T,H,W]
    dP_outputs = dP_outputs.permute(0, 3, 1, 2)   # [B,T,H,W]

    print(f"\n=== {tag.upper()} SHAPES ===")
    print("Inputs:", dP_inputs.shape, "Outputs:", dP_outputs.shape)

    B, T, H, W = dP_outputs.shape

    # ---- (A) BEFORE stats (per-(B,T)) ----
    # Apply physics transform conditionally, then clamp zeros for stats
    if apply_denorm:
        dP_outputs_phys = dnorm_dP(dP_outputs)
    else:
        dP_outputs_phys = dP_outputs  # no denorm on TEST

    dP_outputs_phys = torch.clamp(dP_outputs_phys, min=0.0)  # clamp for ALL splits

    out_BT1HW = dP_outputs_phys.unsqueeze(2)  # [B,T,1,H,W]
    per_bt_min = out_BT1HW.amin(dim=(3, 4)).squeeze(2)  # [B,T]
    per_bt_max = out_BT1HW.amax(dim=(3, 4)).squeeze(2)  # [B,T]
    global_min = out_BT1HW.amin(dim=(0, 2, 3, 4))       # [T] or scalar if you prefer; keeping [T] aligns w/ sg
    global_max = out_BT1HW.amax(dim=(0, 2, 3, 4))

    torch.save({
        "per_bt_min": per_bt_min.cpu(),
        "per_bt_max": per_bt_max.cpu(),
        "global_min": global_min.cpu(),
        "global_max": global_max.cpu(),
    }, os.path.join(save_dir, f"{tag}_min_max.pt"))

    # ---- (B) FLATTEN and save BEFORE per-sample stats ----
    dP_inputs_flat  = dP_inputs.reshape(-1, 1, H, W)           # [N,1,H,W]
    dP_outputs_flat = dP_outputs_phys.reshape(-1, 1, H, W)     # [N,1,H,W] — phys+clamped
    print("Flattened shape:", dP_inputs_flat.shape)

    # BEFORE logs (.txt + .pt) in flattened order
    write_before_flat_log(save_dir, dP_outputs_flat, tag)
    save_per_sample_stats(save_dir, dP_outputs_flat, T, tag)

    # ---- (C) NORMALIZE ----
    # Inputs: global [-1,1]
    min_in = dP_inputs_flat.min()
    max_in = dP_inputs_flat.max()
    dP_inputs_flat = 2 * (dP_inputs_flat - min_in) / (max_in - min_in) - 1

    # Outputs: per-sample [-1,1]
    after_txt = os.path.join(save_dir, f"{tag}_min_max_after_norm.txt")
    with open(after_txt, "w") as logf:
        logf.write("Sample-wise Min–Max Values AFTER Normalization (expect ~[-1,1])\n")
        logf.write("=" * 90 + "\n")
        for i in range(len(dP_outputs_flat)):
            mn = dP_outputs_flat[i].min()
            mx = dP_outputs_flat[i].max()
            if (mx - mn) == 0:
                dP_outputs_flat[i] = torch.zeros_like(dP_outputs_flat[i])
            else:
                dP_outputs_flat[i] = 2 * (dP_outputs_flat[i] - mn) / (mx - mn) - 1
            logf.write(f"Sample {i+1}: Min After={dP_outputs_flat[i].min().item():.4f}, "
                       f"Max After={dP_outputs_flat[i].max().item():.4f}\n")

    print(f"[{tag}] AFTER normalization log saved -> {after_txt}")
    return dP_inputs_flat, dP_outputs_flat


# --------------------------------------------------
# Orchestration: prepare train/val/test datasets
# --------------------------------------------------
print("=== TRAINING DATASET PREPARATION ===")
train_inputs, train_outputs = process_split(
    tag="dP_train",
    input_path=DP_VAE_ROOT / "training_dataset_pressure_buildup/dP_train_inputs.pt",
    output_path=DP_VAE_ROOT / "training_dataset_pressure_buildup/dP_train_outputs.pt",
    save_dir=DP_VAE_ROOT / "training_dataset_pressure_buildup",
    apply_denorm=True
)

print("\n=== VALIDATION DATASET PREPARATION ===")
val_inputs, val_outputs = process_split(
    tag="dP_val",
    input_path=DP_VAE_ROOT / "validation_dataset_pressure_buildup/dP_val_inputs.pt",
    output_path=DP_VAE_ROOT / "validation_dataset_pressure_buildup/dP_val_outputs.pt",
    save_dir=DP_VAE_ROOT / "validation_dataset_pressure_buildup",
    apply_denorm=True
)

print("\n=== TEST DATASET PREPARATION ===")
test_inputs, test_outputs = process_split(
    tag="dP_test",
    input_path=DP_VAE_ROOT / "test_dataset_pressure_buildup/dP_test_inputs.pt",
    output_path=DP_VAE_ROOT / "test_dataset_pressure_buildup/dP_test_outputs.pt",
    save_dir=DP_VAE_ROOT / "test_dataset_pressure_buildup",
    apply_denorm=False   # <-- NO denorm on test, only clamp
)

# Save normalized flattened targets
torch.save(train_outputs, DP_VAE_ROOT / "train_dP_target.pt")
torch.save(val_outputs,   DP_VAE_ROOT / "val_dP_target.pt")
torch.save(test_outputs,  DP_VAE_ROOT / "test_dP_target.pt")

print("\nShapes:")
print(train_outputs.shape, val_outputs.shape, test_outputs.shape)
print("\nPressure Build-Up Dataset Prepared!")
