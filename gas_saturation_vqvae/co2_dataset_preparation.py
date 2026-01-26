###############################################################################
# CO₂ dataset preparation helpers (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import os
import torch
from paths import CO2_DATA_ROOT

VQVAE_CO2_ROOT = CO2_DATA_ROOT


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
    index_map = torch.stack([idxs // T, idxs % T], dim=1)

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
def process_split(tag, input_path, output_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    sg_inputs  = torch.load(input_path).float()   # [B,H,W,T,C]
    sg_outputs = torch.load(output_path).float()  # [B,H,W,T]
    sg_inputs  = sg_inputs[:, :, :, :, 2][:, :, :, :17]
    sg_outputs = sg_outputs[:, :, :, :17]

    sg_inputs  = sg_inputs.permute(0, 3, 1, 2)   # [B,T,H,W]
    sg_outputs = sg_outputs.permute(0, 3, 1, 2)
    print(f"\n=== {tag.upper()} SHAPES ===")
    print("Inputs:", sg_inputs.shape, "Outputs:", sg_outputs.shape)

    B, T, H, W = sg_outputs.shape

    # --- (A) per-(B,T) stats (for completeness)
    out_BT1HW = sg_outputs.unsqueeze(2)
    per_bt_min = out_BT1HW.amin(dim=(3,4)).squeeze(2)
    per_bt_max = out_BT1HW.amax(dim=(3,4)).squeeze(2)
    global_min = out_BT1HW.amin()
    global_max = out_BT1HW.amax()
    
    torch.save({
        "per_bt_min": per_bt_min.cpu(),
        "per_bt_max": per_bt_max.cpu(),
        "global_min": global_min.cpu(),
        "global_max": global_max.cpu(),
    }, os.path.join(save_dir, f"{tag}_min_max.pt"))

    # --- (B) Flatten [B*T,1,H,W]
    sg_inputs_flat  = sg_inputs.reshape(-1, 1, H, W)
    sg_outputs_flat = sg_outputs.reshape(-1, 1, H, W)
    print("Flattened shape:", sg_inputs_flat.shape)

    # BEFORE logs (.txt + .pt)
    write_before_flat_log(save_dir, sg_outputs_flat, tag)
    save_per_sample_stats(save_dir, sg_outputs_flat, T, tag)

    # --- Normalize
    # Global normalization for inputs
    min_in = sg_inputs_flat.min()
    max_in = sg_inputs_flat.max()
    sg_inputs_flat = 2 * (sg_inputs_flat - min_in) / (max_in - min_in) - 1

    # Per-sample normalization for outputs
    after_txt = os.path.join(save_dir, f"{tag}_min_max_after_norm.txt")
    with open(after_txt, "w") as logf:
        logf.write("Sample-wise Min–Max Values AFTER Normalization (expect ~[-1,1])\n")
        logf.write("=" * 90 + "\n")
        for i in range(len(sg_outputs_flat)):
            mn = sg_outputs_flat[i].min()
            mx = sg_outputs_flat[i].max()
            if (mx - mn) == 0:
                sg_outputs_flat[i] = torch.zeros_like(sg_outputs_flat[i])
            else:
                sg_outputs_flat[i] = 2 * (sg_outputs_flat[i] - mn) / (mx - mn) - 1
            logf.write(f"Sample {i+1}: Min After={sg_outputs_flat[i].min().item():.4f}, "
                       f"Max After={sg_outputs_flat[i].max().item():.4f}\n")

    print(f"[{tag}] AFTER normalization log saved -> {after_txt}")
    return sg_inputs_flat, sg_outputs_flat


# --------------------------------------------------
# Orchestration: prepare train/val/test datasets
# --------------------------------------------------
print("=== TRAINING DATASET PREPARATION ===")
train_inputs, train_outputs = process_split(
    tag="sg_train",
    input_path=VQVAE_CO2_ROOT / "training_dataset_gas_saturation/sg_train_inputs.pt",
    output_path=VQVAE_CO2_ROOT / "training_dataset_gas_saturation/sg_train_outputs.pt",
    save_dir=VQVAE_CO2_ROOT / "training_dataset_gas_saturation"
)

print("\n=== VALIDATION DATASET PREPARATION ===")
val_inputs, val_outputs = process_split(
    tag="sg_val",
    input_path=VQVAE_CO2_ROOT / "validation_dataset_gas_saturation/sg_val_inputs.pt",
    output_path=VQVAE_CO2_ROOT / "validation_dataset_gas_saturation/sg_val_outputs.pt",
    save_dir=VQVAE_CO2_ROOT / "validation_dataset_gas_saturation"
)

print("\n=== TEST DATASET PREPARATION ===")
test_inputs, test_outputs = process_split(
    tag="sg_test",
    input_path=VQVAE_CO2_ROOT / "test_dataset_gas_saturation/sg_test_inputs.pt",
    output_path=VQVAE_CO2_ROOT / "test_dataset_gas_saturation/sg_test_outputs.pt",
    save_dir=VQVAE_CO2_ROOT / "test_dataset_gas_saturation"
)

# Save normalized flattened targets
torch.save(train_outputs, VQVAE_CO2_ROOT / "train_sg_target.pt")
torch.save(val_outputs,   VQVAE_CO2_ROOT / "val_sg_target.pt")
torch.save(test_outputs,  VQVAE_CO2_ROOT / "test_sg_target.pt")

print("\nShapes:")
print(train_outputs.shape, val_outputs.shape, test_outputs.shape)
print("\n Gas Saturation Dataset Prepared Successfully!")
