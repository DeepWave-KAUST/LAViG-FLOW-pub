###############################################################################
# ΔP (Pressure Buildup) Dataset Preparation (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Preprocess the raw pressure buildup tensors for the diffusion/autoencoder
#   pipelines. The script trims the temporal window, adds a channel dimension,
#   permutes tensors into [B, T, C, H, W], applies local min–max normalization,
#   and logs statistics for all dataset splits.
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import os
import torch
from paths import DP_DATA_ROOT, JOINT_DP_ROOT

DP_VAE_ROOT = DP_DATA_ROOT
JOINT_DP_ROOT.mkdir(parents=True, exist_ok=True)

def ensure_dir(path):
    """Create the directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# Define function to adjust pressure buildup data for correct plotting
# ---------------------------------------------------------------------
def dnorm_dP(dP):
    """
    Adjusts pressure buildup data to ensure it is plotted correctly.

    This function applies a scaling factor and offset to the input pressure 
    buildup data so that the values are suitable for accurate visualization.

    Parameters
    ----------
    dP : float or ndarray
        Input pressure buildup values to be adjusted.

    Returns
    -------
    float or ndarray
        Adjusted pressure buildup values for plotting.

    Notes
    -----
    The adjustment formula is:
        dP = dP * scale + offset
    where:
        scale = 18.772821433027488
        offset = 4.172939172019009
    """
    dP = dP * 18.772821433027488
    dP = dP + 4.172939172019009 
    return dP 

# --------------------------------------------------
# Training dataset preparation
# --------------------------------------------------
print("TRAINING DATASET PREPARATION\n")

# Define training dataset directory path
data_dir_dP_train = DP_VAE_ROOT / "training_dataset_pressure_buildup"

# Define inputs (conditions) and outputs (targets) variables
train_dP_inputs = torch.load(f'{data_dir_dP_train}/dP_train_inputs.pt')
train_dP_outputs = torch.load(f'{data_dir_dP_train}/dP_train_outputs.pt')

print(train_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(train_dP_outputs.shape)

# For inputs and outputs we take the first 17 time frames
train_dP_inputs  = train_dP_inputs[:, :, :, :17]  
train_dP_outputs = train_dP_outputs[:, :, :, :17]
print(train_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(train_dP_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]

# Add channel dimension for train_dP_outputs 
train_dP_outputs = train_dP_outputs.unsqueeze(-1)
print(train_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(train_dP_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

# Prepare input shape to video diffusion model 
# Inputs: [B, H, W, T, C] → [B, T, C, H, W]
train_dP_inputs = train_dP_inputs.permute(0, 3, 4, 1, 2)
train_dP_outputs = train_dP_outputs.permute(0, 3, 4, 1, 2)

# Print new shape
print("New shape:", train_dP_inputs.shape)   # Expected: [num_samples, time_frames, num_features (channels), height, width] -> All conditions (8 time steps)
print("New shape:", train_dP_outputs.shape)  # Expected: [num_samples, time_frames, num_features (channels), height, width] -> Pressure Build-Up (8 time steps)

# Local normalization for all 12 conditions 
# train_dP_inputs shape: [B, T, C, H, W]
B, T, Cin, H, W = train_dP_inputs.shape

# Compute min/max over (H,W) only → keep T
mins_inp = train_dP_inputs.amin(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]
maxs_inp = train_dP_inputs.amax(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]

# Define file path to save the MIN-MAX values BEFORE local normalization for the CONDITIONS
data_dir_dP_train = ensure_dir(JOINT_DP_ROOT / "training_dataset_pressure_buildup")
save_path = os.path.join(
    data_dir_dP_train, "dP_train_conditions_min_max_before_normalization.txt"
)

with open(save_path, "w") as f:
    f.write("Condition Min–Max Values BEFORE Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")

    for c in range(Cin):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*50 + "\n")
            cmins = mins_inp[:, t, c, 0, 0]  # [B]
            cmaxs = maxs_inp[:, t, c, 0, 0]  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={cmins[i].item():.4f}, Max={cmaxs[i].item():.4f}\n")

print(f"[inputs] BEFORE norm log saved -> {save_path}")

# === Local min–max normalization per (sample, time, condition) over (H,W) ===
# train_dP_inputs: [B, T, C, H, W]
B, T, C, H, W = train_dP_inputs.shape

# Compute per-(B,T,C) min/max over (H,W)
mins_inp = train_dP_inputs.amin(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
maxs_inp = train_dP_inputs.amax(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
ranges   = maxs_inp - mins_inp

# Avoid divide-by-zero (constant fields): where range==0, keep original values
zero_mask = ranges == 0
ranges_safe = ranges.masked_fill(zero_mask, 1.0)

# Normalize to [-1, 1]
norm_inp = 2.0 * (train_dP_inputs - mins_inp) / ranges_safe - 1.0
# Put back original values where field is constant across (H,W)
norm_inp = torch.where(zero_mask, train_dP_inputs, norm_inp)

# Store back
train_dP_inputs = norm_inp

# =========== SAVE min/max AFTER normalization (per condition, per time, per sample) ===========
save_path = os.path.join(data_dir_dP_train, "dP_train_conditions_min_max_after_normalization.txt")
with open(save_path, "w") as f:
    f.write("Condition Min–Max Values AFTER Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")
    for c in range(C):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*60 + "\n")
            # Reduce over (H,W) *after* normalization to show the per-(B,T,C) extrema
            mins_post = train_dP_inputs[:, t, c].amin(dim=(1, 2))  # [B]
            maxs_post = train_dP_inputs[:, t, c].amax(dim=(1, 2))  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={mins_post[i].item():.4f}, Max={maxs_post[i].item():.4f}\n")

print(f"[inputs] AFTER norm log saved -> {save_path}")

print("\nChecking global min/max for each condition and time frame after normalization:")

B, T, C, H, W = train_dP_inputs.shape

for c in range(C):  # loop over conditions
    print(f"\n=== Condition {c+1} ===")
    for t in range(T):  # loop over time frames
        global_min = train_dP_inputs[:, t, c, :, :].amin().item()
        global_max = train_dP_inputs[:, t, c, :, :].amax().item()
        print(f"  Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")
        
# Local normalization for Pressure Build-Up
# 1) Map to physical/ranged values
train_dP_outputs = dnorm_dP(train_dP_outputs)

# 2) Clamp negatives to zero (elementwise, across all samples/time frames)
train_dP_outputs = torch.clamp(train_dP_outputs, min=0)

# === Save min and max values for both conditional and unconditional generation ===
B, T, C, H, W = train_dP_outputs.shape
assert C == 1, f"Expected C=1, got {C}"

# Per-(sample,time) stats: [B,T]
per_bt_min = train_dP_outputs.amin(dim=(3, 4)).squeeze(2)  # [B,T]
per_bt_max = train_dP_outputs.amax(dim=(3, 4)).squeeze(2)  # [B,T]

# Global per-time stats across all samples/pixels: [T]
global_min = train_dP_outputs.amin(dim=(0, 2, 3, 4))       # [T]
global_max = train_dP_outputs.amax(dim=(0, 2, 3, 4))       # [T]

save_path = os.path.join(data_dir_dP_train, "dP_train_min_max.pt")
torch.save(
    {
        "per_bt_min": per_bt_min.cpu(),   # [B,T]  for each sample b (1 to 4500) and each time index t (1 to 8) for conditional generation when we have the ground truth
        "per_bt_max": per_bt_max.cpu(),   # [B,T]  for each sample b (1 to 4500) and each time index t (1 to 8) for conditional generation when we have the ground truth
        "global_min": global_min.cpu(),   # [T]    for each time index t, we find the single min and max across all samples for unconditional generation, there is no ground truth
        "global_max": global_max.cpu(),   # [T]    for each time index t, we find the single min and max across all samples for unconditional generation, there is no ground truth
    },
    save_path,
)
print(f"Saved stats -> {save_path}")
# === Save min and max values for both conditional and unconditional generation ===

save_path = os.path.join(data_dir_dP_train, "dP_train_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Pressure Build-Up Min–Max BEFORE Normalization\n")
    f.write("="*80 + "\n")

    for t in range(T):  # loop over 8 time frames
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        # Reduce over (C,H,W) for each sample
        mins = train_dP_outputs[:, t].amin(dim=(1, 2, 3))  # [B]
        maxs = train_dP_outputs[:, t].amax(dim=(1, 2, 3))  # [B]

        for i in range(B):
            f.write(f"Sample {i+1}: Min={mins[i].item():.4f}, Max={maxs[i].item():.4f}\n")

print(f"[outputs] BEFORE norm log saved -> {save_path}")

print("\nChecking global min/max for Pressure Build-Up before normalization:")

B, T, C, H, W = train_dP_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = train_dP_outputs[:, t].amin().item()
    global_max = train_dP_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# train_dP_outputs: [B, T, 1, H, W]
B, T, C, H, W = train_dP_outputs.shape

# Where to save logs/stats
after_log_path = os.path.join(data_dir_dP_train, "dP_train_min_max_after_normalization.txt")

# --- containers to STORE per-(sample,time) min/max for later de-normalization ---
# Save them as torch tensors
min_train_dP = torch.empty((B, T), dtype=train_dP_outputs.dtype, device=train_dP_outputs.device)
max_train_dP = torch.empty((B, T), dtype=train_dP_outputs.dtype, device=train_dP_outputs.device)

# --- normalize per sample & per time frame, and fill the stats containers ---
with open(after_log_path, "w") as f:
    f.write("Pressure Build-Up Min–Max AFTER Normalization (per sample, per time frame)\n")
    f.write("="*80 + "\n")

    for t in range(T):
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        for i in range(B):
            # slice [1,H,W]
            arr = train_dP_outputs[i, t]  # tensor view, no copy

            mn = arr.min()
            mx = arr.max()

            # store stats for later de-normalization
            min_train_dP[i, t] = mn
            max_train_dP[i, t] = mx

            # normalize to [-1, 1]
            norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
            train_dP_outputs[i, t] = norm

            # log post-normalization extrema (should be ~[-1, 1])
            f.write(f"Sample {i+1}: Min={norm.min().item():.4f}, Max={norm.max().item():.4f}\n")

print(f"[outputs] AFTER norm log saved -> {after_log_path}")

print("\nChecking global min/max for Pressure Build-Up after normalization:")

B, T, C, H, W = train_dP_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = train_dP_outputs[:, t].amin().item()
    global_max = train_dP_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Define variables for conditional diffusion model
train_dP_label = train_dP_inputs
train_dP_target = train_dP_outputs

print("END OF TRAINING DATASET PREPARATION\n")



# --------------------------------------------------
# Validation dataset preparation
# --------------------------------------------------
print("VALIDATION DATASET PREPARATION\n")

# Define validation dataset directory path
data_dir_dP_val = DP_VAE_ROOT / "validation_dataset_pressure_buildup"

# Define inputs (conditions) and outputs (targets) variables
val_dP_inputs = torch.load(f'{data_dir_dP_val}/dP_val_inputs.pt')
val_dP_outputs = torch.load(f'{data_dir_dP_val}/dP_val_outputs.pt')

print(val_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(val_dP_outputs.shape)

# For inputs and outputs we take the first 17 time frames
val_dP_inputs  = val_dP_inputs[:, :, :, :17]  
val_dP_outputs = val_dP_outputs[:, :, :, :17]
print(val_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(val_dP_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]

# Add channel dimension for val_dP_outputs 
val_dP_outputs = val_dP_outputs.unsqueeze(-1)
print(val_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(val_dP_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

# Prepare input shape to video diffusion model 
# Inputs: [B, H, W, T, C] → [B, T, C, H, W]
val_dP_inputs = val_dP_inputs.permute(0, 3, 4, 1, 2)
val_dP_outputs = val_dP_outputs.permute(0, 3, 4, 1, 2)

# Print new shape
print("New shape:", val_dP_inputs.shape)   # Expected: [num_samples, time_frames, num_features (channels), height, width] -> All conditions (8 time steps)
print("New shape:", val_dP_outputs.shape)  # Expected: [num_samples, time_frames, num_features (channels), height, width] -> Pressure Build-Up (8 time steps)


# Local Normalization for all 12 conditions 
# val_dP_inputs shape: [B, T, C, H, W]
B, T, Cin, H, W = val_dP_inputs.shape

# Compute min/max over (H,W) only → keep T
mins_inp = val_dP_inputs.amin(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]
maxs_inp = val_dP_inputs.amax(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]

# Define file path to save the MIN-MAX values BEFORE local normalization for the CONDITIONS
data_dir_dP_val = ensure_dir(JOINT_DP_ROOT / "validation_dataset_pressure_buildup")
save_path = os.path.join(data_dir_dP_val, "dP_val_conditions_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Condition Min–Max Values BEFORE Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")

    for c in range(Cin):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*50 + "\n")
            cmins = mins_inp[:, t, c, 0, 0]  # [B]
            cmaxs = maxs_inp[:, t, c, 0, 0]  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={cmins[i].item():.4f}, Max={cmaxs[i].item():.4f}\n")

print(f"[inputs] BEFORE norm log saved -> {save_path}")

# === Local min–max normalization per (sample, time, condition) over (H,W) ===
# val_dP_inputs: [B, T, C, H, W]
B, T, C, H, W = val_dP_inputs.shape

# Compute per-(B,T,C) min/max over (H,W)
mins_inp = val_dP_inputs.amin(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
maxs_inp = val_dP_inputs.amax(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
ranges   = maxs_inp - mins_inp

# Avoid divide-by-zero (constant fields): where range==0, keep original values
zero_mask = ranges == 0
ranges_safe = ranges.masked_fill(zero_mask, 1.0)

# Normalize to [-1, 1]
norm_inp = 2.0 * (val_dP_inputs - mins_inp) / ranges_safe - 1.0
# Put back original values where field is constant across (H,W)
norm_inp = torch.where(zero_mask, val_dP_inputs, norm_inp)

# Store back
val_dP_inputs = norm_inp

# =========== SAVE min/max AFTER normalization (per condition, per time, per sample) ===========
save_path = os.path.join(data_dir_dP_val, "dP_val_conditions_min_max_after_normalization.txt")
with open(save_path, "w") as f:
    f.write("Condition Min–Max Values AFTER Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")
    for c in range(C):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*60 + "\n")
            # Reduce over (H,W) *after* normalization to show the per-(B,T,C) extrema
            mins_post = val_dP_inputs[:, t, c].amin(dim=(1, 2))  # [B]
            maxs_post = val_dP_inputs[:, t, c].amax(dim=(1, 2))  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={mins_post[i].item():.4f}, Max={maxs_post[i].item():.4f}\n")

print(f"[inputs] AFTER norm log saved -> {save_path}")

print("\nChecking global min/max for each condition and time frame after normalization:")

B, T, C, H, W = val_dP_inputs.shape

for c in range(C):  # loop over conditions
    print(f"\n=== Condition {c+1} ===")
    for t in range(T):  # loop over time frames
        global_min = val_dP_inputs[:, t, c, :, :].amin().item()
        global_max = val_dP_inputs[:, t, c, :, :].amax().item()
        print(f"  Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")


# Local normalization for pressure build-up
# 1) Map to physical/ranged values
val_dP_outputs = dnorm_dP(val_dP_outputs)

# 2) Clamp negatives to zero (elementwise, across all samples/time frames)
val_dP_outputs = torch.clamp(val_dP_outputs, min=0)

# === Save min and max values for both conditional and unconditional generation ===
B, T, C, H, W = val_dP_outputs.shape
assert C == 1, f"Expected C=1, got {C}"

# Per-(sample,time) stats: [B,T]
per_bt_min = val_dP_outputs.amin(dim=(3, 4)).squeeze(2)  # [B,T]
per_bt_max = val_dP_outputs.amax(dim=(3, 4)).squeeze(2)  # [B,T]

# Global per-time stats across all samples/pixels: [T]
global_min = val_dP_outputs.amin(dim=(0, 2, 3, 4))       # [T]
global_max = val_dP_outputs.amax(dim=(0, 2, 3, 4))       # [T]

save_path = os.path.join(data_dir_dP_val, "dP_val_min_max.pt")
torch.save(
    {
        "per_bt_min": per_bt_min.cpu(),   # [B,T]
        "per_bt_max": per_bt_max.cpu(),   # [B,T]
        "global_min": global_min.cpu(),   # [T]
        "global_max": global_max.cpu(),   # [T]
    },
    save_path,
)
print(f"Saved stats -> {save_path}")
# === Save min and max values for both conditional and unconditional generation ===

save_path = os.path.join(data_dir_dP_val, "dP_val_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Pressure Build-Up Min–Max BEFORE Normalization\n")
    f.write("="*80 + "\n")

    for t in range(T):  # loop over 8 time frames
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        # Reduce over (C,H,W) for each sample
        mins = val_dP_outputs[:, t].amin(dim=(1, 2, 3))  # [B]
        maxs = val_dP_outputs[:, t].amax(dim=(1, 2, 3))  # [B]

        for i in range(B):
            f.write(f"Sample {i+1}: Min={mins[i].item():.4f}, Max={maxs[i].item():.4f}\n")

print(f"[outputs] BEFORE norm log saved -> {save_path}")

print("\nChecking global min/max for Pressure Build-Up before normalization:")

B, T, C, H, W = val_dP_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = val_dP_outputs[:, t].amin().item()
    global_max = val_dP_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# val_dP_outputs: [B, T, 1, H, W]
B, T, C, H, W = val_dP_outputs.shape

# Where to save logs/stats
after_log_path = os.path.join(data_dir_dP_val, "dP_val_min_max_after_normalization.txt")

# --- containers to STORE per-(sample,time) min/max for later de-normalization ---
# Save them as torch tensors
min_val_dP = torch.empty((B, T), dtype=val_dP_outputs.dtype, device=val_dP_outputs.device)
max_val_dP = torch.empty((B, T), dtype=val_dP_outputs.dtype, device=val_dP_outputs.device)

# --- normalize per sample & per time frame, and fill the stats containers ---
with open(after_log_path, "w") as f:
    f.write("Pressure Build-Up Min–Max AFTER Normalization (per sample, per time frame)\n")
    f.write("="*80 + "\n")

    for t in range(T):
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        for i in range(B):
            # slice [1,H,W]
            arr = val_dP_outputs[i, t]  # tensor view, no copy

            mn = arr.min()
            mx = arr.max()

            # store stats for later de-normalization
            min_val_dP[i, t] = mn
            max_val_dP[i, t] = mx

            # normalize to [-1, 1]
            norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
            val_dP_outputs[i, t] = norm

            # log post-normalization extrema (should be ~[-1, 1])
            f.write(f"Sample {i+1}: Min={norm.min().item():.4f}, Max={norm.max().item():.4f}\n")

print(f"[outputs] AFTER norm log saved -> {after_log_path}")

print("\nChecking global min/max for Pressure Build-Up after normalization:")

B, T, C, H, W = val_dP_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = val_dP_outputs[:, t].amin().item()
    global_max = val_dP_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Define variables for conditional diffusion model
val_dP_label = val_dP_inputs
val_dP_target = val_dP_outputs

print("END OF VALIDATION DATASET PREPARATION\n")


# --------------------------------------------------
# Test dataset preparation
# --------------------------------------------------
print("TEST DATASET PREPARATION\n")

# Define validation dataset directory path
data_dir_dP_test = DP_VAE_ROOT / "test_dataset_pressure_buildup"

# Define inputs (conditions) and outputs (targets) variables
test_dP_inputs = torch.load(f'{data_dir_dP_test}/dP_test_inputs.pt')
test_dP_outputs = torch.load(f'{data_dir_dP_test}/dP_test_outputs.pt')

print(test_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(test_dP_outputs.shape)
 
# For inputs and outputs we take the first 17 time frames
test_dP_inputs  = test_dP_inputs[:, :, :, :17]  
test_dP_outputs = test_dP_outputs[:, :, :, :17]
print(test_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(test_dP_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]

# Add channel dimension for test_dP_outputs 
test_dP_outputs = test_dP_outputs.unsqueeze(-1)
print(test_dP_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(test_dP_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

# Prepare input shape to video diffusion model 
# Inputs: [B, H, W, T, C] → [B, T, C, H, W]
test_dP_inputs = test_dP_inputs.permute(0, 3, 4, 1, 2)
test_dP_outputs = test_dP_outputs.permute(0, 3, 4, 1, 2)

# Print new shape
print("New shape:", test_dP_inputs.shape)   # Expected: [num_samples, time_frames, num_features (channels), height, width] -> All conditions (8 time steps)
print("New shape:", test_dP_outputs.shape)  # Expected: [num_samples, time_frames, num_features (channels), height, width] -> Pressure Build-Up (8 time steps)

## Local normalization for all 12 conditions 
# test_dP_inputs shape: [B, T, C, H, W]
B, T, Cin, H, W = test_dP_inputs.shape

# Compute min/max over (H,W) only → keep T
mins_inp = test_dP_inputs.amin(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]
maxs_inp = test_dP_inputs.amax(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]

# Define file path to save the MIN-MAX values BEFORE local normalization for the CONDITIONS
data_dir_dP_test = ensure_dir(JOINT_DP_ROOT / "test_dataset_pressure_buildup")
save_path = os.path.join(data_dir_dP_test, "dP_test_conditions_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Condition Min–Max Values BEFORE Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")

    for c in range(Cin):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*50 + "\n")
            cmins = mins_inp[:, t, c, 0, 0]  # [B]
            cmaxs = maxs_inp[:, t, c, 0, 0]  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={cmins[i].item():.4f}, Max={cmaxs[i].item():.4f}\n")

print(f"[inputs] BEFORE norm log saved -> {save_path}")

# === Local min–max normalization per (sample, time, condition) over (H,W) ===
# test_dP_inputs: [B, T, C, H, W]
B, T, C, H, W = test_dP_inputs.shape

# Compute per-(B,T,C) min/max over (H,W)
mins_inp = test_dP_inputs.amin(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
maxs_inp = test_dP_inputs.amax(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
ranges   = maxs_inp - mins_inp

# Avoid divide-by-zero (constant fields): where range==0, keep original values
zero_mask = ranges == 0
ranges_safe = ranges.masked_fill(zero_mask, 1.0)

# Normalize to [-1, 1]
norm_inp = 2.0 * (test_dP_inputs - mins_inp) / ranges_safe - 1.0
# Put back original values where field is constant across (H,W)
norm_inp = torch.where(zero_mask, test_dP_inputs, norm_inp)

# Store back
test_dP_inputs = norm_inp

# =========== SAVE min/max AFTER normalization (per condition, per time, per sample) ===========
save_path = os.path.join(data_dir_dP_test, "dP_test_conditions_min_max_after_normalization.txt")
with open(save_path, "w") as f:
    f.write("Condition Min–Max Values AFTER Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")
    for c in range(C):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*60 + "\n")
            # Reduce over (H,W) *after* normalization to show the per-(B,T,C) extrema
            mins_post = test_dP_inputs[:, t, c].amin(dim=(1, 2))  # [B]
            maxs_post = test_dP_inputs[:, t, c].amax(dim=(1, 2))  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={mins_post[i].item():.4f}, Max={maxs_post[i].item():.4f}\n")

print(f"[inputs] AFTER norm log saved -> {save_path}")

print("\nChecking global min/max for each condition and time frame after normalization:")

B, T, C, H, W = test_dP_inputs.shape

for c in range(C):  # loop over conditions
    print(f"\n=== Condition {c+1} ===")
    for t in range(T):  # loop over time frames
        global_min = test_dP_inputs[:, t, c, :, :].amin().item()
        global_max = test_dP_inputs[:, t, c, :, :].amax().item()
        print(f"  Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Local normalization for pressure build-up 

## FOR THE TEST DATASET WE DO NOT APPLY THE dnorm_dP FUNCTION

# Apply dnorm_dP function to ensure proper Pressure Build-Up range
#test_dP_outputs=dnorm_dP(test_dP_outputs)

# Check Min and Max values 
#min_test_dP = test_dP_outputs.min()
#max_test_dP = test_dP_outputs.max()
#print(min_test_dP) 
#print(max_test_dP)

# Some dP values are negative. Replace negative values with zeros
test_dP_outputs = torch.clamp(test_dP_outputs, min=0)

# === Save min and max values for both conditional and unconditional generation ===
B, T, C, H, W = test_dP_outputs.shape
assert C == 1, f"Expected C=1, got {C}"

# Per-(sample,time) stats: [B,T]
per_bt_min = test_dP_outputs.amin(dim=(3, 4)).squeeze(2)  # [B,T]
per_bt_max = test_dP_outputs.amax(dim=(3, 4)).squeeze(2)  # [B,T]

# Global per-time stats across all samples/pixels: [T]
global_min = test_dP_outputs.amin(dim=(0, 2, 3, 4))       # [T]
global_max = test_dP_outputs.amax(dim=(0, 2, 3, 4))       # [T]

save_path = os.path.join(data_dir_dP_test, "dP_test_min_max.pt")
torch.save(
    {
        "per_bt_min": per_bt_min.cpu(),   # [B,T]
        "per_bt_max": per_bt_max.cpu(),   # [B,T]
        "global_min": global_min.cpu(),   # [T]
        "global_max": global_max.cpu(),   # [T]
    },
    save_path,
)
print(f"Saved stats -> {save_path}")
# === Save min and max values for both conditional and unconditional generation ===

save_path = os.path.join(data_dir_dP_test, "dP_test_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Pressure Build-Up Min–Max BEFORE Normalization\n")
    f.write("="*80 + "\n")

    for t in range(T):  # loop over 8 time frames
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        # Reduce over (C,H,W) for each sample
        mins = test_dP_outputs[:, t].amin(dim=(1, 2, 3))  # [B]
        maxs = test_dP_outputs[:, t].amax(dim=(1, 2, 3))  # [B]

        for i in range(B):
            f.write(f"Sample {i+1}: Min={mins[i].item():.4f}, Max={maxs[i].item():.4f}\n")

print(f"[outputs] BEFORE norm log saved -> {save_path}")

print("\nChecking global min/max for Pressure Build-Up before normalization:")

B, T, C, H, W = test_dP_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = test_dP_outputs[:, t].amin().item()
    global_max = test_dP_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# test_dP_outputs: [B, T, 1, H, W]
B, T, C, H, W = test_dP_outputs.shape

# Where to save logs/stats
after_log_path = os.path.join(data_dir_dP_test, "dP_test_min_max_after_normalization.txt")

# --- containers to STORE per-(sample,time) min/max for later de-normalization ---
# Save them as torch tensors
min_test_dP = torch.empty((B, T), dtype=test_dP_outputs.dtype, device=test_dP_outputs.device)
max_test_dP = torch.empty((B, T), dtype=test_dP_outputs.dtype, device=test_dP_outputs.device)

# --- normalize per sample & per time frame, and fill the stats containers ---
with open(after_log_path, "w") as f:
    f.write("Pressure Build-Up Min–Max AFTER Normalization (per sample, per time frame)\n")
    f.write("="*80 + "\n")

    for t in range(T):
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        for i in range(B):
            # slice [1,H,W]
            arr = test_dP_outputs[i, t]  # tensor view, no copy

            mn = arr.min()
            mx = arr.max()

            # store stats for later de-normalization
            min_test_dP[i, t] = mn
            max_test_dP[i, t] = mx

            # normalize to [-1, 1]
            norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
            test_dP_outputs[i, t] = norm

            # log post-normalization extrema (should be ~[-1, 1])
            f.write(f"Sample {i+1}: Min={norm.min().item():.4f}, Max={norm.max().item():.4f}\n")

print(f"[outputs] AFTER norm log saved -> {after_log_path}")

print("\nChecking global min/max for Pressure Build-Up after normalization:")

B, T, C, H, W = test_dP_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = test_dP_outputs[:, t].amin().item()
    global_max = test_dP_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Define variables for conditional diffusion model
test_dP_label = test_dP_inputs
test_dP_target = test_dP_outputs

print("END OF TEST DATASET PREPARATION\n")

torch.save(train_dP_target, JOINT_DP_ROOT / "train_dP_target.pt")
torch.save(val_dP_target,   JOINT_DP_ROOT / "val_dP_target.pt")
torch.save(test_dP_target,  JOINT_DP_ROOT / "test_dP_target.pt")

print(train_dP_target.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(val_dP_target.shape)   # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(test_dP_target.shape)  # Shape: [num_samples, height, width, time_steps, num_features (channels)]

print("\nPressure Build-Up Dataset Prepared!")