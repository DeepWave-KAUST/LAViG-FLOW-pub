###############################################################################
# CO₂ Gas Saturation Dataset Preparation (2025)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description:
#   Standalone preprocessing script that loads the raw CO₂ gas saturation
#   tensors, trims the temporal window, permutes them into [B, T, C, H, W],
#   performs per-sample local min–max normalization, and writes diagnostic
#   statistics for training/validation/test splits.
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import os
import torch
from paths import CO2_DATA_ROOT, JOINT_CO2_ROOT as _JOINT_CO2_ROOT

VQVAE_CO2_ROOT = CO2_DATA_ROOT
JOINT_CO2_ROOT = _JOINT_CO2_ROOT
JOINT_CO2_ROOT.mkdir(parents=True, exist_ok=True)

def ensure_dir(path):
    """Create the directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

# --------------------------------------------------
# Training dataset preparation
# --------------------------------------------------

print("TRAINING DATASET PREPARATION\n")

# Define training dataset directory path
data_dir_sg_train = VQVAE_CO2_ROOT / "training_dataset_gas_saturation"

# Define inputs (conditions) and outputs (targets) variables
train_sg_inputs = torch.load(f"{data_dir_sg_train}/sg_train_inputs.pt")
train_sg_outputs = torch.load(f"{data_dir_sg_train}/sg_train_outputs.pt")

print(train_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(train_sg_outputs.shape)

# For inputs and outputs we take the first 17 time frames
train_sg_inputs  = train_sg_inputs[:, :, :, :17]  
train_sg_outputs = train_sg_outputs[:, :, :, :17]
print(train_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(train_sg_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]

# Add channel dimension for train_sg_outputs 
train_sg_outputs = train_sg_outputs.unsqueeze(-1)
print(train_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(train_sg_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

# Prepare input shape to video diffusion model 
# Inputs: [B, H, W, T, C] → [B, T, C, H, W]
train_sg_inputs = train_sg_inputs.permute(0, 3, 4, 1, 2)
train_sg_outputs = train_sg_outputs.permute(0, 3, 4, 1, 2)

# Print new shape
print("New shape:", train_sg_inputs.shape)   # Expected: [num_samples, time_frames, num_features (channels), height, width] -> All conditions (8 time steps)
print("New shape:", train_sg_outputs.shape)  # Expected: [num_samples, time_frames, num_features (channels), height, width] -> Pressure Build-Up (8 time steps)

# Local normalization for all 12 conditions 
# train_dP_inputs shape: [B, T, C, H, W]
B, T, Cin, H, W = train_sg_inputs.shape

# Compute min/max over (H,W) only → keep T
mins_inp = train_sg_inputs.amin(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]
maxs_inp = train_sg_inputs.amax(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]

# Define file path to save the MIN-MAX values BEFORE local normalization for the CONDITIONS
data_dir_sg_train = ensure_dir(JOINT_CO2_ROOT / "training_dataset_gas_saturation")
save_path = os.path.join(
    data_dir_sg_train, "sg_train_conditions_min_max_before_normalization.txt"
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
B, T, C, H, W = train_sg_inputs.shape

# Compute per-(B,T,C) min/max over (H,W)
mins_inp = train_sg_inputs.amin(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
maxs_inp = train_sg_inputs.amax(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
ranges   = maxs_inp - mins_inp

# Avoid divide-by-zero (constant fields): where range==0, keep original values
zero_mask = ranges == 0
ranges_safe = ranges.masked_fill(zero_mask, 1.0)

# Normalize to [-1, 1]
norm_inp = 2.0 * (train_sg_inputs - mins_inp) / ranges_safe - 1.0
# Put back original values where field is constant across (H,W)
norm_inp = torch.where(zero_mask, train_sg_inputs, norm_inp)

# Store back
train_sg_inputs = norm_inp

# =========== SAVE min/max AFTER normalization (per condition, per time, per sample) ===========
save_path = os.path.join(data_dir_sg_train, "sg_train_conditions_min_max_after_normalization.txt")
with open(save_path, "w") as f:
    f.write("Condition Min–Max Values AFTER Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")
    for c in range(C):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*60 + "\n")
            # Reduce over (H,W) *after* normalization to show the per-(B,T,C) extrema
            mins_post = train_sg_inputs[:, t, c].amin(dim=(1, 2))  # [B]
            maxs_post = train_sg_inputs[:, t, c].amax(dim=(1, 2))  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={mins_post[i].item():.4f}, Max={maxs_post[i].item():.4f}\n")

print(f"[inputs] AFTER norm log saved -> {save_path}")

print("\nChecking global min/max for each condition and time frame after normalization:")

B, T, C, H, W = train_sg_inputs.shape

for c in range(C):  # loop over conditions
    print(f"\n=== Condition {c+1} ===")
    for t in range(T):  # loop over time frames
        global_min = train_sg_inputs[:, t, c, :, :].amin().item()
        global_max = train_sg_inputs[:, t, c, :, :].amax().item()
        print(f"  Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# === Save min and max values for both conditional and unconditional generation ===
B, T, C, H, W = train_sg_outputs.shape
assert C == 1, f"Expected C=1, got {C}"

# Per-(sample,time) stats: [B,T]
per_bt_min = train_sg_outputs.amin(dim=(3, 4)).squeeze(2)  # [B,T]
per_bt_max = train_sg_outputs.amax(dim=(3, 4)).squeeze(2)  # [B,T]

# Global per-time stats across all samples/pixels: [T]
global_min = train_sg_outputs.amin(dim=(0, 2, 3, 4))       # [T]
global_max = train_sg_outputs.amax(dim=(0, 2, 3, 4))       # [T]

save_path = os.path.join(data_dir_sg_train, "sg_train_min_max.pt")
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

save_path = os.path.join(data_dir_sg_train, "sg_train_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Gas Saturation Min–Max BEFORE Normalization\n")
    f.write("="*80 + "\n")

    for t in range(T):  # loop over 8 time frames
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        # Reduce over (C,H,W) for each sample
        mins = train_sg_outputs[:, t].amin(dim=(1, 2, 3))  # [B]
        maxs = train_sg_outputs[:, t].amax(dim=(1, 2, 3))  # [B]

        for i in range(B):
            f.write(f"Sample {i+1}: Min={mins[i].item():.4f}, Max={maxs[i].item():.4f}\n")

print(f"[outputs] BEFORE norm log saved -> {save_path}")

print("\nChecking global min/max for Gas Saturation before normalization:")

B, T, C, H, W = train_sg_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = train_sg_outputs[:, t].amin().item()
    global_max = train_sg_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# train_dP_outputs: [B, T, 1, H, W]
B, T, C, H, W = train_sg_outputs.shape

# Where to save logs/stats
after_log_path = os.path.join(data_dir_sg_train, "sg_train_min_max_after_normalization.txt")

# --- containers to STORE per-(sample,time) min/max for later de-normalization ---
# Save them as torch tensors
min_train_sg = torch.empty((B, T), dtype=train_sg_outputs.dtype, device=train_sg_outputs.device)
max_train_sg = torch.empty((B, T), dtype=train_sg_outputs.dtype, device=train_sg_outputs.device)

# --- normalize per sample & per time frame, and fill the stats containers ---
with open(after_log_path, "w") as f:
    f.write("Gas Saturation Min–Max AFTER Normalization (per sample, per time frame)\n")
    f.write("="*80 + "\n")

    for t in range(T):
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        for i in range(B):
            # slice [1,H,W]
            arr = train_sg_outputs[i, t]  # tensor view, no copy

            mn = arr.min()
            mx = arr.max()

            # store stats for later de-normalization
            min_train_sg[i, t] = mn
            max_train_sg[i, t] = mx

            # normalize to [-1, 1]
            norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
            train_sg_outputs[i, t] = norm

            # log post-normalization extrema (should be ~[-1, 1])
            f.write(f"Sample {i+1}: Min={norm.min().item():.4f}, Max={norm.max().item():.4f}\n")

print(f"[outputs] AFTER norm log saved -> {after_log_path}")

print("\nChecking global min/max for Gas Saturation after normalization:")

B, T, C, H, W = train_sg_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = train_sg_outputs[:, t].amin().item()
    global_max = train_sg_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Define variables for conditional diffusion model
train_sg_label = train_sg_inputs
train_sg_target = train_sg_outputs

print("END OF TRAINING DATASET PREPARATION\n")



# --------------------------------------------------
# Validation dataset preparation
# --------------------------------------------------
print("VALIDATION DATASET PREPARATION\n")

# Define validation dataset directory path
data_dir_sg_val = VQVAE_CO2_ROOT / "validation_dataset_gas_saturation"

# Define inputs (conditions) and outputs (targets) variables
val_sg_inputs = torch.load(f'{data_dir_sg_val}/sg_val_inputs.pt')
val_sg_outputs = torch.load(f'{data_dir_sg_val}/sg_val_outputs.pt')

print(val_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(val_sg_outputs.shape)

# For inputs and outputs we take the first 17 time frames
val_sg_inputs  = val_sg_inputs[:, :, :, :17]  
val_sg_outputs = val_sg_outputs[:, :, :, :17]
print(val_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(val_sg_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]

# Add channel dimension for val_dP_outputs 
val_sg_outputs = val_sg_outputs.unsqueeze(-1)
print(val_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(val_sg_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

# Prepare input shape to video diffusion model 
# Inputs: [B, H, W, T, C] → [B, T, C, H, W]
val_sg_inputs = val_sg_inputs.permute(0, 3, 4, 1, 2)
val_sg_outputs = val_sg_outputs.permute(0, 3, 4, 1, 2)

# Print new shape
print("New shape:", val_sg_inputs.shape)   # Expected: [num_samples, time_frames, num_features (channels), height, width] -> All conditions (8 time steps)
print("New shape:", val_sg_outputs.shape)  # Expected: [num_samples, time_frames, num_features (channels), height, width] -> Pressure Build-Up (8 time steps)


# Local Normalization for all 12 conditions 
# val_dP_inputs shape: [B, T, C, H, W]
B, T, Cin, H, W = val_sg_inputs.shape

# Compute min/max over (H,W) only → keep T
mins_inp = val_sg_inputs.amin(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]
maxs_inp = val_sg_inputs.amax(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]

# Define file path to save the MIN-MAX values BEFORE local normalization for the CONDITIONS
data_dir_sg_val = ensure_dir(JOINT_CO2_ROOT / "validation_dataset_gas_saturation")
save_path = os.path.join(data_dir_sg_val, "sg_val_conditions_min_max_before_normalization.txt")

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
B, T, C, H, W = val_sg_inputs.shape

# Compute per-(B,T,C) min/max over (H,W)
mins_inp = val_sg_inputs.amin(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
maxs_inp = val_sg_inputs.amax(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
ranges   = maxs_inp - mins_inp

# Avoid divide-by-zero (constant fields): where range==0, keep original values
zero_mask = ranges == 0
ranges_safe = ranges.masked_fill(zero_mask, 1.0)

# Normalize to [-1, 1]
norm_inp = 2.0 * (val_sg_inputs - mins_inp) / ranges_safe - 1.0
# Put back original values where field is constant across (H,W)
norm_inp = torch.where(zero_mask, val_sg_inputs, norm_inp)

# Store back
val_sg_inputs = norm_inp

# =========== SAVE min/max AFTER normalization (per condition, per time, per sample) ===========
save_path = os.path.join(data_dir_sg_val, "sg_val_conditions_min_max_after_normalization.txt")
with open(save_path, "w") as f:
    f.write("Condition Min–Max Values AFTER Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")
    for c in range(C):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*60 + "\n")
            # Reduce over (H,W) *after* normalization to show the per-(B,T,C) extrema
            mins_post = val_sg_inputs[:, t, c].amin(dim=(1, 2))  # [B]
            maxs_post = val_sg_inputs[:, t, c].amax(dim=(1, 2))  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={mins_post[i].item():.4f}, Max={maxs_post[i].item():.4f}\n")

print(f"[inputs] AFTER norm log saved -> {save_path}")

print("\nChecking global min/max for each condition and time frame after normalization:")

B, T, C, H, W = val_sg_inputs.shape

for c in range(C):  # loop over conditions
    print(f"\n=== Condition {c+1} ===")
    for t in range(T):  # loop over time frames
        global_min = val_sg_inputs[:, t, c, :, :].amin().item()
        global_max = val_sg_inputs[:, t, c, :, :].amax().item()
        print(f"  Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# === Save min and max values for both conditional and unconditional generation ===
B, T, C, H, W = val_sg_outputs.shape
assert C == 1, f"Expected C=1, got {C}"

# Per-(sample,time) stats: [B,T]
per_bt_min = val_sg_outputs.amin(dim=(3, 4)).squeeze(2)  # [B,T]
per_bt_max = val_sg_outputs.amax(dim=(3, 4)).squeeze(2)  # [B,T]

# Global per-time stats across all samples/pixels: [T]
global_min = val_sg_outputs.amin(dim=(0, 2, 3, 4))       # [T]
global_max = val_sg_outputs.amax(dim=(0, 2, 3, 4))       # [T]

save_path = os.path.join(data_dir_sg_val, "sg_val_min_max.pt")
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

save_path = os.path.join(data_dir_sg_val, "sg_val_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Gas Saturation Min–Max BEFORE Normalization\n")
    f.write("="*80 + "\n")

    for t in range(T):  # loop over 8 time frames
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        # Reduce over (C,H,W) for each sample
        mins = val_sg_outputs[:, t].amin(dim=(1, 2, 3))  # [B]
        maxs = val_sg_outputs[:, t].amax(dim=(1, 2, 3))  # [B]

        for i in range(B):
            f.write(f"Sample {i+1}: Min={mins[i].item():.4f}, Max={maxs[i].item():.4f}\n")

print(f"[outputs] BEFORE norm log saved -> {save_path}")

print("\nChecking global min/max for Gas Saturation before normalization:")

B, T, C, H, W = val_sg_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = val_sg_outputs[:, t].amin().item()
    global_max = val_sg_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# val_dP_outputs: [B, T, 1, H, W]
B, T, C, H, W = val_sg_outputs.shape

# Where to save logs/stats
after_log_path = os.path.join(data_dir_sg_val, "sg_val_min_max_after_normalization.txt")

# --- containers to STORE per-(sample,time) min/max for later de-normalization ---
# Save them as torch tensors
min_val_sg = torch.empty((B, T), dtype=val_sg_outputs.dtype, device=val_sg_outputs.device)
max_val_sg = torch.empty((B, T), dtype=val_sg_outputs.dtype, device=val_sg_outputs.device)

# --- normalize per sample & per time frame, and fill the stats containers ---
with open(after_log_path, "w") as f:
    f.write("Gas Saturation Min–Max AFTER Normalization (per sample, per time frame)\n")
    f.write("="*80 + "\n")

    for t in range(T):
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        for i in range(B):
            # slice [1,H,W]
            arr = val_sg_outputs[i, t]  # tensor view, no copy

            mn = arr.min()
            mx = arr.max()

            # store stats for later de-normalization
            min_val_sg[i, t] = mn
            max_val_sg[i, t] = mx

            # normalize to [-1, 1]
            norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
            val_sg_outputs[i, t] = norm

            # log post-normalization extrema (should be ~[-1, 1])
            f.write(f"Sample {i+1}: Min={norm.min().item():.4f}, Max={norm.max().item():.4f}\n")

print(f"[outputs] AFTER norm log saved -> {after_log_path}")

print("\nChecking global min/max for Gas Saturation after normalization:")

B, T, C, H, W = val_sg_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = val_sg_outputs[:, t].amin().item()
    global_max = val_sg_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Define variables for conditional diffusion model
val_sg_label = val_sg_inputs
val_sg_target = val_sg_outputs

# --------------------------------------------------
# Test dataset preparation
# --------------------------------------------------
print("END OF VALIDATION DATASET PREPARATION\n")

print("TEST DATASET PREPARATION\n")

# Define validation dataset directory path
data_dir_sg_test = VQVAE_CO2_ROOT / "test_dataset_gas_saturation"

# Define inputs (conditions) and outputs (targets) variables
test_sg_inputs = torch.load(f'{data_dir_sg_test}/sg_test_inputs.pt')
test_sg_outputs = torch.load(f'{data_dir_sg_test}/sg_test_outputs.pt')

print(test_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(test_sg_outputs.shape)
 
# For inputs and outputs we take the first 17 time frames
test_sg_inputs  = test_sg_inputs[:, :, :, :17]  
test_sg_outputs = test_sg_outputs[:, :, :, :17]
print(test_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]
print(test_sg_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features]

# Add channel dimension for test_dP_outputs 
test_sg_outputs = test_sg_outputs.unsqueeze(-1)
print(test_sg_inputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(test_sg_outputs.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

# Prepare input shape to video diffusion model 
# Inputs: [B, H, W, T, C] → [B, T, C, H, W]
test_sg_inputs = test_sg_inputs.permute(0, 3, 4, 1, 2)
test_sg_outputs = test_sg_outputs.permute(0, 3, 4, 1, 2)

# Print new shape
print("New shape:", test_sg_inputs.shape)   # Expected: [num_samples, time_frames, num_features (channels), height, width] -> All conditions (8 time steps)
print("New shape:", test_sg_outputs.shape)  # Expected: [num_samples, time_frames, num_features (channels), height, width] -> Pressure Build-Up (8 time steps)

## Local normalization for all 12 conditions 
# test_dP_inputs shape: [B, T, C, H, W]
B, T, Cin, H, W = test_sg_inputs.shape

# Compute min/max over (H,W) only → keep T
mins_inp = test_sg_inputs.amin(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]
maxs_inp = test_sg_inputs.amax(dim=(3,4), keepdim=True)  # [B, T, C, 1, 1]

# Define file path to save the MIN-MAX values BEFORE local normalization for the CONDITIONS
data_dir_sg_test = ensure_dir(JOINT_CO2_ROOT / "test_dataset_gas_saturation")
save_path = os.path.join(data_dir_sg_test, "sg_test_conditions_min_max_before_normalization.txt")

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
B, T, C, H, W = test_sg_inputs.shape

# Compute per-(B,T,C) min/max over (H,W)
mins_inp = test_sg_inputs.amin(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
maxs_inp = test_sg_inputs.amax(dim=(3, 4), keepdim=True)   # [B, T, C, 1, 1]
ranges   = maxs_inp - mins_inp

# Avoid divide-by-zero (constant fields): where range==0, keep original values
zero_mask = ranges == 0
ranges_safe = ranges.masked_fill(zero_mask, 1.0)

# Normalize to [-1, 1]
norm_inp = 2.0 * (test_sg_inputs - mins_inp) / ranges_safe - 1.0
# Put back original values where field is constant across (H,W)
norm_inp = torch.where(zero_mask, test_sg_inputs, norm_inp)

# Store back
test_sg_inputs = norm_inp

# =========== SAVE min/max AFTER normalization (per condition, per time, per sample) ===========
save_path = os.path.join(data_dir_sg_test, "sg_test_conditions_min_max_after_normalization.txt")
with open(save_path, "w") as f:
    f.write("Condition Min–Max Values AFTER Local Normalization (per sample, per condition, per time frame)\n")
    f.write("="*100 + "\n")
    for c in range(C):
        f.write(f"\n=== Condition {c+1} ===\n")
        for t in range(T):
            f.write(f"\n  Time frame {t+1}:\n" + "-"*60 + "\n")
            # Reduce over (H,W) *after* normalization to show the per-(B,T,C) extrema
            mins_post = test_sg_inputs[:, t, c].amin(dim=(1, 2))  # [B]
            maxs_post = test_sg_inputs[:, t, c].amax(dim=(1, 2))  # [B]
            for i in range(B):
                f.write(f"Sample {i+1}: Min={mins_post[i].item():.4f}, Max={maxs_post[i].item():.4f}\n")

print(f"[inputs] AFTER norm log saved -> {save_path}")

print("\nChecking global min/max for each condition and time frame after normalization:")

B, T, C, H, W = test_sg_inputs.shape

for c in range(C):  # loop over conditions
    print(f"\n=== Condition {c+1} ===")
    for t in range(T):  # loop over time frames
        global_min = test_sg_inputs[:, t, c, :, :].amin().item()
        global_max = test_sg_inputs[:, t, c, :, :].amax().item()
        print(f"  Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Local normalization for pressure build-up 

# === Save min and max values for both conditional and unconditional generation ===
B, T, C, H, W = test_sg_outputs.shape
assert C == 1, f"Expected C=1, got {C}"

# Per-(sample,time) stats: [B,T]
per_bt_min = test_sg_outputs.amin(dim=(3, 4)).squeeze(2)  # [B,T]
per_bt_max = test_sg_outputs.amax(dim=(3, 4)).squeeze(2)  # [B,T]

# Global per-time stats across all samples/pixels: [T]
global_min = test_sg_outputs.amin(dim=(0, 2, 3, 4))       # [T]
global_max = test_sg_outputs.amax(dim=(0, 2, 3, 4))       # [T]

save_path = os.path.join(data_dir_sg_test, "sg_test_min_max.pt")
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

save_path = os.path.join(data_dir_sg_test, "sg_test_min_max_before_normalization.txt")

with open(save_path, "w") as f:
    f.write("Gas Saturation Min–Max BEFORE Normalization\n")
    f.write("="*80 + "\n")

    for t in range(T):  # loop over 8 time frames
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        # Reduce over (C,H,W) for each sample
        mins = test_sg_outputs[:, t].amin(dim=(1, 2, 3))  # [B]
        maxs = test_sg_outputs[:, t].amax(dim=(1, 2, 3))  # [B]

        for i in range(B):
            f.write(f"Sample {i+1}: Min={mins[i].item():.4f}, Max={maxs[i].item():.4f}\n")

print(f"[outputs] BEFORE norm log saved -> {save_path}")

print("\nChecking global min/max for Gas Saturation before normalization:")

B, T, C, H, W = test_sg_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = test_sg_outputs[:, t].amin().item()
    global_max = test_sg_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# test_dP_outputs: [B, T, 1, H, W]
B, T, C, H, W = test_sg_outputs.shape

# Where to save logs/stats
after_log_path = os.path.join(data_dir_sg_test, "sg_test_min_max_after_normalization.txt")

# --- containers to STORE per-(sample,time) min/max for later de-normalization ---
# Save them as torch tensors
min_test_sg = torch.empty((B, T), dtype=test_sg_outputs.dtype, device=test_sg_outputs.device)
max_test_sg = torch.empty((B, T), dtype=test_sg_outputs.dtype, device=test_sg_outputs.device)

# --- normalize per sample & per time frame, and fill the stats containers ---
with open(after_log_path, "w") as f:
    f.write("Gas Saturation Min–Max AFTER Normalization (per sample, per time frame)\n")
    f.write("="*80 + "\n")

    for t in range(T):
        f.write(f"\n=== Time frame {t+1} ===\n")
        f.write("-"*40 + "\n")

        for i in range(B):
            # slice [1,H,W]
            arr = test_sg_outputs[i, t]  # tensor view, no copy

            mn = arr.min()
            mx = arr.max()

            # store stats for later de-normalization
            min_test_sg[i, t] = mn
            max_test_sg[i, t] = mx

            # normalize to [-1, 1]
            norm = 2.0 * (arr - mn) / (mx - mn) - 1.0
            test_sg_outputs[i, t] = norm

            # log post-normalization extrema (should be ~[-1, 1])
            f.write(f"Sample {i+1}: Min={norm.min().item():.4f}, Max={norm.max().item():.4f}\n")

print(f"[outputs] AFTER norm log saved -> {after_log_path}")

print("\nChecking global min/max for Gas Saturation after normalization:")

B, T, C, H, W = test_sg_outputs.shape  # [4500, 8, 1, 96, 200]

for t in range(T):  # loop over time frames
    global_min = test_sg_outputs[:, t].amin().item()
    global_max = test_sg_outputs[:, t].amax().item()
    print(f"Time Frame {t+1}: Global Min = {global_min:.4f}, Global Max = {global_max:.4f}")

# Define variables for conditional diffusion model
test_sg_label = test_sg_inputs
test_sg_target = test_sg_outputs

print("END OF TEST DATASET PREPARATION\n")

torch.save(train_sg_target, JOINT_CO2_ROOT / "train_sg_target.pt")
torch.save(val_sg_target,   JOINT_CO2_ROOT / "val_sg_target.pt")
torch.save(test_sg_target,  JOINT_CO2_ROOT / "test_sg_target.pt")

print(train_sg_target.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(val_sg_target.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]
print(test_sg_target.shape) # Shape: [num_samples, height, width, time_steps, num_features (channels)]

print("\nGas Saturation Dataset Prepared!")