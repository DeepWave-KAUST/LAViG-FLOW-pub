# Pressure Build-Up VAE

Minimal guide for preparing the Pressure Build-Up dataset, training the VAE, and exporting reconstructions.

### CO₂ Dataset Info
- Official CO₂ Geological Sequestration dataset: [download link](https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing).

### 1. Environment
- Create/activate your env (example): `conda env create -f ../environment.yml && conda activate lavig-flow`

#### 1.1 Paths
- Before running any script, open `paths.py` at the repo root and replace `Path("/...")` with the absolute parent directory where `LAViG-FLOW` lives (e.g. `Path("/scratch/user")`). All other `"/.../"` placeholders in this README or in the SLURM templates should point to paths under that same root.

### 2. Prepare Dataset

#### 2.1 Collect Raw Tensors
- Before running the preparation script, copy or symlink the upstream `.pt` tensors into the expected layout. Rename the raw files (`*_a.pt`, `*_u.pt`) to the names below and drop them into the matching folders (create them if they do not exist).
- Pressure Build-Up:
  - Training (`n = 4,500`):
    - `dP_train_a.pt -> /.../LAViG-FLOW/pressure_buildup_vae/dP_data/training_dataset_pressure_buildup/dP_train_inputs.pt`
    - `dP_train_u.pt -> /.../LAViG-FLOW/pressure_buildup_vae/dP_data/training_dataset_pressure_buildup/dP_train_outputs.pt`
  - Validation (`n = 500`):
    - `dP_val_a.pt -> /.../LAViG-FLOW/pressure_buildup_vae/dP_data/validation_dataset_pressure_buildup/dP_val_inputs.pt`
    - `dP_val_u.pt -> /.../LAViG-FLOW/pressure_buildup_vae/dP_data/validation_dataset_pressure_buildup/dP_val_outputs.pt`
  - Test (`n = 500`):
    - `dP_test_a.pt -> /.../LAViG-FLOW/pressure_buildup_vae/dP_data/test_dataset_pressure_buildup/dP_test_inputs.pt`
    - `dP_test_u.pt -> /.../LAViG-FLOW/pressure_buildup_vae/dP_data/test_dataset_pressure_buildup/dP_test_outputs.pt`
- Run the script once to emit normalised tensors and statistics:
  ```
  python dP_dataset_preparation.py
  ```
- Expected outputs: `dP_data/{train,val,test}_sg_target.pt`

### 3. Train the VAE Model
- Update `dP.yaml` if you need to point to custom dataset paths or change hyperparameters.
- Launch training (Accelerate handles single/multi-GPU automatically):
  ```
  python training_vae.py --config dP.yaml
  ```
- Optional flags:
  - `--override '{"training.num_epochs": 300}'` injects quick JSON overrides.
  - `--resume path/to/checkpoint.pt` restarts from a checkpoint.

#### 3.1 Pretrained Checkpoints (Optional)
- If you only need inference, download the pre-trained assets [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-) instead of training from scratch.
- Create `/.../LAViG-FLOW/pressure_buildup_vae/trained_models/` at the project root if it is missing, then drop the files below:
  - `/.../LAViG-FLOW/pressure_buildup_vae/trained_models/vae_model.pth` → VAE checkpoint consumed by the reconstruction scripts.
- After placing these files you can jump straight to the reconstruction step.

### 4. Evaluate Reconstructions
- Every reconstruction entry point expects both the YAML config (to rebuild the model/dataset transforms) and a checkpoint path. Tweak the literals as needed:
  ```
  python validation_dataset_reconstructions.py \
    --config dP.yaml \
    --checkpoint trained_models/vae_model.pth \
    --output-dir results/reconstructions/validation \
    --sample-offset 0 \
    --max-samples 8500 \
    --mosaic-mode clip \
    --mosaic-count 100 \
    --mosaic-cols 17 \
    --device cuda
  ```
- Swap the script name for the other splits (`training_dataset_reconstructions.py`, `test_dataset_reconstructions.py`), or invoke `dataset_reconstruction.run_cli` and pass `--split train|validation|test`. All flags can be overridden inline the same way.

### 5. Tips
- Set a seed for reproducibility with the YAML key `training.seed`.
- Keep an eye on TensorBoard logs written to `/.../LAViG-FLOW/pressure_buildup_vae/results/tensorboard_logs/` during training.

### SLURM Batch Scripts (HPC Users)
- If you work on a supercomputer, adapt the templates in `/.../LAViG-FLOW/pressure_buildup_vae/slurm_files/`.
- `slurm_files/dP_dataset_preparation.slurm` preprocesses the raw tensors into normalised training/validation/test splits.
- `slurm_files/training_vae.slurm` launches multi-GPU training via Accelerate.
- `slurm_files/{training,validation,test}_dataset_reconstructions.slurm` generate reconstructions per split.
- Replace every `"/.../"` placeholder with the site-specific absolute path, ensure the correct conda/env module is activated, and adjust `#SBATCH` resources to match your queue.
