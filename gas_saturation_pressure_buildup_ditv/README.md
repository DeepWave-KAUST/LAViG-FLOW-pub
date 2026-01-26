# CO₂ Gas Saturation + Pressure Build-Up DiTV

End-to-end instructions for preparing the datasets, training the DiTV model, sampling, and computing metrics for the coupled CO₂ gas saturation + pressure build-up workflow.

### 1. Environment & Paths
- Create/activate the shared env: `conda env create -f ../environment.yml && conda activate lavig-flow`
- Update `paths.py` at the repo root so `Path("/...")` points to the absolute parent directory that contains `LAViG-FLOW`. All remaining `/.../` placeholders in the YAML files and SLURM templates should resolve under that same root.

### 2. Prerequisites
- Train (or download) the single-task autoencoders:
  - `gas_saturation_vqvae/trained_models/vqvae_model.pth`
  - `pressure_buildup_vae/trained_models/vae_model.pth`
- Edit `joint.yaml`, `joint_autoreg.yaml`, and `joint_eval.yaml` so every `stats_paths`, `file_map`, `ckpt_path`, `log_dir`, etc. entry points to your filesystem.

### 3. Dataset Preparation
Scripts in this folder take the **raw** tensors that already live under the single-task projects (`gas_saturation_vqvae/co2_data/*`, `pressure_buildup_vae/dP_data/*`) and re-package them for the joint DiTV pipeline. In other words, you download/rename the upstream `.pt` files only once inside the VQ-VAE / VAE folders, and every helper below simply streams those canonical tensors to avoid duplicating hundreds of gigabytes. For the exact raw-file naming scheme, refer to `gas_saturation_vqvae/README.md` and `pressure_buildup_vae/README.md`. Run the helpers below once the raw tensors are in place and the paths have been updated:

```bash
# Gas saturation tensors 
python gas_saturation_pressure_buildup_ditv/co2_dataset_preparation.py

# Pressure build-up tensors 
python gas_saturation_pressure_buildup_ditv/dP_dataset_preparation.py

# Gas saturation validation tensors → evaluation bundle (reuses raw val tensors from `gas_saturation_vqvae`)
python gas_saturation_pressure_buildup_ditv/co2_evaluation_dataset_preparation.py

# Pressure build-up validation tensors → evaluation bundle (reuses raw val tensors from `pressure_buildup_vae`)
python gas_saturation_pressure_buildup_ditv/dP_evaluation_dataset_preparation.py
```

HPC users can mirror the same steps via:
- `slurm_files/co2_dataset_preparation.slurm` / `dP_dataset_preparation.slurm`.
- `slurm_files/co2_evaluation_dataset_preparation.slurm` / `dP_evaluation_dataset_preparation.slurm` for the validation/evaluation bundle.

Optional visual sanity checks:

```bash
python gas_saturation_pressure_buildup_ditv/co2_and_dP_samples_visualization.py \
  --dataset-types train test validation \
  --parameters sg dP \
  --max-samples 12 \
  --device cuda \
  --xlim-sg 0 1500 \
  --xlim-dp 0 1500
```

Use `--max-samples -1` to scan every sample in a split, or swap `--dataset-types`/`--parameters` to target just one modality.

HPC users can run the same helper via `slurm_files/co2_and_dP_samples_visualization.slurm`. Update the `/...` placeholders for `#SBATCH --output/--error` and `PROJECT_ROOT`, then submit:

```bash
sbatch gas_saturation_pressure_buildup_ditv/slurm_files/co2_and_dP_samples_visualization.slurm
```

The figures land under `gas_saturation_pressure_buildup_ditv/samples_*`.

### 4. Multi-GPU DiTV Training
Training happens in **two explicit phases**: (1) unconditional joint modelling so the DiTV learns a shared latent of CO₂ + ΔP trajectories, followed by (2) autoregressive fine-tuning that teaches the network to roll forward in time chunk-by-chunk. Both stages reuse the same entrypoint (`multi_gpu_train_ditv.py`) but different YAMLs and checkpoint knobs.  
*Shortcut:* If you only need inference/evaluation, download the pretrained checkpoints (`joint_dit.pth`, `joint_dit_autoreg.pth`,`vqvae_model.pth`,`vae_model.pth`) from [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-). Place them under `gas_saturation_vqvae/trained_models/`, `pressure_buildup_vae/trained_models/`, and `gas_saturation_pressure_buildup_ditv/trained_models/` as referenced in the YAMLs, then skip to Section 5.

#### 4.1 Phase 1 — Unconditional joint training (`joint.yaml`)
- Objective: train for 1,000 epochs (`train_params.ditv_epochs`) with full-length clips so the model stores paired pressure/saturation distributions without autoregressive masking.
- Config highlights:
  - `dataset_params.file_map` points to the tensors produced by the prep scripts.
  - `autoencoder_params.*.ckpt_path` load the frozen VQ-VAE (CO₂) and VAE (ΔP).
  - `train_params.ditv_ckpt_name` / `ditv_ckpt_complete_name` define where the baseline checkpoint lands (this path is later reused by Phase 2).
- Launch command (mirrors `slurm_files/multi_gpu_train_ditv.slurm`; adjust resources as needed):

```bash
accelerate launch \
  --num_processes 6 \
  --num_machines 1 \
  --mixed_precision fp16 \
  gas_saturation_pressure_buildup_ditv/multi_gpu_train_ditv.py \
  --config gas_saturation_pressure_buildup_ditv/joint.yaml
```

- HPC workflow: edit `slurm_files/multi_gpu_train_ditv.slurm` so every `/...` placeholder (project root, `#SBATCH --output/--error`) matches your filesystem, then submit with:

  ```bash
  sbatch gas_saturation_pressure_buildup_ditv/slurm_files/multi_gpu_train_ditv.slurm
  ```

  The template activates `lavig-flow`, exports `PYTHONPATH`/`TORCHVISION_DISABLE_IMAGE`, seeds NCCL env vars, and invokes the same `accelerate launch` command with `srun`.

- Outputs:
  - `trained_models/joint_dit.pth` → lightweight checkpoint (model weights only). This is the artifact consumed by sampling/eval and passed to Phase 2 via `train_params.base_ditv_ckpt`.
  - `trained_models/joint_dit_full.pth` → optimizer + scheduler state for resuming long Phase‑1 runs; not needed for inference.
  - Scalars/TensorBoard logs in `train_params.log_dir` and loss plots in `train_params.loss_dir`.

#### 4.2 Phase 2 — Autoregressive fine-tuning (`joint_autoreg.yaml`)
- Objective: start from the Phase-1 weights and fine-tune with sliding windows (15 context + 2 predict frames, stride 1) so the model can forecast future CO₂/ΔP slices autoregressively.
- Config highlights:
  - `dataset_params.autoregressive` defines the context/predict windows.
  - `train_params.base_ditv_ckpt` must point to the Phase-1 checkpoint (`joint_dit.pth`).
  - New outputs are written to `train_params.ditv_ckpt_name` / `ditv_ckpt_complete_name` (e.g., `joint_dit_autoreg*.pth`).
  - `autoregressive_params` toggles context masking/noise versus gradient flow.
- Launch command (same script, different config; see `slurm_files/multi_gpu_train_ditv_fine_tuning.slurm`):

```bash
accelerate launch \
  --num_processes 6 \
  --num_machines 1 \
  --mixed_precision fp16 \
  gas_saturation_pressure_buildup_ditv/multi_gpu_train_ditv.py \
  --config gas_saturation_pressure_buildup_ditv/joint_autoreg.yaml
```

- HPC workflow: customize `slurm_files/multi_gpu_train_ditv_fine_tuning.slurm` the same way (paths, partitions, time) and submit:

  ```bash
  sbatch gas_saturation_pressure_buildup_ditv/slurm_files/multi_gpu_train_ditv_fine_tuning.slurm
  ```

  That job reuses the Phase‑1 checkpoint via `train_params.base_ditv_ckpt`, writes logs under `logs_autoreg/`, and mirrors the CLI flags shown above.

- Outputs:
  - `trained_models/joint_dit_autoreg.pth` → fine-tuned weights for autoregressive sampling/eval.
  - `trained_models/joint_dit_autoreg_full.pth` → resume-only bundle (weights + optimizer/scheduler) if Phase‑2 is interrupted.
  - Separate TensorBoard + loss directories (`tensorboard_logs_autoreg`, `loss_plots_autoreg`) so you can track fine-tuning independently.

### 5. Sampling & Visualisation
`sample_ditv.py` serves both the unconditional (Phase‑1) and autoregressive (Phase‑2) checkpoints—the config you pass determines the behaviour.

#### 5.1 Unconditional inspection (`joint.yaml`)
- Purpose: sanity-check the 1,000-epoch baseline by letting it hallucinate entire CO₂/ΔP clips without autoregressive stitching.
- Command (override `--num-samples` to see more or fewer trajectories; `--num-chunks` is forced to 1 because `joint.yaml` has no autoregressive block):

  ```bash
  python gas_saturation_pressure_buildup_ditv/sample_ditv.py \
    --config gas_saturation_pressure_buildup_ditv/joint.yaml \
    --checkpoint gas_saturation_pressure_buildup_ditv/trained_models/joint_dit.pth \
    --num-samples 24 \
    --num-chunks 1
  ```

- HPC workflow: fill in the `/...` placeholders inside `slurm_files/sample_ditv.slurm` (project root, log paths) and submit via `sbatch`. That wrapper activates `lavig-flow`, exports `PYTHONPATH`, and runs the same CLI through `srun`.

#### 5.2 Autoregressive rollouts (`joint_autoreg.yaml`)
- Purpose: start from the fine-tuned checkpoint and roll forward progressively. The config exposes `dataset_params.autoregressive.context_frames = 15` and `predict_frames = 2`, so the temporal budget is:

  ```
  total_frames = context_frames + predict_frames + (num_chunks - 1) * predict_frames
  ```

  With the defaults, chunk = 1 yields 15 context frames + 2 predicted frames (17 total). Each extra chunk adds two more predicted frames (chunk = 3 produces the base 15 context + 6 predicted frames; equivalently, four additional frames beyond chunk = 1). Keep `chunk` in `[1, 4]` to stay within the SLURM templates.

- Command (set both the autoregressive config and checkpoint, then pick how many samples/chunks you want to visualise):

  ```bash
  python gas_saturation_pressure_buildup_ditv/sample_ditv.py \
    --config gas_saturation_pressure_buildup_ditv/joint_autoreg.yaml \
    --checkpoint gas_saturation_pressure_buildup_ditv/trained_models/joint_dit_autoreg.pth \
    --num-samples 16 \
    --num-chunks 4
  ```

- HPC workflow: edit `slurm_files/sample_ditv_fine_tuning.slurm` (paths, resources) and submit with `sbatch`. The script points to `joint_autoreg.yaml`, writes logs under `logs_autoreg/`, and honours the same CLI flags via exported environment variables.

- Outputs (for both modes): `sample_ditv.py` saves `joint_samples.pt`, PNG mosaics, and MP4 strips beneath the directory specified by `infer_params.output_dir` inside the chosen config.

### 6. Autoregressive Evaluation
Before running any of the evaluation scripts, make sure the validation bundles have been generated with `co2_evaluation_dataset_preparation.py` and `dP_evaluation_dataset_preparation.py` (or their SLURM counterparts); the commands below expect the resulting files under `co2_data_evaluation/` and `dP_data_evaluation/`.
#### 6.1 Visual validation benchmark (autoregressive rollouts)
- Objective: use the evaluation dataset prepared earlier (validation split only) to inspect how the fine-tuned DiTV forecasts future frames relative to ground truth.
- `visualize_autoregressive_sample.py` consumes `joint_eval.yaml`, picks a validation sample, feeds the first 15 frames as context, and asks the model to autoregressively predict the remaining frames. Each additional chunk adds two predicted frames (`context_frames=15`, `predict_frames=2`), so chunk = 1 compares a 15→17 rollout, chunk = 4 compares a 15→23 rollout (15 context + 8 predicted frames).
- Recommended workflow: loop over both sample indices and chunk counts so you cover multiple wells and horizons. Example (CPU/GPU interactive shell):

  ```bash
  for chunk in 1 2 3 4; do
    for sample in $(seq 0 10); do
      python gas_saturation_pressure_buildup_ditv/visualize_autoregressive_sample.py \
        --config gas_saturation_pressure_buildup_ditv/joint_eval.yaml \
        --sample-index "${sample}" \
        --chunks "${chunk}" \
        --output-dir gas_saturation_pressure_buildup_ditv/pseudo_conditional_generation_autoreg
    done
  done
  ```

- Each run writes CO₂/ΔP comparison grids under `--output-dir`, showing context frames, predictions, ground truth, and errors side by side so you can visually confirm the diffusion model tracks the validation set.
- HPC workflow: edit `/.../slurm_files/visualize_autoregressive.slurm` (paths, queues, CHUNK_LIST/SAMPLE_INDEX_LIST/OUTPUT_DIR variables) and submit with `sbatch`. The template iterates over every chunk in `[1,4]` and every validation sample index you specify.

#### 6.2 Deterministic error metrics

[Validation set has 500 samples; process all of them once per chunk.]

```bash
python gas_saturation_pressure_buildup_ditv/evaluate_autoregressive.py \
  --config gas_saturation_pressure_buildup_ditv/joint_eval.yaml \
  --chunks 1 \
  --batch-size 2 \
  --max-samples 500 \
  --seed 42 \
  --device cuda \
  > logs/chunk1_eval_autoreg.json

python gas_saturation_pressure_buildup_ditv/evaluate_autoregressive.py \
  --config gas_saturation_pressure_buildup_ditv/joint_eval.yaml \
  --chunks 2 \
  --batch-size 2 \
  --max-samples 500 \
  --seed 42 \
  --device cuda \
  > logs/chunk2_eval_autoreg.json
# repeat for --chunks 3 and --chunks 4
```

- Run the script four times (chunks = 1, 2, 3, 4) so you benchmark every rollout horizon the model was trained on.
- Each run reports MAE, MSE, and RMSE (per modality) over the predicted frames, plus a summary snippet with the number of evaluated samples and frames.
- `--max-samples 500` covers the entire validation split; drop it if you want a quicker smoke test.
- HPC workflow: use `/.../slurm_files/evaluate_autoregressive.slurm` and set `CHUNK_LIST`, `MAX_SAMPLES`, `BATCH_SIZE`, etc. before `sbatch`. The template assumes one chunk per submission (e.g. `CHUNK_LIST=1`), and it writes each run to `${JOINT_ROOT}/metrics/chunk${CHUNK_LIST}_autoreg.json` unless you override `OUTPUT_JSON`.

#### 6.3 Perceptual + video-quality metrics (SSIM / PSNR / LPIPS / FVD)
- Before running the metrics script, download the StyleGAN-V I3D detector (`i3d_torchscript.pt`) from [here]() and place it under `gas_saturation_pressure_buildup_ditv/metric_detectors/` (create the folder if it does not exist). The script looks for that file to compute LPIPS/FVD; without it, PSNR/SSIM/LPIPS/FVD will fail.

```bash
export STYLEGAN_V_SRC=/abs/path/to/stylegan-v-main/src  # optional if you keep the vendored folder
python gas_saturation_pressure_buildup_ditv/common_metrics_on_video_quality.py \
  --config gas_saturation_pressure_buildup_ditv/joint_eval.yaml \
  --chunks 4 \
  --batch-size 2 \
  --max-samples 500 \
  --seed 42 \
  --stylegan-src "${STYLEGAN_V_SRC}" \
  --fvd-batch-size 16 \
  --fvd-num-frames 23 \
  > logs/chunk4_perceptual_metrics.json
```

- Leave `--stylegan-src` unset if the vendored `stylegan-v-main/src` folder sits one level up (the script resolves it automatically). Otherwise, point it to your own checkout.
- Pass `--skip-fvd` when detector downloads or GPU memory are an issue.
- Results print to stdout and can be stored with `--output-json`. If you prefer a persisted JSON, either redirect stdout (as in the example above) or pass `--output-json /abs/path/to/chunk4_perceptual.json`. The SLURM template (`slurm_files/common_metrics_on_video_quality.slurm`) writes to `${JOINT_ROOT}/metrics/chunk${CHUNK_LIST}_perceptual.json` by default (one chunk per submission) unless you override `OUTPUT_JSON`.

### 7. Tips
- Keep `paths.py` in sync with wherever you copy datasets/checkpoints; most scripts import those constants.
- The evaluation scripts rely on the same stats tensors generated during dataset preparation (`*_min_max.pt`). Regenerate them if you move datasets across machines.
- Set `TORCHVISION_DISABLE_IMAGE=1` and `PYTHONPATH` to include the repo when launching from scratch (the SLURM templates already do this).
- For reproducibility, stick to the seeds in the configs (`train_params.seed`, `--seed` flags) unless you are explicitly exploring variability.
[If you use user-site installs, prepend:]

```bash
USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="${USER_SITE}:${PWD}:${PYTHONPATH}"
```
