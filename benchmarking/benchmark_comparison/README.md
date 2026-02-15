# LAViG-FLOW Baseline Comparison

This folder contains a small evaluation harness to compare baseline models
(Conv-FNO, FNO, U-FNO, vanilla MIONet, Fourier MIONet)
against the LAViG-FLOW autoregressive metrics setup.
Architecture attribution: model architectures are sourced from the cited
baseline papers; this folder provides only benchmarking/evaluation scripts.

## Paths
- Before running any script, open `paths.py` at the `LAViG-FLOW` repo root and replace `Path("/...")` with the absolute parent directory where `LAViG-FLOW` lives (for example, `Path("/scratch/user")`).
- All `"/.../"` placeholders in this README, YAML configs, and SLURM templates should point to paths under that same root.

It computes:
- Autoregressive metrics: MSE / MAE / RMSE / PSNR
- Quality metrics: SSIM / PSNR / LPIPS / FVD
- Generation time per video

## Files
- `config.yaml`: Fill with your data paths and checkpoints.
- `eval_baselines.py`: Runs metrics and outputs JSON + optional LaTeX tables.
- `timing_baselines.py`: Measures inference time per video.

## Quick start
1. Download pretrained baseline checkpoints [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-).
2. Put them in each baseline folder under `saved_models/`.
3. Edit `config.yaml` and set local checkpoint paths under `models`.
4. Run:
   - `python eval_baselines.py --config config.yaml --output-json results.json --output-tex tables.tex`
   - `python timing_baselines.py --config config.yaml --output-json timing.json`

## Notes
- The scripts expect the *same* validation inputs used by the baseline models
  and the *same* evaluation targets used by LAViG-FLOW.
- Baseline checkpoints are not stored in this repository.
- For quality metrics (LPIPS/FVD), you may need extra dependencies and the
  StyleGAN-V repo. See `config.yaml`.
- If memory is an issue, set `max_samples_quality` in the config.

## Baseline Architecture References
- FNO: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations." https://doi.org/10.48550/arXiv.2010.08895
- U-FNO / conv-FNO baseline reference used here: Wen et al., "U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow." https://doi.org/10.1016/j.advwatres.2022.104180
- Vanilla MIONet: Jin et al., "MIONet: Learning multiple-input operators via tensor product." https://doi.org/10.48550/arXiv.2202.06137
- Fourier-MIONet: Jiang et al., "Fourier-MIONet: Fourier-enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration." https://doi.org/10.1016/j.ress.2024.110392
