# Fourier-MIONet baseline

This folder provides a **Fourier-MIONet** baseline.
Architecture attribution: this repository uses a benchmark implementation of
the architecture reported in the cited paper; the original architecture was
proposed by the cited authors, not by this repository.

## Paths
- Before running any script, open `paths.py` at the `LAViG-FLOW` repo root and replace `Path("/...")` with the absolute parent directory where `LAViG-FLOW` lives (for example, `Path("/scratch/user")`).
- All `"/.../"` placeholders in this README, YAML configs, and SLURM templates should point to paths under that same root.

## Files
- `model.py`: Fourier-enhanced branch/decoder modules.
- `train_Fourier_MIONet_sg.py`: train SG model from YAML config.
- `train_Fourier_MIONet_dp.py`: train dP model from YAML config.
- `train_sg.yaml`: default config for SG training.
- `train_dp.yaml`: default config for dP training.
- `slurm_files/`: Slurm scripts for training.

## Train
```bash
python train_Fourier_MIONet_sg.py --config train_sg.yaml
python train_Fourier_MIONet_dp.py --config train_dp.yaml
```

## Pretrained checkpoints
Pretrained checkpoints are not stored in this repository.
Download them [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-).
For shared evaluation, set local checkpoint paths in
`../benchmark_comparison/config.yaml` under:
- `models.fourier_mionet.gas`
- `models.fourier_mionet.pressure`

## Eval
Run shared baseline evaluation from `../benchmark_comparison/eval_baselines.py`.

Edit the YAML files to set `data_dir`, `save_dir`, and training hyperparameters. Data is read from `co2_data/` and `dP_data/` under `data_dir`.
Note: A local copy of the modified `deepxde` is included in this folder.

## Original Architecture Reference
- Jiang, Zhongyi; Zhu, Min; Lu, Lu. "Fourier-MIONet: Fourier-enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration." Reliability Engineering & System Safety, 251 (2024), 110392. https://doi.org/10.1016/j.ress.2024.110392
