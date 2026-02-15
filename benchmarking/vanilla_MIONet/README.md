# vanilla MIONet baseline

This folder provides a **vanilla MIONet** baseline.
Architecture attribution: this repository uses a benchmark implementation of
the architecture reported in the cited paper; the original architecture was
proposed by the cited authors, not by this repository.

## Paths
- Before running any script, open `paths.py` at the `LAViG-FLOW` repo root and replace `Path("/...")` with the absolute parent directory where `LAViG-FLOW` lives (for example, `Path("/scratch/user")`).
- All `"/.../"` placeholders in this README, YAML configs, and SLURM templates should point to paths under that same root.

## Files
- `model.py`: encoder + MIONet Cartesian product model.
- `train_MIONet_vanilla_sg.py`: train SG model from YAML config.
- `train_MIONet_vanilla_dp.py`: train dP model from YAML config.
- `train_sg.yaml`: default config for SG training.
- `train_dp.yaml`: default config for dP training.
- `slurm_files/`: Slurm scripts for training.

## Train
```bash
python train_MIONet_vanilla_sg.py --config train_sg.yaml
python train_MIONet_vanilla_dp.py --config train_dp.yaml
```

## Pretrained checkpoints
Pretrained checkpoints are not stored in this repository.
Download them [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-).
For shared evaluation, set local checkpoint paths in
`../benchmark_comparison/config.yaml` under:
- `models.vanilla_mionet.gas`
- `models.vanilla_mionet.pressure`

## Eval
Run shared baseline evaluation from `../benchmark_comparison/eval_baselines.py`.

Edit the YAML files to set `data_dir`, `save_dir`, and training hyperparameters. Data is read from `co2_data/` and `dP_data/` under `data_dir`.
Note: A local copy of the modified `deepxde` is included in this folder.

## Original Architecture Reference
- Jin, Pengzhan; Meng, Shuai; Lu, Lu. "MIONet: Learning multiple-input operators via tensor product." arXiv:2202.06137. https://doi.org/10.48550/arXiv.2202.06137
