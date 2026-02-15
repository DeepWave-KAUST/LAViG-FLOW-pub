# FNO baseline

This folder provides an **FNO** baseline.
Architecture attribution: this repository uses a benchmark implementation of
the architecture reported in the cited paper; the original architecture was
proposed by the cited authors, not by this repository.

## Paths
- Before running any script, open `paths.py` at the `LAViG-FLOW` repo root and replace `Path("/...")` with the absolute parent directory where `LAViG-FLOW` lives (for example, `Path("/scratch/user")`).
- All `"/.../"` placeholders in this README, YAML configs, and SLURM templates should point to paths under that same root.

## Files
- `fno.py`: FNO model (6 Fourier layers, no U-Net path).
- `lploss.py`: relative Lp loss.
- `train_FNO_sg.py`: train SG model from YAML config.
- `train_FNO_dp.py`: train dP model from YAML config.
- `train_sg.yaml`: default config for SG training.
- `train_dp.yaml`: default config for dP training.

## Train
```bash
python train_FNO_sg.py --config train_sg.yaml
python train_FNO_dp.py --config train_dp.yaml
```

## Pretrained checkpoints
Pretrained checkpoints are not stored in this repository.
Download them [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-).
For shared evaluation, set local checkpoint paths in
`../benchmark_comparison/config.yaml` under:
- `models.fno.gas`
- `models.fno.pressure`

## Eval
Run shared baseline evaluation from `../benchmark_comparison/eval_baselines.py`.

Edit the YAML files to set `data_dir`, `save_dir`, and hyperparameters (epochs, batch size, scheduler, etc.).

## Original Architecture Reference
- Li, Zongyi; Kovachki, Nikola; Azizzadenesheli, Kamyar; Liu, Burigede; Bhattacharya, Kaushik; Stuart, Andrew; Anandkumar, Anima. "Fourier Neural Operator for Parametric Partial Differential Equations." arXiv:2010.08895. https://doi.org/10.48550/arXiv.2010.08895
