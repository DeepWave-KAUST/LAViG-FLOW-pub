# U-FNO baseline

This folder provides a **U-FNO** baseline.
Architecture attribution: this repository uses a benchmark implementation of
the architecture reported in the cited paper; the original architecture was
proposed by the cited authors, not by this repository.

## Paths
- Before running any script, open `paths.py` at the `LAViG-FLOW` repo root and replace `Path("/...")` with the absolute parent directory where `LAViG-FLOW` lives (for example, `Path("/scratch/user")`).
- All `"/.../"` placeholders in this README, YAML configs, and SLURM templates should point to paths under that same root.

## Files
- `ufno.py`: U-FNO model definition.
- `lploss.py`: relative Lp loss.
- `train_UFNO_sg.py`: train SG model from YAML config.
- `train_UFNO_dp.py`: train dP model from YAML config.
- `train_sg.yaml`: default config for SG training.
- `train_dp.yaml`: default config for dP training.

## Train
```bash
python train_UFNO_sg.py --config train_sg.yaml
python train_UFNO_dp.py --config train_dp.yaml
```

## Pretrained checkpoints
Pretrained checkpoints are not stored in this repository.
Download them [here](https://drive.google.com/drive/u/0/folders/1GIlXzK6zPh7Y1MAeLJYNwfOS_DnkdB1-).
For shared evaluation, set local checkpoint paths in
`../benchmark_comparison/config.yaml` under:
- `models.ufno.gas`
- `models.ufno.pressure`

## Eval
Run shared baseline evaluation from `../benchmark_comparison/eval_baselines.py`.

Edit the YAML files to set `data_dir`, `save_dir`, and hyperparameters (epochs, batch size, scheduler, etc.).

## Original Architecture Reference
- Wen, Gege; Li, Zongyi; Azizzadenesheli, Kamyar; Anandkumar, Anima; Benson, Sally M. "U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow." Advances in Water Resources. https://doi.org/10.1016/j.advwatres.2022.104180
