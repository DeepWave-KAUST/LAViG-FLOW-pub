# StyleGAN-V Snapshot

This folder vendors the official [StyleGAN-V](https://github.com/universome/stylegan-v) release so that the CO₂ projects can reuse the original training and metric utilities (FVD/LPIPS helpers, detector weights, dataset tools, etc.) without requiring an extra checkout. The code under `src/` is a direct snapshot from the upstream repository and should be treated as third-party code.

### Reference
- *StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and
Perks of StyleGAN2* — Skorokhodov et al., 2022 ([Paper](https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf))
- GitHub: https://github.com/universome/stylegan-v 

If you use this code in a publication, please cite the paper above in addition to the LAViG-FLOW work.

### Usage inside LAViG-FLOW
- `gas_saturation_pressure_buildup_ditv/common_metrics_on_video_quality.py` imports `metrics.metric_utils` from this snapshot to compute LPIPS/FVD.
- The helper looks for this folder automatically (relative path `../../stylegan-v-main`), but you can also set `STYLEGAN_V_SRC=/path/to/stylegan-v-main/src` to point at a different checkout.

### Updating the snapshot
1. Download the latest upstream archive or clone the repo.
2. Replace the contents of `stylegan-v-main/src/` with the new snapshot.
3. Document the upstream commit/date here so others know which version is vendored.
4. Re-run any affected tests/metrics to ensure compatibility (StyleGAN-V occasionally changes checkpoint formats and detector URLs).

No local modifications should be made here unless absolutely necessary. Instead, patch upstream or add wrappers elsewhere in the repo so we can keep the snapshot identical to the official release.
