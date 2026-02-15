"""
Central location for absolute paths used across the project.

Update PROJECT_ROOT to match your environment before running any scripts.
REPO_ROOT assumes the repository lives under PROJECT_ROOT / "LAViG-FLOW".
"""

from pathlib import Path

PROJECT_ROOT = Path("/...").resolve() # This is a parent root example. In my case it was PROJECT_ROOT = Path("/ibex/project/c2315/vittoria").resolve() 
REPO_ROOT = PROJECT_ROOT / "LAViG-FLOW"

# Project sub-roots
GAS_SATURATION_VQVAE_ROOT = REPO_ROOT / "gas_saturation_vqvae"
PRESSURE_BUILDUP_VAE_ROOT = REPO_ROOT / "pressure_buildup_vae"
JOINT_DITV_ROOT = REPO_ROOT / "gas_saturation_pressure_buildup_ditv"
BENCHMARKING_ROOT = REPO_ROOT / "benchmarking"
BENCHMARK_COMPARISON_ROOT = BENCHMARKING_ROOT / "benchmark_comparison"
FNO_BENCHMARK_ROOT = BENCHMARKING_ROOT / "fno"
UFNO_BENCHMARK_ROOT = BENCHMARKING_ROOT / "ufno"
CONV_FNO_BENCHMARK_ROOT = BENCHMARKING_ROOT / "conv-fno"
VANILLA_MIONET_BENCHMARK_ROOT = BENCHMARKING_ROOT / "vanilla_MIONet"
FOURIER_MIONET_BENCHMARK_ROOT = BENCHMARKING_ROOT / "Fourier_MIONet"
STYLEGAN_V_ROOT = REPO_ROOT / "stylegan-v-main"

# Dataset directories
CO2_DATA_ROOT = GAS_SATURATION_VQVAE_ROOT / "co2_data"
CO2_EVAL_DATA_ROOT = GAS_SATURATION_VQVAE_ROOT / "co2_data_evaluation"
DP_DATA_ROOT = PRESSURE_BUILDUP_VAE_ROOT / "dP_data"
DP_EVAL_DATA_ROOT = PRESSURE_BUILDUP_VAE_ROOT / "dP_data_evaluation"
JOINT_CO2_ROOT = JOINT_DITV_ROOT / "co2_data"
JOINT_DP_ROOT = JOINT_DITV_ROOT / "dP_data"
