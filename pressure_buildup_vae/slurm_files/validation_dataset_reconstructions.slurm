#!/bin/bash -l

#SBATCH --job-name=vae-recon-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=240GB
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/pressure_buildup_vae/logs/vae_recon_validation.%j.out
#SBATCH --error=/.../LAViG-FLOW/pressure_buildup_vae/logs/vae_recon_validation.%j.err

set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...} # Update parent root
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
PRESSURE_ROOT="${REPO_ROOT}/pressure_buildup_vae"


LOG_DIR="${PRESSURE_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHVISION_DISABLE_IMAGE=1

CONFIG_PATH=${CONFIG_PATH:-${PRESSURE_ROOT}/trained_models/used_train_config.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${PRESSURE_ROOT}/trained_models/vae_model.pth}
SCRIPT_PATH=${SCRIPT_PATH:-${PRESSURE_ROOT}/validation_dataset_reconstructions.py}
OUTPUT_DIR=${OUTPUT_DIR:-${PRESSURE_ROOT}/results/reconstructions/validation}
DEVICE=${DEVICE:-cuda}

# ------------------------------------------------------------------
# User-configurable defaults (edit here or override via environment)
# ------------------------------------------------------------------
DEFAULT_SAMPLE_OFFSET=0
DEFAULT_MAX_SAMPLES=8500      # total number of samples
DEFAULT_MOSAIC_COUNT=17       # number of plotted time frames 
DEFAULT_NUM_PLOT_SAMPLES=100  # number of plotted samples
DEFAULT_MOSAIC_COLS=17

SAMPLE_OFFSET=${SAMPLE_OFFSET:-${DEFAULT_SAMPLE_OFFSET}}
MAX_SAMPLES=${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}
MOSAIC_COUNT=${MOSAIC_COUNT:-${DEFAULT_MOSAIC_COUNT}}
NUM_PLOT_SAMPLES=${NUM_PLOT_SAMPLES:-${DEFAULT_NUM_PLOT_SAMPLES}}
MOSAIC_COLS=${MOSAIC_COLS:-${DEFAULT_MOSAIC_COLS}}

if [[ -n "${NUM_PLOT_SAMPLES}" ]]; then
    MOSAIC_COUNT="${NUM_PLOT_SAMPLES}"
fi

PYTHON_ARGS=(
    --config "${CONFIG_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --device "${DEVICE}"
    --output-dir "${OUTPUT_DIR}"
    --sample-offset "${SAMPLE_OFFSET}"
   --mosaic-count "${MOSAIC_COUNT}"
    --mosaic-cols "${MOSAIC_COLS}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    PYTHON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${MOSAIC_ERROR_VMAX:-}" ]]; then
    PYTHON_ARGS+=(--mosaic-error-vmax "${MOSAIC_ERROR_VMAX}")
fi

srun python "${SCRIPT_PATH}" "${PYTHON_ARGS[@]}"
