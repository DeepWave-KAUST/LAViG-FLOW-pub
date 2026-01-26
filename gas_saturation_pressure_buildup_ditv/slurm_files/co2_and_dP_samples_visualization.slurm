#!/bin/bash -l

#SBATCH --job-name=co2-dP-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=260GB
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs/co2_and_dP_samples_visualization.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs/co2_and_dP_samples_visualization.%j.err

set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


mkdir -p ${JOINT_ROOT}/logs

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

srun -n ${SLURM_NTASKS} python ${JOINT_ROOT}/co2_and_dP_samples_visualization.py \
  --max-samples 20 \
  --dataset-types train \
  --parameters sg dP \
  --device cpu \
  --xlim-sg 0 1500 \
  --xlim-dp 0 1500
