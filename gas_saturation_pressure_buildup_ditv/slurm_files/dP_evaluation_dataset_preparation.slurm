#!/bin/bash -l

#SBATCH --job-name=dp-eval-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=260GB
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs/dp_eval_dataset.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs/dp_eval_dataset.%j.err
set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"

mkdir -p ${JOINT_ROOT}/logs

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

srun -n ${SLURM_NTASKS} python ${JOINT_ROOT}/dP_evaluation_dataset_preparation.py