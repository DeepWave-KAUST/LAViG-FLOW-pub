#!/bin/bash -l 

#SBATCH --job-name=dP-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4     # Increase if needed
#SBATCH --mem=260GB 
#SBATCH --time=01:00:00
#SBATCH --partition=batch 
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs/dP_dataset_preparation.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs/dP_dataset_preparation.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


# make sure the logs dir exists
mkdir -p ${JOINT_ROOT}/logs

# activate your conda env
source activate lavig-flow

# (optional) ensure your package path is visible
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# run the python script
srun -n ${SLURM_NTASKS} python ${JOINT_ROOT}/dP_dataset_preparation.py
