#!/bin/bash -l 


#SBATCH --job-name=dP-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4     # Increase if needed
#SBATCH --mem=260GB 
#SBATCH --time=00:30:00
#SBATCH --partition=batch 
#SBATCH --output=/.../LAViG-FLOW/pressure_buildup_vae/logs/dP_dataset_preparation.%j.out
#SBATCH --error=/.../LAViG-FLOW/pressure_buildup_vae/logs/dP_dataset_preparation.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...} # Update parent root
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
PRESSURE_ROOT="${REPO_ROOT}/pressure_buildup_vae"


# make sure the logs dir exists
mkdir -p ${PRESSURE_ROOT}/logs

# activate your conda env
source activate lavig-flow

# (optional) ensure your package path is visible
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# run the python script
srun -n ${SLURM_NTASKS} python ${PRESSURE_ROOT}/dP_dataset_preparation.py
