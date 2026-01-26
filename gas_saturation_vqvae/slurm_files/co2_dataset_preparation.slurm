#!/bin/bash -l 

#SBATCH --job-name=co2-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4    
#SBATCH --mem=260GB 
#SBATCH --time=00:30:00
#SBATCH --partition=batch 
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_vqvae/logs/co2_dataset_preparation.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_vqvae/logs/co2_dataset_preparation.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...} # Update parent root
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
GAS_VQVAE_ROOT="${REPO_ROOT}/gas_saturation_vqvae"


# make sure the logs dir exists
mkdir -p ${GAS_VQVAE_ROOT}/logs

# activate your conda env
source activate lavig-flow

# (optional) ensure your package path is visible
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# run the python script
srun -n ${SLURM_NTASKS} python ${GAS_VQVAE_ROOT}/co2_dataset_preparation.py
