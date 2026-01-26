#!/bin/bash -l

#SBATCH --job-name=vae-train
#SBATCH --nodes=1
#SBATCH --ntasks=8              
#SBATCH --ntasks-per-node=8     
#SBATCH --cpus-per-task=4 
#SBATCH --constrain=v100
#SBATCH --gres=gpu:8
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/pressure_buildup_vae/logs/vae_training.%j.out
#SBATCH --error=/.../LAViG-FLOW/pressure_buildup_vae/logs/vae_training.%j.err

set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...} # Update parent root
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
PRESSURE_ROOT="${REPO_ROOT}/pressure_buildup_vae"


LOG_DIR=${PRESSURE_ROOT}/logs
mkdir -p "${LOG_DIR}"

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHVISION_DISABLE_IMAGE=1

CONFIG_PATH=${PRESSURE_ROOT}/dP.yaml
TRAIN_SCRIPT=${PRESSURE_ROOT}/training_vae.py

NUM_GPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}"
JOB_HASH=${SLURM_JOB_ID:-0}
MASTER_PORT="${MASTER_PORT:-$((10000 + JOB_HASH % 20000))}"
export MASTER_ADDR MASTER_PORT

MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-fp16}

srun --ntasks=1 accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --num_machines "${SLURM_JOB_NUM_NODES:-1}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --dynamo_backend no \
    --main_process_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG_PATH}"