#!/bin/bash -l


#SBATCH --job-name=fourier-mionet-train-gas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH --time=48:00:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/benchmarking/Fourier_MIONet/logs/train_Fourier_MIONet_sg.%j.out
#SBATCH --error=/.../LAViG-FLOW/benchmarking/Fourier_MIONet/logs/train_Fourier_MIONet_sg.%j.err
set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
BENCH_ROOT=${BENCH_ROOT:-${REPO_ROOT}/benchmarking}
MODEL_ROOT="${BENCH_ROOT}/Fourier_MIONet"

source activate lavig-flow
mkdir -p "${MODEL_ROOT}/logs" "${MODEL_ROOT}/saved_models"
cd "${MODEL_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + RANDOM % 20000))
export NCCL_DEBUG=WARN

srun accelerate launch   --main_process_port ${MASTER_PORT}   --num_processes ${SLURM_NTASKS}   --num_machines ${SLURM_JOB_NUM_NODES}   train_Fourier_MIONet_sg.py --config train_sg.yaml
