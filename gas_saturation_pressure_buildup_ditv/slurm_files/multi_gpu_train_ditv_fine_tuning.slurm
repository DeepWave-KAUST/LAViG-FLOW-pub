#!/bin/bash -l

#SBATCH --job-name=mgpu-train-ditv-autoreg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:6
#SBATCH --mem=260GB
#SBATCH --time=48:00:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/multi_gpu_train_ditv.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/multi_gpu_train_ditv.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


mkdir -p ${JOINT_ROOT}/logs_autoreg

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHVISION_DISABLE_IMAGE=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + RANDOM % 20000))
export NCCL_DEBUG=WARN

srun accelerate launch \
  --main_process_port ${MASTER_PORT} \
  --num_processes ${SLURM_GPUS_ON_NODE} \
  --num_machines ${SLURM_JOB_NUM_NODES} \
  --mixed_precision fp16 \
  ${JOINT_ROOT}/multi_gpu_train_ditv.py \
  --config ${JOINT_ROOT}/joint_autoreg.yaml
