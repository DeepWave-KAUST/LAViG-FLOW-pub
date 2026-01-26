#!/bin/bash -l

#SBATCH --job-name=sample-ditv-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=260GB
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/sample_ditv.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/sample_ditv.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


mkdir -p ${JOINT_ROOT}/logs_autoreg

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=()
CHECKPOINT=${CHECKPOINT:-}
NUM_SAMPLES=${NUM_SAMPLES:-}
NUM_CHUNKS=${NUM_CHUNKS:-}

if [ -n "${CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${NUM_SAMPLES}" ]; then
    EXTRA_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi
if [ -n "${NUM_CHUNKS}" ]; then
    EXTRA_ARGS+=(--num-chunks "${NUM_CHUNKS}")
fi

srun -n ${SLURM_NTASKS} python ${JOINT_ROOT}/sample_ditv.py \
    --config ${JOINT_ROOT}/joint_autoreg.yaml \
    "${EXTRA_ARGS[@]}"