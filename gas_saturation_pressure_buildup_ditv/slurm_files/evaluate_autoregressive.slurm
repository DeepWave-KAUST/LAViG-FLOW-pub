#!/bin/bash -l

#SBATCH --job-name=evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=260GB
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/evaluate_autoreg.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/evaluate_autoreg.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


mkdir -p ${JOINT_ROOT}/logs_autoreg

source activate lavig-flow

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH=${CONFIG_PATH:-${JOINT_ROOT}/joint_eval.yaml}
CHUNK_LIST=${CHUNK_LIST:-"4"} # Specify exactly one chunk per run (submit separate jobs for 1,2,3,4)
MAX_SAMPLES=${MAX_SAMPLES:-500}
BATCH_SIZE=${BATCH_SIZE:-2}
SEED_VALUE=${SEED_VALUE:-42}
OUTPUT_JSON=${OUTPUT_JSON:-${JOINT_ROOT}/metrics/chunk${CHUNK_LIST}_autoreg.json}
DEVICE_OVERRIDE=${DEVICE_OVERRIDE:-}

CMD=(python ${JOINT_ROOT}/evaluate_autoregressive.py
    --config "${CONFIG_PATH}"
    --chunks "${CHUNK_LIST}"
    --batch-size ${BATCH_SIZE}
    --seed ${SEED_VALUE})

if [ -n "${MAX_SAMPLES}" ]; then
    CMD+=(--max-samples ${MAX_SAMPLES})
fi
if [ -n "${OUTPUT_JSON}" ]; then
    mkdir -p "$(dirname -- "${OUTPUT_JSON}")"
    CMD+=(--output-json "${OUTPUT_JSON}")
fi
if [ -n "${DEVICE_OVERRIDE}" ]; then
    CMD+=(--device "${DEVICE_OVERRIDE}")
fi

srun -n ${SLURM_NTASKS} "${CMD[@]}"
