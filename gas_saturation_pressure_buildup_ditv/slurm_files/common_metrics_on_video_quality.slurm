#!/bin/bash -l


#SBATCH --job-name=video-quality-metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=260GB
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/common_metrics.%j.out
#SBATCH --error=/.../LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/common_metrics.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


mkdir -p ${JOINT_ROOT}/logs_autoreg

source activate lavig-flow

USER_SITE=$(python -c "import site; print(site.getusersitepackages())")

STYLEGAN_ROOT_DIR=${STYLEGAN_ROOT_DIR:-${REPO_ROOT}/stylegan-v-main}
STYLEGAN_SRC_DIR=${STYLEGAN_SRC_DIR:-${STYLEGAN_ROOT_DIR}/src}
STYLEGAN_SRC=${STYLEGAN_SRC:-${STYLEGAN_ROOT_DIR}}
export STYLEGAN_V_SRC=${STYLEGAN_ROOT_DIR}
export PYTHONPATH="${USER_SITE}:${REPO_ROOT}:${STYLEGAN_ROOT_DIR}:${STYLEGAN_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH=${CONFIG_PATH:-${JOINT_ROOT}/joint_eval.yaml}
CHUNK_LIST=${CHUNK_LIST:-"4"}   # if chuncks = 4, we will have 23 time frames to compare with 23 time frames from the validation dataset that has 24 frames max
MAX_SAMPLES=${MAX_SAMPLES:-500} # 500 because the validation dataset has 500 samples 
BATCH_SIZE=${BATCH_SIZE:-2}
SEED_VALUE=${SEED_VALUE:-42}
OUTPUT_JSON=${OUTPUT_JSON:-${JOINT_ROOT}/metrics/chunk${CHUNK_LIST}_quality.json}
FVD_BATCH_SIZE=${FVD_BATCH_SIZE:-16}
FVD_NUM_FRAMES=${FVD_NUM_FRAMES:-}
SKIP_FVD=${SKIP_FVD:-0}

CMD=(python ${JOINT_ROOT}/common_metrics_on_video_quality.py
    --config "${CONFIG_PATH}"
    --chunks "${CHUNK_LIST}"
    --batch-size ${BATCH_SIZE}
    --seed ${SEED_VALUE}
    --stylegan-src "${STYLEGAN_SRC}"
    --fvd-batch-size ${FVD_BATCH_SIZE})

if [ -n "${MAX_SAMPLES}" ]; then
    CMD+=(--max-samples ${MAX_SAMPLES})
fi
if [ -n "${OUTPUT_JSON}" ]; then
    mkdir -p "$(dirname -- "${OUTPUT_JSON}")"
    CMD+=(--output-json "${OUTPUT_JSON}")
fi
if [ -n "${FVD_NUM_FRAMES}" ]; then
    CMD+=(--fvd-num-frames ${FVD_NUM_FRAMES})
fi
if [ "${SKIP_FVD}" = "1" ]; then
    CMD+=(--skip-fvd)
fi

srun -n ${SLURM_NTASKS} "${CMD[@]}"