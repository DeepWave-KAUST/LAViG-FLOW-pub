#!/bin/bash -l

#SBATCH --job-name=lavig_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/benchmarking/benchmark_comparison/logs/eval_lavig_metrics.%j.out
#SBATCH --error=/.../LAViG-FLOW/benchmarking/benchmark_comparison/logs/eval_lavig_metrics.%j.err
set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
BENCH_ROOT=${BENCH_ROOT:-${REPO_ROOT}/benchmarking}
COMPARE_ROOT="${BENCH_ROOT}/benchmark_comparison"
LAVIG_ROOT=${LAVIG_ROOT:-${REPO_ROOT}}
JOINT_ROOT="${LAVIG_ROOT}/gas_saturation_pressure_buildup_ditv"
QUALITY_METRICS_SCRIPT=${QUALITY_METRICS_SCRIPT:-${BENCH_ROOT}/common_metrics_on_video_quality.py}

if [ ! -f "${QUALITY_METRICS_SCRIPT}" ]; then
  QUALITY_METRICS_SCRIPT="${JOINT_ROOT}/common_metrics_on_video_quality.py"
fi

mkdir -p ${COMPARE_ROOT}/logs ${COMPARE_ROOT}/results/lavig_flow

source activate lavig-flow

STYLEGAN_ROOT_DIR=${STYLEGAN_ROOT_DIR:-${LAVIG_ROOT}/stylegan-v-main}
STYLEGAN_SRC_DIR=${STYLEGAN_SRC_DIR:-${STYLEGAN_ROOT_DIR}/src}
export STYLEGAN_V_SRC=${STYLEGAN_ROOT_DIR}
export PYTHONPATH="${REPO_ROOT}:${BENCH_ROOT}:${LAVIG_ROOT}:${STYLEGAN_ROOT_DIR}:${STYLEGAN_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH=${CONFIG_PATH:-${JOINT_ROOT}/joint_eval.yaml}
CHUNK_LIST=${CHUNK_LIST:-"1,2,3,4"}
MAX_SAMPLES=${MAX_SAMPLES:-500}
BATCH_SIZE=${BATCH_SIZE:-2}
SEED_VALUE=${SEED_VALUE:-42}
DEVICE_OVERRIDE=${DEVICE_OVERRIDE:-}
OUTPUT_DIR=${OUTPUT_DIR:-${COMPARE_ROOT}/results/lavig_flow}
PHYSICAL_UNITS=${PHYSICAL_UNITS:-0}

FVD_BATCH_SIZE=${FVD_BATCH_SIZE:-16}
FVD_NUM_FRAMES=${FVD_NUM_FRAMES:-}
SKIP_FVD=${SKIP_FVD:-0}
FVD_BOOTSTRAP_REPEATS=${FVD_BOOTSTRAP_REPEATS:-8}
FVD_BOOTSTRAP_SAMPLE_SIZE=${FVD_BOOTSTRAP_SAMPLE_SIZE:-}
FVD_BOOTSTRAP_SEED=${FVD_BOOTSTRAP_SEED:-12345}

IFS=',' read -ra CHUNKS <<< "${CHUNK_LIST}"

for CHUNK in "${CHUNKS[@]}"; do
  CMD=(python ${JOINT_ROOT}/evaluate_autoregressive.py
    --config "${CONFIG_PATH}"
    --chunks "${CHUNK}"
    --batch-size ${BATCH_SIZE}
    --seed ${SEED_VALUE}
    --output-json "${OUTPUT_DIR}/chunk${CHUNK}_autoreg.json")
  if [ "${PHYSICAL_UNITS}" = "1" ]; then
    CMD+=(--physical-units)
  fi
  if [ -n "${MAX_SAMPLES}" ]; then
    CMD+=(--max-samples ${MAX_SAMPLES})
  fi
  if [ -n "${DEVICE_OVERRIDE}" ]; then
    CMD+=(--device "${DEVICE_OVERRIDE}")
  fi
  srun -n ${SLURM_NTASKS} "${CMD[@]}"

  CMD=(python ${QUALITY_METRICS_SCRIPT}
    --config "${CONFIG_PATH}"
    --chunks "${CHUNK}"
    --batch-size ${BATCH_SIZE}
    --seed ${SEED_VALUE}
    --stylegan-src "${STYLEGAN_ROOT_DIR}"
    --fvd-batch-size ${FVD_BATCH_SIZE}
    --fvd-bootstrap-repeats ${FVD_BOOTSTRAP_REPEATS}
    --fvd-bootstrap-seed ${FVD_BOOTSTRAP_SEED}
    --output-json "${OUTPUT_DIR}/chunk${CHUNK}_quality.json")
  if [ -n "${MAX_SAMPLES}" ]; then
    CMD+=(--max-samples ${MAX_SAMPLES})
  fi
  if [ -n "${FVD_NUM_FRAMES}" ]; then
    CMD+=(--fvd-num-frames ${FVD_NUM_FRAMES})
  fi
  if [ -n "${FVD_BOOTSTRAP_SAMPLE_SIZE}" ]; then
    CMD+=(--fvd-bootstrap-sample-size ${FVD_BOOTSTRAP_SAMPLE_SIZE})
  fi
  if [ -n "${DEVICE_OVERRIDE}" ]; then
    CMD+=(--device "${DEVICE_OVERRIDE}")
  fi
  if [ "${SKIP_FVD}" = "1" ]; then
    CMD+=(--skip-fvd)
  fi
  srun -n ${SLURM_NTASKS} "${CMD[@]}"
done
