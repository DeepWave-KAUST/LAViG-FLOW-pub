#!/bin/bash -l

#SBATCH --job-name=baseline_time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=260GB
#SBATCH --time=72:00:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/benchmarking/benchmark_comparison/logs/timing_1cpu.%j.out
#SBATCH --error=/.../LAViG-FLOW/benchmarking/benchmark_comparison/logs/timing_1cpu.%j.err
set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
BENCH_ROOT=${BENCH_ROOT:-${REPO_ROOT}/benchmarking}
COMPARE_ROOT="${BENCH_ROOT}/benchmark_comparison"

mkdir -p ${COMPARE_ROOT}/logs ${COMPARE_ROOT}/results

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH=${CONFIG_PATH:-${COMPARE_ROOT}/config.yaml}
OUTPUT_JSON=${OUTPUT_JSON:-${COMPARE_ROOT}/results/timing_1cpu.json}
DEVICE_OVERRIDE=${DEVICE_OVERRIDE:-}

CMD=(python ${COMPARE_ROOT}/timing_baselines.py
    --config "${CONFIG_PATH}"
    --output-json "${OUTPUT_JSON}")

if [ -n "${DEVICE_OVERRIDE}" ]; then
    CMD+=(--device "${DEVICE_OVERRIDE}")
fi

srun -n ${SLURM_NTASKS} "${CMD[@]}"
