#!/bin/bash -l

#SBATCH --job-name=baseline_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constrain=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH --time=00:50:00
#SBATCH --partition=batch
#SBATCH --output=/.../LAViG-FLOW/benchmarking/benchmark_comparison/logs/eval_baselines.%j.out
#SBATCH --error=/.../LAViG-FLOW/benchmarking/benchmark_comparison/logs/eval_baselines.%j.err
set -euo pipefail

HOST_PROJECT_ROOT=${PROJECT_ROOT:-/...}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
BENCH_ROOT=${BENCH_ROOT:-${REPO_ROOT}/benchmarking}
COMPARE_ROOT="${BENCH_ROOT}/benchmark_comparison"

mkdir -p ${COMPARE_ROOT}/logs ${COMPARE_ROOT}/results

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_PATH=${CONFIG_PATH:-${COMPARE_ROOT}/config.yaml}
OUTPUT_JSON=${OUTPUT_JSON:-${COMPARE_ROOT}/results/results.json}
OUTPUT_TEX=${OUTPUT_TEX:-${COMPARE_ROOT}/results/tables.tex}
LAVIG_ROOT=${LAVIG_ROOT:-${REPO_ROOT}}

CMD=(python ${COMPARE_ROOT}/eval_baselines.py
    --config "${CONFIG_PATH}"
    --output-json "${OUTPUT_JSON}"
    --output-tex "${OUTPUT_TEX}"
    --lavig-root "${LAVIG_ROOT}")

srun -n ${SLURM_NTASKS} "${CMD[@]}"
