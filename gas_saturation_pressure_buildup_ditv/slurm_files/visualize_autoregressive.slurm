#!/bin/bash -l

#SBATCH --job-name=joint-visualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=260GB
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --output=/ibex/project/c2315/vittoria/LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/visualize_autoreg.%j.out
#SBATCH --error=/ibex/project/c2315/vittoria/LAViG-FLOW/gas_saturation_pressure_buildup_ditv/logs_autoreg/visualize_autoreg.%j.err
set -euo pipefail
HOST_PROJECT_ROOT=${PROJECT_ROOT:-/ibex/project/c2315/vittoria}
REPO_ROOT=${REPO_ROOT:-${HOST_PROJECT_ROOT}/LAViG-FLOW}
JOINT_ROOT="${REPO_ROOT}/gas_saturation_pressure_buildup_ditv"


mkdir -p ${JOINT_ROOT}/logs_autoreg

source activate lavig-flow

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CHUNK_LIST=${CHUNK_LIST:-"4"} # chunks should be from 1 to 4. Run the scripts 4 times, so you have the results for all the chunks.
#SAMPLE_INDEX_LIST=$(seq -s, 0 10) # we plot from sample 0 to sample 10
SAMPLE_INDEX_LIST=${SAMPLE_INDEX_LIST:-"8,12,15,19,51,55,67,155,159,168,194,195,209,211,233,239,247,263,268,271,368,399,406,420,475,488"}
OUTPUT_DIR=${OUTPUT_DIR:-${JOINT_ROOT}/pseudo_conditional_generation_autoreg}

IFS=',' read -ra _CHUNKS <<< "${CHUNK_LIST}"
IFS=',' read -ra _SAMPLE_INDICES <<< "${SAMPLE_INDEX_LIST}"

for CHUNK_COUNT in "${_CHUNKS[@]}"; do
    if [ -z "${CHUNK_COUNT}" ]; then
        continue
    fi
    for SAMPLE_IDX in "${_SAMPLE_INDICES[@]}"; do
        if [ -z "${SAMPLE_IDX}" ]; then
            continue
        fi
        CMD=(python ${JOINT_ROOT}/visualize_autoregressive_sample.py
            --config ${JOINT_ROOT}/joint_eval.yaml
            --sample-index ${SAMPLE_IDX}
            --chunks ${CHUNK_COUNT}
            --output-dir "${OUTPUT_DIR}")

        srun -n ${SLURM_NTASKS} "${CMD[@]}"
    done
done
