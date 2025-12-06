#!/usr/bin/bash

# DIGRA-only ordered experiment:
# - Build on first 100k points
# - Evaluate QPS/recall on 100k groundtruth
# - Insert next 100k ordered
# - Evaluate QPS/recall on 200k groundtruth

set -euo pipefail

ROOT="/research/projects/zp128/RangeIndexWithRandomInsertion"
BIN="${ROOT}/build/benchmark/ordered_dsg_vs_digra"

# Datasets
DATASETS=("deep" "wiki-image")
DATASET_PATHS=(
  "${ROOT}/data/deep_sorted_10M.fvecs"
  "${ROOT}/data/wiki_image_embedding.fvecs"
)
QUERY_PATHS=(
  "${ROOT}/data/deep1B_queries.fvecs"
  "${ROOT}/data/wiki_image_querys.fvecs"
)

# Hyper-params
EF_MAX_LIST=(500 1000)
# DIGRA ef_construction per dataset: deep=300, wiki-image=400
EF_CON_LIST=(300 400)

TOTAL_N=200000

for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
  dataset="${DATASETS[$i]}"
  dataset_path="${DATASET_PATHS[$i]}"
  query_path="${QUERY_PATHS[$i]}"
  ef_max="${EF_MAX_LIST[$i]}"
  ef_con="${EF_CON_LIST[$i]}"

  LOG_DIR="${ROOT}/log/ordered_dsg_vs_digra/${dataset}"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/N${TOTAL_N}_efm${ef_max}_efc${ef_con}.log"

  {
    echo "Running DIGRA-only ordered benchmark for ${dataset} -> ${LOG_FILE}"
    echo "${BIN} -N ${TOTAL_N} -ef_construction ${ef_con} -ef_max ${ef_max} -dataset ${dataset} -dataset_path ${dataset_path} -query_path ${query_path}"
    stdbuf -oL -eL "${BIN}" -N ${TOTAL_N} -ef_construction ${ef_con} -ef_max ${ef_max} \
      -dataset "${dataset}" -dataset_path "${dataset_path}" -query_path "${query_path}"
  } 2>&1 | grep -Ev --line-buffered '^[[:space:]]*(Inser[t]?[[:space:]]+a[[:space:]]+batch)' | tee -a "${LOG_FILE}"

done

exit 0


