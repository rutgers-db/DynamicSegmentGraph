#!/usr/bin/env bash
set -euo pipefail

# 2026-01-11 Zhencan Peng: dynamic workload wrapper for insert + query.
#
# This script runs the *dynamic insert+query workload*:
# - Load a partial index built by scripts/dynamic/run_build_dynamic_index.sh
# - Insert remaining labels from the saved remaining-label file
# - Query and report recall/QPS just like the static query workload

usage() {
  cat <<'EOF'
Usage: run_insert_and_query_dynamic_index.sh [SEARCH_EF]

Optional SEARCH_EF overrides the search ef passed to insert_and_query_dynamic_index.
EOF
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/insert_and_query_dynamic_index"

configure_and_build() {
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --target insert_and_query_dynamic_index
}

if [[ ! -x "${BIN}" ]]; then
  echo "[DSG] insert_and_query_dynamic_index binary not found, building..."
  configure_and_build
fi

SEARCH_EF=""
if [[ $# -gt 1 ]]; then
  usage
elif [[ $# -eq 1 ]]; then
  SEARCH_EF="$1"
fi

DATASET="deep"
# DATASET="wikipedia"
DATA_SIZE="100000"
BUILD_RATIO="${BUILD_RATIO:-0.5}"

# Dataset-specific paths
declare -A DEFAULT_DATASET_PATHS=(
  ["deep"]="${ROOT_DIR}/data/deep10M.bin"
  ["wikipedia"]="${ROOT_DIR}/data/wiki_image_embedding.bin"
  ["yt8m-video"]="${ROOT_DIR}/data/yt8m_sorted_by_timestamp_video_embedding_1M.bin"
)
declare -A DEFAULT_QUERY_PATHS=(
  ["deep"]="${ROOT_DIR}/data/deep_query.bin"
  ["wikipedia"]="${ROOT_DIR}/data/wiki_image_query.bin"
  ["yt8m-video"]="${ROOT_DIR}/data/yt8m_video_query_10k.bin"
)

DATASET_PATH="${DEFAULT_DATASET_PATHS[${DATASET}]:-}"
QUERY_PATH="${DEFAULT_QUERY_PATHS[${DATASET}]:-}"

if [[ -z "${DATASET_PATH}" || -z "${QUERY_PATH}" ]]; then
  echo "[DSG] Dataset or query path missing for ${DATASET}." >&2
  exit 1
fi

# Dataset-specific parameters (must match how the partial index was built)
case "${DATASET}" in
  "deep")
    INDEX_K="16"
    EF_CONSTRUCTION="100"
    EF_MAX="300"
    ALPHA="1"
    ;;
  "wikipedia"|"yt8m-video")
    INDEX_K="32"
    EF_CONSTRUCTION="160"
    EF_MAX="600"
    if [[ "${DATASET}" == "yt8m-video" ]]; then
      ALPHA="1.3"
    else
      ALPHA="1.1"
    fi
    ;;
  *)
    echo "[DSG] Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

INDEX_PATH="${ROOT_DIR}/index/dynamic/${DATASET}/${DATASET}_N${DATA_SIZE}_ratio${BUILD_RATIO}_k${INDEX_K}_efc${EF_CONSTRUCTION}_efm${EF_MAX}_alpha${ALPHA}.index"
REMAINING_LABELS_PATH="${ROOT_DIR}/index/dynamic/${DATASET}/${DATASET}_N${DATA_SIZE}_ratio${BUILD_RATIO}.remaining_labels.bin"
GROUND_ROOT="${ROOT_DIR}/groundtruth/static"
QUERY_NUM="1000"
QUERY_K="10"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "[DSG] Partial index file ${INDEX_PATH} not found. Build it first." >&2
  exit 1
fi
if [[ ! -f "${REMAINING_LABELS_PATH}" ]]; then
  echo "[DSG] Remaining labels file ${REMAINING_LABELS_PATH} not found. Build it first." >&2
  exit 1
fi

CMD_ARGS=(
  -dataset "${DATASET}"
  -N "${DATA_SIZE}"
  -dataset_path "${DATASET_PATH}"
  -query_path "${QUERY_PATH}"
  -index_path "${INDEX_PATH}"
  -remaining_labels_path "${REMAINING_LABELS_PATH}"
  -groundtruth_root "${GROUND_ROOT}"
  -query_num "${QUERY_NUM}"
  -query_k "${QUERY_K}"
)

if [[ -n "${SEARCH_EF}" ]]; then
  CMD_ARGS+=(-search_ef "${SEARCH_EF}")
fi

# Create log directory and file
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/dynamic/${DATASET}/insert_query"
mkdir -p "${LOG_DIR}"

LOG_FILENAME="${DATASET}_N${DATA_SIZE}_ratio${BUILD_RATIO}_k${INDEX_K}_efc${EF_CONSTRUCTION}_efm${EF_MAX}_alpha${ALPHA}"
if [[ -n "${SEARCH_EF}" ]]; then
  LOG_FILENAME="${LOG_FILENAME}_ef${SEARCH_EF}"
fi
LOG_FILENAME="${LOG_FILENAME}_${TIMESTAMP}.log"
LOG_PATH="${LOG_DIR}/${LOG_FILENAME}"

echo "[DSG] Logging output to ${LOG_PATH}"
"${BIN}" "${CMD_ARGS[@]}" | tee "${LOG_PATH}"

