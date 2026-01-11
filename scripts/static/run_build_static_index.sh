#!/usr/bin/env bash
set -euo pipefail

# 2026-01-10 Zhencan Peng: static workload wrapper for building a static index.
#
# This script runs the *static workload* (build a fixed index for a fixed dataset).
# Future dynamic workload scripts should live under scripts/dynamic/.

ROOT_DIR="/common/users/zp128/DynamicSegmentGraph"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/build_static_index"

configure_and_build() {
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --target build_static_index
}

if [[ ! -x "${BIN}" ]]; then
  echo "[DSG] build_static_index binary not found, building..."
  configure_and_build
fi

DATASET="deep"
# DATASET="wikipedia"
DATA_SIZE="100000"

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

# Dataset-specific parameters
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

INDEX_PATH="${ROOT_DIR}/index/static/${DATASET}/${DATASET}_N${DATA_SIZE}_k${INDEX_K}_efc${EF_CONSTRUCTION}_efm${EF_MAX}_alpha${ALPHA}.index"
mkdir -p "$(dirname "${INDEX_PATH}")"

# Create log directory and file
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/static/${DATASET}/build"
mkdir -p "${LOG_DIR}"

# Build log filename with parameters
LOG_FILENAME="${DATASET}_N${DATA_SIZE}_k${INDEX_K}_efc${EF_CONSTRUCTION}_efm${EF_MAX}_alpha${ALPHA}_${TIMESTAMP}.log"
LOG_PATH="${LOG_DIR}/${LOG_FILENAME}"

echo "[DSG] Running build_static_index..."
echo "[DSG] Logging output to ${LOG_PATH}"
/usr/bin/time -v "${BIN}" \
  -dataset "${DATASET}" \
  -N "${DATA_SIZE}" \
  -dataset_path "${DATASET_PATH}" \
  -query_path "${QUERY_PATH}" \
  -index_path "${INDEX_PATH}" \
  -k "${INDEX_K}" \
  -ef_construction "${EF_CONSTRUCTION}" \
  -ef_max "${EF_MAX}" \
  -alpha "${ALPHA}" | tee "${LOG_PATH}"

