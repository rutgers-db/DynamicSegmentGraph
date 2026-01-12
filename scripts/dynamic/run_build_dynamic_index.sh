#!/usr/bin/env bash
set -euo pipefail

# 2026-01-11 Zhencan Peng: dynamic workload wrapper for building an initial index.
#
# This script runs the *dynamic build workload*:
# - Build an initial DSG index from a random subset of labels (default: half).
# - Persist the remaining labels to disk for later insertion/query workloads.

ROOT_DIR="/common/users/zp128/DynamicSegmentGraph"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/build_dynamic_index"

configure_and_build() {
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --target build_dynamic_index
}

if [[ ! -x "${BIN}" ]]; then
  echo "[DSG] build_dynamic_index binary not found, building..."
  configure_and_build
fi

# DATASET="${DATASET:-deep}"
DATASET="wikipedia"
DATA_SIZE="${DATA_SIZE:-100000}"
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

if [[ -z "${DATASET_PATH}" ]]; then
  echo "[DSG] Dataset path missing for ${DATASET}." >&2
  exit 1
fi

# Dataset-specific parameters (same defaults as static scripts)
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

OUT_DIR="${ROOT_DIR}/index/dynamic/${DATASET}"
mkdir -p "${OUT_DIR}"

INDEX_PATH="${OUT_DIR}/${DATASET}_N${DATA_SIZE}_ratio${BUILD_RATIO}_k${INDEX_K}_efc${EF_CONSTRUCTION}_efm${EF_MAX}_alpha${ALPHA}.index"
REMAINING_LABELS_PATH="${OUT_DIR}/${DATASET}_N${DATA_SIZE}_ratio${BUILD_RATIO}.remaining_labels.bin"

# Create log directory and file
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/dynamic/${DATASET}/build"
mkdir -p "${LOG_DIR}"
LOG_FILENAME="${DATASET}_N${DATA_SIZE}_ratio${BUILD_RATIO}_k${INDEX_K}_efc${EF_CONSTRUCTION}_efm${EF_MAX}_alpha${ALPHA}_${TIMESTAMP}.log"
LOG_PATH="${LOG_DIR}/${LOG_FILENAME}"

echo "[DSG] Running build_dynamic_index..."
echo "[DSG] Logging output to ${LOG_PATH}"
/usr/bin/time -v "${BIN}" \
  -dataset "${DATASET}" \
  -N "${DATA_SIZE}" \
  -dataset_path "${DATASET_PATH}" \
  -index_path "${INDEX_PATH}" \
  -remaining_labels_path "${REMAINING_LABELS_PATH}" \
  -k "${INDEX_K}" \
  -ef_construction "${EF_CONSTRUCTION}" \
  -ef_max "${EF_MAX}" \
  -alpha "${ALPHA}" \
  -build_ratio "${BUILD_RATIO}" \
  ${QUERY_PATH:+ -query_path "${QUERY_PATH}"} | tee "${LOG_PATH}"

echo "[DSG] Index: ${INDEX_PATH}"
echo "[DSG] Remaining labels: ${REMAINING_LABELS_PATH}"

