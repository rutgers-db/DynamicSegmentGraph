#!/usr/bin/env bash
set -euo pipefail

# 2026-01-10 Zhencan Peng: static workload wrapper for groundtruth generation.
#
# This script generates groundtruth files under groundtruth/static/.
# Future dynamic workload scripts should live under scripts/dynamic/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/generate_groundtruth_static"

if [[ ! -x "${BIN}" ]]; then
  echo "Binary ${BIN} not found. Build it first with:" >&2
  echo "  cmake -S ${ROOT_DIR} -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} --target generate_groundtruth_static" >&2
  exit 1
fi

GROUND_DIR="${GROUND_DIR:-${ROOT_DIR}/groundtruth/static}"
DATA_SIZE="1000000"
declare -A DEFAULT_DATASET_PATHS=(
  ["deep"]="${ROOT_DIR}/data/deep10M.bin"
  ["wikipedia"]="${ROOT_DIR}/data/wiki_image_embedding.bin"
  ["yt8m"]="${ROOT_DIR}/data/yt8m_sorted_by_timestamp_video_embedding_1M.bin"
)
declare -A DEFAULT_QUERY_PATHS=(
  ["deep"]="${ROOT_DIR}/data/deep_query.bin"
  ["wikipedia"]="${ROOT_DIR}/data/wiki_image_query.bin"
  ["yt8m"]="${ROOT_DIR}/data/yt8m_video_query_10k.bin"
)

datasets=("deep" "wikipedia" "yt8m")

for dataset in "${datasets[@]}"; do
  dataset_path="${DEFAULT_DATASET_PATHS[${dataset}]:-}"
  query_path="${DEFAULT_QUERY_PATHS[${dataset}]:-}"

  if [[ -z "${dataset_path}" || -z "${query_path}" ]]; then
    echo "Dataset or query path missing for ${dataset}." >&2
    exit 1
  fi

  echo "Running groundtruth generation (static) for dataset ${dataset}..."
  "${BIN}" \
    -dataset "${dataset}" \
    -N "${DATA_SIZE}" \
    -dataset_path "${dataset_path}" \
    -query_path "${query_path}" \
    -groundtruth_root "${GROUND_DIR}"
done

