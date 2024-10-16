#!/bin/bash

# Define root directory, dataset, and other variables
ROOT_DIR="/research/projects/zp128/RangeIndexWithRandomInsertion/"
N=10000
index_k=8
ef_max=500
ef_construction=100

# Define dataset paths
# List of datasets #
DATASETS=(
  "wiki-image" "deep" "yt8m-video") # "yt8m-video"

# List of dataset paths with root_path appended
DATASET_PATHS=(
  "${root_path}data/wiki_image_embedding.fvecs" 
  "${root_path}data/deep_sorted_10M.fvecs" 
  "${root_path}data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs") #

QUERY_PATHS=("${root_path}data/wiki_image_querys.fvecs" "${root_path}data/deep1B_queries.fvecs" "${root_path}data/yt8m_video_querys_10k.fvecs") # Example: wiki_image_querys

GROUNDTRUTH_PATHs=("../groundtruth/wiki_image_benchmark-groundtruth-deep-1k-num1000-k10.arbitrary.cvs"
  "../groundtruth/deep_benchmark-groundtruth-deep-1k-num1000-k10.arbitrary.cvs"
  "../groundtruth/yt8m-video_benchmark-groundtruth-deep-1k-num1000-k10.arbitrary.cvs")

# Define methods and versions
METHODS=("compact") # Example: "Seg2D"
VERSIONS=("0_0" "0_1" "1_0" "1_1")

# Iterate over methods and versions
for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
  dataset="${DATASETS[$i]}"
  dataset_path="${DATASET_PATHS[$i]}"
  query_path="${QUERY_PATHS[$i]}"
  GROUNDTRUTH_PATH="${GROUNDTRUTH_PATHs[$i]}"

  # Determine index size suffix
  if [ "$N" -ge 1000000 ]; then
    INDEX_SIZE="$((N / 1000000))m"
  else
    INDEX_SIZE="$((N / 1000))k"
  fi

  for VERSION in "${VERSIONS[@]}"; do
    for index_k in "${index_k_arr[@]}"; do
      for ef_max in "${ef_max_arr[@]}"; do
        for ef_construction in "${ef_construction_arr[@]}"; do
          # Define index path and log path
          INDEX_PATH="${ROOT_DIR}opt_index/${DATASET}/${INDEX_SIZE}_${VERSION}"
          LOG_PATH="${INDEX_PATH}/${index_k}_${ef_max}_${ef_construction}_${VERSION}_query.log"

          # Create the index directory if it doesn't exist
          if [ ! -d "$INDEX_PATH" ]; then
            echo "Creating directory: $INDEX_PATH"
            mkdir -p "$INDEX_PATH"
          fi

          # Construct the command string for display
          CMD="./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
            -dataset $DATASET -method $METHOD -dataset_path \"$DATASET_PATH\" -query_path \"$QUERY_PATH\" \
            -groundtruth_path \"$GROUNDTRUTH_PATH\" -index_path \"$INDEX_PATH/compact_${index_k}_${ef_max}_${ef_construction}.bin\""

          # Display the command being executed
          echo "Running query for:"
          echo "  Dataset      : $DATASET"
          echo "  Method       : $METHOD"
          echo "  Version      : $VERSION"
          echo "  index_k      : $index_k"
          echo "  ef_max       : $ef_max"
          echo "  ef_construction: $ef_construction"
          echo "  Index Path   : $INDEX_PATH"
          echo "  Log File     : $LOG_PATH"
          echo "Executing: $CMD"

          # Execute the command and log the output
          ./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
            -dataset $DATASET -method $METHOD -dataset_path "$DATASET_PATH" -query_path "$QUERY_PATH" \
            -groundtruth_path "$GROUNDTRUTH_PATH" -index_path "${INDEX_PATH}/compact_${index_k}_${ef_max}_${ef_construction}.bin" >>"$LOG_PATH" 2>&1

          echo "Finished query for version: $VERSION with index_k: $index_k, ef_max: $ef_max, ef_construction: $ef_construction"
          echo "------------------------------------------------------"
        done
      done
    done
  done
done

exit 0
