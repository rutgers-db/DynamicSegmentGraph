#!/bin/bash

# Define root directory, dataset, and other variables
ROOT_DIR="/research/projects/zp128/RangeIndexWithRandomInsertion/"
N=10000
DATASET="yt8m-video"  # Example: "wiki-image"
index_k=8
ef_max=500
ef_construction=100

# Define dataset paths
DATASET_PATH="../data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs"  # Example: yt8m_audio_embedding.fvecs
QUERY_PATH="../data/yt8m_video_querys_10k.fvecs"  # Example: wiki_image_querys
GROUNDTRUTH_PATH="../groundtruth/yt8m-video_benchmark-groundtruth-deep-10k-num1000-k10.arbitrary.cvs"

# Define methods and versions
METHODS=("compact")  # Example: "Seg2D"
VERSIONS=("0_0" "0_1" "1_0" "1_1")

# Iterate over methods and versions
for METHOD in "${METHODS[@]}"; do
  # Determine index size suffix
  if [ "$N" -ge 1000000 ]; then
    INDEX_SIZE="$((N / 1000000))m"
  else
    INDEX_SIZE="$((N / 1000))k"
  fi

  for VERSION in "${VERSIONS[@]}"; do
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
      -groundtruth_path \"$GROUNDTRUTH_PATH\" -index_path \"$INDEX_PATH/compact_8_500_100.bin\""

    # Display the command being executed
    echo "Running query for:"
    echo "  Dataset      : $DATASET"
    echo "  Method       : $METHOD"
    echo "  Version      : $VERSION"
    echo "  Index Path   : $INDEX_PATH"
    echo "  Log File     : $LOG_PATH"
    echo "Executing: $CMD"

    # Execute the command and log the output
    ./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
      -dataset $DATASET -method $METHOD -dataset_path "$DATASET_PATH" -query_path "$QUERY_PATH" \
      -groundtruth_path "$GROUNDTRUTH_PATH" -index_path "${INDEX_PATH}/compact_8_500_100.bin" >>"$LOG_PATH" 2>&1

    echo "Finished query for version: $VERSION"
    echo "------------------------------------------------------"
  done
done

exit 0
