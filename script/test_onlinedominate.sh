#!/bin/zsh

# Define root directory, N, dataset, and method as variables
LOG_DIR="/research/projects/zp128/RangeIndexWithRandomInsertion/"
N=1000000
DATASETS=("yt8m-video" "wiki-image" "deep")  # List of datasets
DATASET_PATHS=("../data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs" "../data/wiki_image_embedding.fvecs" "../data/deep1B_queries.fvecs")  # List of dataset paths
QUERY_PATHS=("../data/yt8m_video_querys_10k.fvecs" "../data/wiki_image_querys.fvecs" "../data/deep10M.fvecs")  # List of query paths
GROUNDTRUTH_PATHS=("../groundtruth/yt8m_video_benchmark-groundtruth-deep-1m-num1000-k10.fullrange.cvs" "../groundtruth/wiki_image_benchmark-groundtruth-deep-1m-num1000-k10.fullrange.cvs" "../groundtruth/deep_benchmark-groundtruth-deep-1m-num1000-k10.fullrange.cvs")  # List of groundtruth paths

# Iterate over datasets and corresponding paths
for i in {1..${#DATASETS[@]}}; do
  dataset="${DATASETS[$i-1]}"
  dataset_path="${DATASET_PATHS[$i-1]}"
  query_path="${QUERY_PATHS[$i-1]}"
  groundtruth_path="${GROUNDTRUTH_PATHS[$i-1]}"

  # Calculate index size based on N
  if [ $N -ge 1000000 ]; then
    INDEX_SIZE="$(($N / 1000000))m"
  else
    INDEX_SIZE="$(($N / 1000))k"
  fi

  # Define index path log file
  INDEX_PATH="$LOG_DIR/${dataset}_${INDEX_SIZE}_build.log"

  # Run the benchmark
  ./benchmark/test_onlinedomination -N $N -dataset $dataset -dataset_path $dataset_path -query_path $query_path -groundtruth_path $groundtruth_path

done

exit 0
