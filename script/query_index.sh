#!/bin/bash

# Define root directory, N, dataset, and method as variables

N=1000000
KS=(16 ) #32 48 64
ef_max=1000
# index_k=8
# ef_max=500
ef_construction=300
METHODS=("compact") #"Seg2D" "compact"
root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/"

# List of datasets
DATASETS=("deep" "yt8m-video" "wiki-image" "yt8m-audio")

# List of dataset paths with root_path appended
DATASET_PATHS=("${root_path}data/deep_sorted_10M.fvecs"
  "${root_path}data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs"
  "${root_path}data/wiki_image_embedding.fvecs"
  "${root_path}data/yt8m_audio_embedding.fvecs")

# List of query paths with root_path appended
QUERY_PATHS=("${root_path}data/deep1B_queries.fvecs"
  "${root_path}data/yt8m_video_querys_10k.fvecs"
  "${root_path}data/wiki_image_querys.fvecs"
  "${root_path}data/yt8m_audio_querys_10k.fvecs")

# List of groundtruth paths with root_path appended
GROUNDTRUTH_PATHS=("${root_path}groundtruth/deep_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs"              # "${root_path}groundtruth/deep_compare_prefilter-1m-num1000-k10.arbitrary.cvs"
  "${root_path}groundtruth/yt8m_video_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs"
  "${root_path}groundtruth/wiki_image_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs"
  "${root_path}groundtruth/yt8m_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs")

# Iterate over methods and query index
for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
  dataset="${DATASETS[$i]}"
  dataset_path="${DATASET_PATHS[$i]}"
  query_path="${QUERY_PATHS[$i]}"
  gt_path="${GROUNDTRUTH_PATHS[$i]}"

  for METHOD in "${METHODS[@]}"; do
    if [ $N -ge 1000000 ]; then
      INDEX_SIZE="$(($N / 1000000))m"
    else
      INDEX_SIZE="$(($N / 1000))k"
    fi
    for index_k in "${KS[@]}"; do
      INDEX_PATH="${root_path}test_index/$dataset/$INDEX_SIZE/${METHOD}_${index_k}_${ef_max}_${ef_construction}.bin"
      LOG_PATH="${root_path}test_exp/${METHOD}_${index_k}_${ef_max}_${ef_construction}.log"
      echo ./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $dataset -method $METHOD -dataset_path $dataset_path -query_path $query_path -groundtruth_path $gt_path -index_path $INDEX_PATH
      ./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $dataset -method $METHOD -dataset_path $dataset_path -query_path $query_path -groundtruth_path $gt_path -index_path "$INDEX_PATH" >> $LOG_PATH
    done
  done
  break
done
exit 0
