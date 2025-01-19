#!/bin/bash

# Define root directory and N

N=1000000
ef_construction=300
root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/" # Define the root path

# List of datasets
DATASETS=("deep" "yt8m-video" "wiki-image")
Ks=(16 64 32)  #16 32 32
ef_maxs=(500 2000 1000)

# List of dataset paths with root_path appended
DATASET_PATHS=("${root_path}data/deep_sorted_10M.fvecs"
    "${root_path}data/exp2_used_data/yt8m_video_1_2m.fvecs"
    "${root_path}data/wiki_image_embedding.fvecs")

QUERY_PATHS=(
    "${root_path}data/deep1B_queries.fvecs"
    "${root_path}data/yt8m_video_querys_10k.fvecs"
    "${root_path}data/wiki_image_querys.fvecs"
) #

# Iterate over datasets and their paths using proper indexing
for i in $(seq 1 $((${#DATASETS[@]} - 1))); do
    dataset="${DATASETS[$i]}"
    dataset_path="${DATASET_PATHS[$i]}"
    query_path="${QUERY_PATHS[$i]}"
    index_k="${Ks[$i]}"
    ef_max="${ef_maxs[$i]}"

    # Determine index size suffix
    if [ $N -ge 1000000 ]; then
        INDEX_SIZE="$(($N / 1000000))m"
    else
        INDEX_SIZE="$(($N / 1000))k"
    fi

    # Define index path and log file
    LOG_PATH="${root_path}log/incremental_knnfirst/${dataset}/OuterLog_${INDEX_SIZE}_${index_k}_${ef_max}_${ef_construction}.log"

    # Iterate over methods and run the benchmark

    echo "Running incre_knnf for dataset: $dataset"
    echo "./benchmark/incre_knnf -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $dataset -dataset_path $dataset_path -query_path $query_path"

    ./benchmark/incre_knnf -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
        -dataset $dataset -dataset_path "$dataset_path" -query_path "$query_path" >>"$LOG_PATH"

    break
    keep-job 48
done

exit 0
