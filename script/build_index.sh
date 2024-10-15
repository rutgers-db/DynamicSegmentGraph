#!/bin/bash

# Define root directory and N

N=1000000
index_k=16
ef_max=500
ef_construction=100
METHODS=("compact") #"Seg2D"
root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/" # Define the root path

# List of datasets
DATASETS=("deep" "yt8m-video" "wiki-image" "yt8m-audio")

# List of dataset paths with root_path appended
DATASET_PATHS=("${root_path}data/deep_sorted_10M.fvecs"
    "${root_path}data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs"
    "${root_path}data/wiki_image_embedding.fvecs"
    "${root_path}data/yt8m_audio_embedding.fvecs")

# Iterate over datasets and their paths using proper indexing
for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
    dataset="${DATASETS[$i]}"
    dataset_path="${DATASET_PATHS[$i]}"

    # Determine index size suffix
    if [ $N -ge 1000000 ]; then
        INDEX_SIZE="$(($N / 1000000))m"
    else
        INDEX_SIZE="$(($N / 1000))k"
    fi

    # Define index path and log file
    INDEX_PATH="${root_path}index/${dataset}/${INDEX_SIZE}"
    LOG_PATH="${INDEX_PATH}/${index_k}_${ef_max}_${ef_construction}.log"

    # Create the index path directory if it does not exist
    if [ ! -d "$INDEX_PATH" ]; then
        mkdir -p "$INDEX_PATH"
    fi

    # Iterate over methods and run the benchmark
    for METHOD in "${METHODS[@]}"; do
        echo "Running benchmark for dataset: $dataset, method: $METHOD"
        echo "./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $dataset -method $METHOD -dataset_path $dataset_path -index_path $INDEX_PATH"

        ./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
            -dataset $dataset -method $METHOD -dataset_path "$dataset_path" -index_path "$INDEX_PATH" >>"$LOG_PATH"
    done
    
    # TODO Remove it just for build Deep10M
    break
done

exit 0
