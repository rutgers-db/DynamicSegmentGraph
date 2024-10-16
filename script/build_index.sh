#!/bin/bash

# Define root directory and N

N=10000 #000
METHOD="compact"
VERSIONS=("0_0" "0_1" "1_0" "1_1")                                  #
root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/" # Define the root path

# List of datasets #
DATASETS=(
    "wiki-image" "deep" "yt8m-video") # "yt8m-video"

# List of dataset paths with root_path appended
DATASET_PATHS=(
    "${root_path}data/wiki_image_embedding.fvecs" "${root_path}data/deep_sorted_10M.fvecs" "${root_path}data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs") #

# Define the arrays for index_k, ef_max, and ef_construction
index_k_arr=(4 8 16 32)
ef_max_arr=(400)
ef_construction_arr=(100)
# ef_max_arr=(400 600 800 1000)
# ef_construction_arr=(100 200 300 400)

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

    # Iterate over methods, index_k, ef_max, and ef_construction combinations and run the benchmark
    for VERSION in "${VERSIONS[@]}"; do
        for index_k in "${index_k_arr[@]}"; do
            for ef_max in "${ef_max_arr[@]}"; do
                for ef_construction in "${ef_construction_arr[@]}"; do
                    # Define index path and log file
                    INDEX_PATH="${root_path}opt_index/${dataset}/${INDEX_SIZE}_${VERSION}"
                    LOG_PATH="${INDEX_PATH}/${index_k}_${ef_max}_${ef_construction}_${VERSION}.log"
                    # Create the index path directory if it does not exist
                    if [ ! -d "$INDEX_PATH" ]; then
                        mkdir -p "$INDEX_PATH"
                    fi

                    echo "Running benchmark for dataset: $dataset, method: $METHOD version:$VERSION index_k: $index_k ef_max: $ef_max ef_construction: $ef_construction"
                    echo "./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $dataset -method $METHOD -dataset_path $dataset_path -index_path $INDEX_PATH" -op_version "$VERSION"

                    ./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
                        -dataset $dataset -method $METHOD -dataset_path "$dataset_path" -index_path "$INDEX_PATH" -op_version "$VERSION" >>"$LOG_PATH"
                done
            done
        done
    done
done

exit 0
