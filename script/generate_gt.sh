#!/bin/bash

# Define root directory, N, dataset, and method as variables
Ns=(1000000 ) #50000 100000 150000 200000
data_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/data/"
gt_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/"

# List of datasets
DATASETS=("wiki-image" "yt8m-video" "deep") #

# List of dataset paths with data_root_path appended
DATASET_PATHS=(
    "${data_root_path}wiki_image_embedding.fvecs"
    "${data_root_path}yt8m_sorted_by_timestamp_video_embedding_1M.fvecs"
    "${data_root_path}deep_sorted_10M.fvecs"
) #    

QUERY_PATHS=(
    "${data_root_path}wiki_image_querys.fvecs"
    "${data_root_path}yt8m_video_querys_10k.fvecs"
    "${data_root_path}deep1B_queries.fvecs"
) #    

# Iterate over datasets
for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
    dataset="${DATASETS[$i]}"
    dataset_path="${DATASET_PATHS[$i]}"
    query_path="${QUERY_PATHS[$i]}"
    # Iterate over Ns
    for j in "${!Ns[@]}"; do
        N="${Ns[$j]}"

        # Calculate index size (in 'k' or 'm' format)
        if [ "$N" -ge 1000000 ]; then
            INDEX_SIZE="$(($N / 1000000))m"
        else
            INDEX_SIZE="$(($N / 1000))k"
        fi

        # Groundtruth path prefix
        gt_pathprefix="${gt_root_path}${dataset}_"

        # Print the command being executed for debugging
        echo "./benchmark/generate_groundtruth -dataset $dataset -N $N -dataset_path $dataset_path -query_path $query_path -groundtruth_prefix $gt_pathprefix"

        # Execute the command
        ./benchmark/generate_groundtruth \
            -dataset "$dataset" \
            -N "$N" \
            -dataset_path "$dataset_path" \
            -query_path "$query_path" \
            -groundtruth_prefix "$gt_pathprefix" >>"/research/projects/zp128/RangeIndexWithRandomInsertion/log/${dataset}_${INDEX_SIZE}_gt.log"

    done
done

exit 0


