#!/bin/bash

# Define root directory, N, dataset, and method as variables
Ns=(50000 100000 150000 200000)
data_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/data/exp2_used_data/"
gt_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/exp2_used/"

# List of datasets
DATASETS=("wiki-image" "deep" "yt8m-video")

# List of dataset paths with data_root_path appended
DATASET_PATHS=(
    "${data_root_path}wiki_image_additional200K.fvecs"
    "${data_root_path}deep_additional200K.fvecs"
    "${data_root_path}yt8m_video_additional200K.fvecs"
)

QUERY_PATHS_PREFIX=(
    "${data_root_path}wiki_image_querys"
    "${data_root_path}deep_querys"
    "${data_root_path}yt8m_video_querys"
)

# Iterate over datasets
for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
    dataset="${DATASETS[$i]}"
    dataset_path="${DATASET_PATHS[$i]}"
    query_path_prefix="${QUERY_PATHS_PREFIX[$i]}"
    # Iterate over Ns
    for j in "${!Ns[@]}"; do
        N="${Ns[$j]}"
        query_path="${query_path_prefix}_${j}.fvecs"

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
