#!/bin/bash

# Ensure the script is executed with an input argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "Where <dataset> is one of: deep, wiki, yt8m"
    exit 1
fi

# Input argument: dataset
USER_DATASET="$1"

# Supported datasets
DATASETS=("wiki-image" "yt8m-video" "deep")

# Map input argument to dataset index
declare -A DATASET_INDEX=(
    ["wiki"]=0
    ["yt8m"]=1
    ["deep"]=2
)

# Validate input dataset
if [[ ! "${DATASET_INDEX[$USER_DATASET]}" ]]; then
    echo "Error: Invalid dataset '$USER_DATASET'. Supported datasets are: ${DATASETS[*]}"
    exit 1
fi

# Define root directory, N, and method variables
Ns=(1000000)
data_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/data/"
gt_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/incremental/"
log_root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/log/"

# Dataset paths
DATASET_PATHS=(
    "${data_root_path}wiki_image_embedding.fvecs"
    "${data_root_path}exp2_used_data/yt8m_video_1_2m.fvecs"
    "${data_root_path}deep_sorted_10M.fvecs"
)

# Query paths
QUERY_PATHS=(
    "${data_root_path}wiki_image_querys.fvecs"
    "${data_root_path}yt8m_video_querys_10k.fvecs"
    "${data_root_path}deep1B_queries.fvecs"
)

# Select dataset-specific variables based on user input
index=${DATASET_INDEX[$USER_DATASET]}
dataset="${DATASETS[$index]}"
dataset_path="${DATASET_PATHS[$index]}"
query_path="${QUERY_PATHS[$index]}"

# Iterate over Ns
for N in "${Ns[@]}"; do

    # Calculate index size (in 'k' or 'm' format)
    if [ "$N" -ge 1000000 ]; then
        INDEX_SIZE="$(($N / 1000000))m"
    else
        INDEX_SIZE="$(($N / 1000))k"
    fi

    # Groundtruth path prefix
    gt_pathprefix="${gt_root_path}${dataset}_"

    # Log file path
    log_file="${log_root_path}${dataset}_${INDEX_SIZE}_incre_gt.log"

    # Print command for debugging
    echo "Executing: ./benchmark/get_incrementalgt -dataset $dataset -N $N -dataset_path $dataset_path -query_path $query_path -groundtruth_prefix $gt_pathprefix"

    # Execute the command and append output to log file
    ./benchmark/get_incrementalgt \
        -dataset "$dataset" \
        -N "$N" \
        -dataset_path "$dataset_path" \
        -query_path "$query_path" \
        -groundtruth_prefix "$gt_pathprefix" >>"$log_file" 2>&1

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Command failed for dataset '$dataset', N=$N. Check the log: $log_file"
        exit 1
    fi

    echo "Completed for dataset '$dataset', N=$N. Logs are in: $log_file"
done

exit 0
