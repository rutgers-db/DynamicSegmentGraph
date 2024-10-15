#!/bin/bash

# Define root directory and N

N=10000 #000
index_k=8
ef_max=500
ef_construction=100
METHOD="compact"
VERSIONS=("0_0" "0_1" )   #"1_0" "1_1"                                         
root_path="/research/projects/zp128/RangeIndexWithRandomInsertion/" # Define the root path

# List of datasets
DATASETS=("deep")

# List of dataset paths with root_path appended
DATASET_PATHS=("${root_path}data/deep_sorted_10M.fvecs")

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

    # Iterate over methods and run the benchmark
    for VERSION in "${VERSIONS[@]}"; do
        # Define index path and log file
        INDEX_PATH="${root_path}opt_index/${dataset}/${INDEX_SIZE}_${VERSION}"
        LOG_PATH="${INDEX_PATH}/${index_k}_${ef_max}_${ef_construction}_${VERSION}.log"
        # Create the index path directory if it does not exist
        if [ ! -d "$INDEX_PATH" ]; then
            mkdir -p "$INDEX_PATH"
        fi

        echo "Running benchmark for dataset: $dataset, method: $METHOD version:$VERSION  "
        echo "./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $dataset -method $METHOD -dataset_path $dataset_path -index_path $INDEX_PATH" -op_version "$VERSION"

        ./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max \
            -dataset $dataset -method $METHOD -dataset_path "$dataset_path" -index_path "$INDEX_PATH" -op_version "$VERSION" >>"$LOG_PATH"
    done

    break
done

exit 0
