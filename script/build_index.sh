#!/bin/bash

# Define root directory, N, dataset, and method as variables
ROOT_DIR="/research/projects/zp128/RangeIndexWithRandomInsertion/index"
N=1000000
index_k=16
ef_max=750
ef_construction=100
DATASET="yt8m-video" # wiki-image deep  yt8m-audio
DATASETNAME="yt8m-video"  # wiki deep yt8m-audio
DATASET_PATH="../data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs" # ../data/deep10M.fvecs ../data/wiki_image_embedding.fvecs yt8m_sorted_by_timestamp_video_embedding_1M yt8m_audio_embedding
METHODS=("Seg2D" "compact")

# Iterate over methods and build index
for METHOD in "${METHODS[@]}"; do
    if [ $N -ge 1000000 ]; then
        INDEX_SIZE="$(($N / 1000000))m"
    else
        INDEX_SIZE="$(($N / 1000))k"
    fi
    INDEX_PATH="$ROOT_DIR/$DATASET/$INDEX_SIZE"
    
    # Create index path directory if it does not exist
    if [ ! -d "$INDEX_PATH" ]; then
        mkdir -p "$INDEX_PATH"
    fi

    ./benchmark/build_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $DATASET -method $METHOD -dataset_path $DATASET_PATH -index_path "$INDEX_PATH"
done

exit 0


# ./benchmark/build_index -N 100000 -method Seg2D -dataset_path ../data/deep10M.fvecs -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/deep/100k
# ./benchmark/build_index -N 100000 -method compact -dataset_path ../data/deep10M.fvecs -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/deep/100k
# ./benchmark/build_index -N 100000 -dataset wiki-image -method Seg2D -dataset_path ../data/wiki_image_embedding.fvecs -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/wiki/100k
# ./benchmark/build_index -N 100000 -dataset wiki-image -method compact -dataset_path ../data/wiki_image_embedding.fvecs -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/wiki/100k