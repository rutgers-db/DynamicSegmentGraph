#!/bin/bash

# ./benchmark/generate_groundtruth -N 10000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_prefix ../groundtruth/deep_

#  ./benchmark/generate_groundtruth -dataset yt8m-audio -N 1000000 -dataset_path ../data/yt8m_audio_embedding.fvecs -query_path ../data/yt8m_audio_querys_10k.fvecs -groundtruth_prefix ../groundtruth/yt8m_
 
#  ./benchmark/generate_groundtruth -dataset wiki-image -N 1000000 -dataset_path ../data/wiki_image_embedding.fvecs -query_path ../data/wiki_image_querys.fvecs -groundtruth_prefix ../groundtruth/wiki_image_

# ./benchmark/generate_groundtruth -N 8000000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_prefix ../groundtruth/deep_

# ./benchmark/generate_groundtruth -dataset deep10m -N 8000000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_prefix ../groundtruth/deep_

 ./benchmark/generate_groundtruth -dataset yt8m-video -N 1000000 -dataset_path ../data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs -query_path ../data/yt8m_video_querys_10k.fvecs -groundtruth_prefix ../groundtruth/yt8m_video_


# Exit the script
exit 0
