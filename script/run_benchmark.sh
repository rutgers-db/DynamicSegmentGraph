#!/bin/bash

./benchmark/deep_arbitrary -N 100000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_path ../groundtruth/deep_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs 

# ./benchmark/deep_arbitrary -N 100000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_path ../groundtruth/deep_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs 
 
# ./benchmark/deep_arbitrary -N 1000000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_path ../groundtruth/deep_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs 

# ./benchmark/deep_arbitrary -N 100000 -dataset yt8m-audio -dataset_path ../data/yt8m_audio_embedding.fvecs -query_path ../data/yt8m_audio_querys_10k.fvecs -groundtruth_path ../groundtruth/yt8m_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs 

# ./benchmark/deep_arbitrary -N 1000000 -dataset yt8m-audio -dataset_path ../data/yt8m_audio_embedding.fvecs -query_path ../data/yt8m_audio_querys_10k.fvecs -groundtruth_path ../groundtruth/yt8m_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs 

# ./benchmark/deep_arbitrary -N 100000 -dataset wiki-image -dataset_path ../data/wiki_image_embedding.fvecs -query_path ../data/wiki_image_querys.fvecs -groundtruth_path ../groundtruth/wiki_image_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs 

# ./benchmark/deep_arbitrary -N 1000000 -dataset wiki-image -dataset_path ../data/wiki_image_embedding.fvecs -query_path ../data/wiki_image_querys.fvecs -groundtruth_path ../groundtruth/wiki_image_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs 
# Exit the script
exit 0
