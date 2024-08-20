#!/bin/zsh

#  ./benchmark/generate_groundtruth -dataset yt8m-audio -N 1000000 -dataset_path ../data/yt8m_audio_embedding.fvecs -query_path ../data/yt8m_audio_querys_10k.fvecs -groundtruth_prefix ../groundtruth/yt8m_
 
 ./benchmark/generate_groundtruth -dataset wiki-image -N 1000000 -dataset_path ../data/wiki_image_embedding.fvecs -query_path ../data/wiki_image_querys.fvecs -groundtruth_prefix ../groundtruth/wiki_image_
# Exit the script
exit 0
