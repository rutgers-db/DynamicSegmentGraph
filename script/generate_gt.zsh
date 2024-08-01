#!/bin/zsh

 ./benchmark/generate_groundtruth -N 1000000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_prefix ../groundtruth/deep_
 
# Exit the script
exit 0
