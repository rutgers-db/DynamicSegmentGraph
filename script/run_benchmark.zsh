#!/bin/zsh

./benchmark/deep_arbitrary -N 100000 -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_path ../groundtruth/deep_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs 
 
# Exit the script
exit 0
