#!/bin/bash

# Iterate over M values 16, 32, and 64
for M in 16 32 64; do
    echo "Running knn_first with -k $M"
    ./benchmark/knn_first -k $M -dataset deep -N 1000000 -dataset_path ../data/deep_sorted_10M.fvecs -query_path ../data/deep1B_queries.fvecs >> ../log/static_knnfirst/Deep_KNNFirst_M${M}.log
done