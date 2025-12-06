#!/bin/bash
keep-job 72
./benchmark/scalibility -dataset deep -dataset_path ../data/deep_sorted_10M.fvecs -query_path ../data/deep1B_queries.fvecs >> /research/projects/zp128/RangeIndexWithRandomInsertion/log/scalibility/deep.log

# ./benchmark/scalibility -dataset wiki-image -k 32 -ef_construction 300  -ef_max 1000 -dataset_path ../data/wiki_image_embedding.fvecs -query_path ../data/wiki_image_querys.fvecs >> /research/projects/zp128/RangeIndexWithRandomInsertion/log/scalibility/wiki.log

# ./benchmark/scalibility -dataset yt8m-video -k 32 -ef_construction 300  -ef_max 1000 -alpha 1.2 -dataset_path ../data/exp2_used_data/yt8m_video_1_2m.fvecs -query_path ../data/yt8m_video_querys_10k.fvecs >> /research/projects/zp128/RangeIndexWithRandomInsertion/log/scalibility/yt8m.log
exit 0
