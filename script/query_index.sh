
#!/bin/bash

# Define root directory, N, dataset, and method as variables
ROOT_DIR="/research/projects/zp128/RangeIndexWithRandomInsertion/index"
N=1000000
DATASET="yt8m-video"  # wiki-image
# index_k=16
# ef_max=750
index_k=8
ef_max=500
ef_construction=100
DATASET_PATH="../data/yt8m_sorted_by_timestamp_video_embedding_1M.fvecs" #yt8m_audio_embedding.fvecs
QUERY_PATH="../data/yt8m_video_querys_10k.fvecs"  # wiki_image_querys
GROUNDTRUTH_PATH="../groundtruth/yt8m_video_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs" #yt8m_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs wiki_image_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary
METHODS=("Seg2D" "compact")

# Iterate over methods and query index
for METHOD in "${METHODS[@]}"; do
  if [ $N -ge 1000000 ]; then
    INDEX_SIZE="$(($N / 1000000))m"
  else
    INDEX_SIZE="$(($N / 1000))k"
  fi
  INDEX_PATH="$ROOT_DIR/$DATASET/$INDEX_SIZE/${METHOD}_${index_k}_${ef_max}_${ef_construction}.bin"

  ./benchmark/query_index -N $N -k $index_k -ef_construction $ef_construction -ef_max $ef_max -dataset $DATASET -method $METHOD -dataset_path $DATASET_PATH -query_path $QUERY_PATH -groundtruth_path $GROUNDTRUTH_PATH -index_path "$INDEX_PATH"
done

exit 0

# #!/bin/bash

# # ./benchmark/query_index -N 100000 -method Seg2D -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_path ../groundtruth/deep_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/deep/100k/Seg2D_8_500_100.bin
# # ./benchmark/query_index -N 100000 -method compact -dataset_path ../data/deep10M.fvecs -query_path ../data/deep1B_queries.fvecs -groundtruth_path ../groundtruth/deep_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/deep/100k/compact_8_500_100.bin
# # ./benchmark/query_index -N 100000 -dataset wiki-image -method Seg2D -dataset_path ../data/wiki_image_embedding.fvecs  -query_path ../data/wiki_image_querys.fvecs -groundtruth_path ../groundtruth/wiki_image_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs  -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/wiki/100k/Seg2D_8_500_100.bin
# # ./benchmark/query_index -N 100000 -dataset wiki-image -method compact -dataset_path ../data/wiki_image_embedding.fvecs -query_path ../data/wiki_image_querys.fvecs -groundtruth_path ../groundtruth/wiki_image_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs  -index_path /research/projects/zp128/RangeIndexWithRandomInsertion/index/wiki/100k/compact_8_500_100.bin

# exit 0
