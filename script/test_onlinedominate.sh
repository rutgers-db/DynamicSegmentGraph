
#!/bin/zsh

./benchmark/test_onlinedomination -N 1000000 -dataset yt8m-audio -dataset_path ../data/yt8m_audio_embedding.fvecs -query_path ../data/yt8m_audio_querys_10k.fvecs -groundtruth_path ../groundtruth/yt8m_benchmark-groundtruth-deep-1m-num1000-k10.fullrange.cvs
# Exit the script
exit 0
