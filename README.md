# Dynamic Range-Filtering Approximate Nearest Neighbor Search

This repository hosts a rebuilt implementation of **Dynamic Segment Graph (DSG)** for range-filtered approximate nearest neighbor search which is published in VLDB25 (Dynamic Range-Filtering Approximate Nearest Neighbor Search). This work is the follow-up to [SeRF](https://github.com/rutgers-db/SeRF) (Segment Graph for Range-Filtering but not supporting random insertion), also developed at the Rutgers Database Lab (**RuDB**).

## Status & Roadmap
- Static build + load: done (DFS compression, CSR storage).
- Insertion of new points: in progress.
- Densification for small query ranges: planned / in progress.
- Reducing the building time: done.

## Quick Start
```bash
mkdir build && cd build
cmake ..
make -j
```

## Workloads

### Static workload (current)
- **CLIs**: `apps/static/` (`build_static_index.cc`, `query_static_index.cc`, `generate_groundtruth_static.cc`)
- **Scripts**: `scripts/static/` (`run_build_static_index.sh`, `run_query_static_index.sh`, `run_generate_groundtruth_static.sh`, `test_sensitivity_static.sh`)
- **Artifacts**:
  - Index: `index/static/<dataset>/...`
  - Logs: `logs/static/<dataset>/...`
  - Groundtruth: `groundtruth/static/`

### Dynamic workload (planned)
- Placeholder directories exist for future work:
  - `apps/dynamic/`
  - `scripts/dynamic/`

## Code Structure (core)
- `include/`: public headers. Core interface lives in `dsg.h`; supporting types (HNSW wrappers, utilities) reside under `include/base_hnsw/` and `include/utils/`.
- `src/`: implementations. The main logic is in `src/dsg.cc`, with shared helpers under `src/utils/`.
- `apps/`: CLI entry points for workloads.
- `scripts/`: helper scripts for common workflows (static now; dynamic planned).

## Datasets
| Dataset | Data type | Dimensions | Search Key |
| :- | :-: | :-: | :-: |
| [DEEP](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) | float | 96 | Synthetic |
| [Youtube-Video](https://research.google.com/youtube8m/download.html) | float | 1024 | Video Release Time |
| [WIT-Image](https://www.kaggle.com/c/wikipedia-image-caption/overview) | float | 2048 | Image Size |

<!-- ## Recommended Hyperparameters (N = 1,000,000)
The following settings are the current recommended defaults for a 1M dataset build for chasing best recall/QPS trade off.

| Dataset (flag) | N | k(M) | ef_construction | ef_max | alpha |
| :- | -: | -: | -: | -: | -: |
| `deep` | 1,000,000 | 24 | 130 | 400 | 1.1 |
| `wikipedia` | 1,000,000 | 32 | 151 | 500 | 1.0 |
| `yt8m-video` | 1,000,000 | 32 | 151 | 500 | 1.3 | -->

## Notes
- Targets C++17; uses STL and SIMD where helpful.
- Datasets are not bundled—point the CLI to your own data files.
- Expect rapid changes while insertion and densification land.

## TradeOff between Index and Query
The index time/size and query performance trade off.
half the size and time of index can just degrade a little bit(10~20%) of query performance. How to trade off it is still a open question.
