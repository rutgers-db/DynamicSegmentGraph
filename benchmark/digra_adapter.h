/**
 * @file digra_adapter.h
 * @brief Thin wrapper to isolate DIGRA's RangeHNSW in a separate translation unit.
 *
 * This avoids namespace symbol collisions between DIGRA's hnswlib and our base_hnsw
 * by keeping DIGRA's headers out of other TUs. Exposes a minimal C++ interface.
 */

#pragma once

#include <queue>
#include <utility>

class DigraIndex {
public:
    DigraIndex(int dim,
               int initialN,
               int maxN,
               const float* baseData,  // size initialN * dim
               const int* keyList,
               const int* valueList,
               int M,
               int ef_construction);

    ~DigraIndex();

    void addPoint(int key, int value, const float* vecData);

    // Returns queue of pairs {dist, label(id)}, closer first on top like DIGRA
    std::priority_queue<std::pair<float, int>> queryRange(const float* query,
                                                           int l, int r,
                                                           int k, int ef_s);

private:
    void* impl_; // opaque RangeHNSW*
};


