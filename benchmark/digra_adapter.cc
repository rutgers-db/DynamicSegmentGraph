// Keep DIGRA headers only in this TU to avoid symbol collisions
#include "../DIGRA/TreeHNSW.hpp"
#include "digra_adapter.h"

struct DigraImpl {
    RangeHNSW* index;
    int dim;
};

DigraIndex::DigraIndex(int dim,
                       int initialN,
                       int maxN,
                       const float* baseData,
                       const int* keyList,
                       const int* valueList,
                       int M,
                       int ef_construction) {
    auto* impl = new DigraImpl();
    impl->dim = dim;
    // DIGRA ctor expects non-const pointers
    auto* dataCopy = const_cast<float*>(baseData);
    auto* keyCopy = const_cast<int*>(keyList);
    auto* valCopy = const_cast<int*>(valueList);
    impl->index = new RangeHNSW(dim, (size_t)initialN, (size_t)maxN, dataCopy, keyCopy, valCopy, M, ef_construction);
    impl_ = impl;
}

DigraIndex::~DigraIndex() {
    auto* impl = reinterpret_cast<DigraImpl*>(impl_);
    delete impl->index;
    delete impl;
}

void DigraIndex::addPoint(int key, int value, const float* vecData) {
    auto* impl = reinterpret_cast<DigraImpl*>(impl_);
    impl->index->addPoint(key, value, (char*)vecData);
}

std::priority_queue<std::pair<float, int>> DigraIndex::queryRange(const float* query,
                                                                   int l, int r,
                                                                   int k, int ef_s) {
    auto* impl = reinterpret_cast<DigraImpl*>(impl_);
    auto pq = impl->index->queryRange((float*)query, l, r, k, ef_s);
    // convert labeltype to int explicitly
    std::priority_queue<std::pair<float, int>> out;
    while (!pq.empty()) {
        out.emplace(pq.top().first, (int)pq.top().second);
        pq.pop();
    }
    return out;
}


