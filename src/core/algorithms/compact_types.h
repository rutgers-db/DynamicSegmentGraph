#pragma once
#include <vector>
#include <limits>
#include <algorithm>

namespace Compact {

template <typename dist_t>
struct CompressedPoint {
    unsigned external_id{0};
    unsigned ll{0}, lr{0}, rl{0}, rr{0};

    CompressedPoint() = default;
    CompressedPoint(unsigned _external_id, unsigned _ll, unsigned _lr, unsigned _rl, unsigned _rr)
        : external_id(_external_id), ll(_ll), lr(_lr), rl(_rl), rr(_rr) {}

    inline bool const if_in_compressed_range(const unsigned &query_L, const unsigned &query_R) const {
        return ((ll <= query_L && query_L <= lr) && (rl <= query_R && query_R <= rr));
    }

    bool operator<(const CompressedPoint &other) const {
        return this->external_id < other.external_id;
    }
};

template <typename dist_t>
struct DirectedPointNeighbors {
    std::vector<CompressedPoint<dist_t>> nns;
    std::vector<CompressedPoint<dist_t>> rev_nns;

    size_t countNeighbors() { return nns.size() + rev_nns.size(); }
};

} // namespace Compact